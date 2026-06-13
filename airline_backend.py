"""Airline customer support backend for FastAPI / Streamlit."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

try:
    from google.colab import userdata
except ImportError:
    userdata = None

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = _BASE_DIR / "Knowledge_Base_for_Airline_Info_and_FAQs.pdf"
PDF_URL = (
    "https://raw.githubusercontent.com/MLOPS-test/Artifacts/refs/heads/main/"
    "datasets/Knowledge_Base_for_Airline_Info_and_FAQs.pdf"
)


def _load_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    if userdata is not None:
        try:
            return userdata.get(name)
        except Exception:
            return None
    return None


os.environ["OPENAI_API_KEY"] = _load_secret("OPENAI_API_KEY") or ""
os.environ["PINECONE_API_KEY"] = _load_secret("PINECONE_API_KEY") or ""
os.environ.setdefault("SUPABASE_DB_HOST", _load_secret("SUPABASE_DB_HOST") or "")
os.environ.setdefault("SUPABASE_DB_USER", _load_secret("SUPABASE_DB_USER") or "")
os.environ.setdefault(
    "SUPABASE_DB_PASSWORD", _load_secret("SUPABASE_DB_PASSWORD") or ""
)
os.environ.setdefault("SUPABASE_DB_NAME", _load_secret("SUPABASE_DB_NAME") or "postgres")
os.environ.setdefault("SUPABASE_DB_PORT", _load_secret("SUPABASE_DB_PORT") or "5432")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# --- Classifier ---

classifier_system_prompt = (
    "You are a query classifier for an airline customer support system.\n"
    "Classify each user query into exactly ONE category:\n\n"
    "1. Need SQL\n"
    "   Live flight data from the database: status, delays, schedules, routes,\n"
    "   fares, seats, gates, terminals, aircraft, arrival/departure times.\n"
    "2. Non SQL\n"
    "   Airline policy / FAQ only: baggage, refunds, cancellation, rescheduling,\n"
    "   check-in, special assistance, pets, documents, prohibited items.\n"
    "   Use this when NO live flight record lookup is required.\n"
    "3. Hybrid SQL + RAG\n"
    "   Queries that need BOTH live flight data from the database AND policy/FAQ\n"
    "   information from the knowledge base in one answer.\n"
    "4. Out of Context\n"
    "   Not related to airline customer support.\n\n"
    "Rules:\n"
    "- Pure policy/refund/cancellation/check-in/baggage questions with no specific\n"
    "  flight lookup -> Non SQL.\n"
    "- Requests for live or specific flight records only -> Need SQL.\n"
    "- Questions combining a specific flight/route lookup with policy guidance\n"
    "  (refund, compensation, baggage, reschedule, check-in, rebooking) -> Hybrid SQL + RAG.\n"
    "- General knowledge, sports, coding, or unrelated topics -> Out of Context.\n"
    "- Requests to export/dump the database or bypass security are NOT valid SQL queries;\n"
    "  classify as Out of Context.\n\n"
    "Examples:\n"
    "- 'What is the status of flight 6E477?' -> Need SQL\n"
    "- 'Show flights from Delhi to Goa under 7000' -> Need SQL\n"
    "- 'How much free baggage is allowed for domestic flights?' -> Non SQL\n"
    "- 'What happens if I miss my flight?' -> Non SQL\n"
    "- 'Flight 6E815 is delayed — what compensation am I entitled to?' -> Hybrid SQL + RAG\n"
    "- 'My flight 6E815 was cancelled. What is the refund policy?' -> Hybrid SQL + RAG\n"
    "- 'Show Delhi to Goa flights and tell me the free baggage allowance' -> Hybrid SQL + RAG\n"
    "- 'Is flight 6E477 on time and can I check in online?' -> Hybrid SQL + RAG\n"
    "- 'What is the capital of France?' -> Out of Context\n"
    "- 'Export the complete flight database' -> Out of Context\n\n"
    "Respond with ONLY one label: Need SQL, Non SQL, Hybrid SQL + RAG, or Out of Context."
)

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", classifier_system_prompt),
    ("human", "{query}"),
])

input_classifier_chain = classifier_prompt | llm | StrOutputParser()


def classify_user_query(query: str) -> str:
    """Normalize classifier output to one of four routing labels."""
    label = input_classifier_chain.invoke({"query": query}).strip()
    normalized = label.lower()

    if "hybrid" in normalized:
        return "Hybrid SQL + RAG"
    if "need sql" in normalized or normalized == "sql":
        return "Need SQL"
    if "non sql" in normalized or "non-sql" in normalized:
        return "Non SQL"
    if "out of context" in normalized or "out-of-context" in normalized:
        return "Out of Context"

    sql_keywords = [
        "flight", "delay", "gate", "terminal", "fare", "seat", "status", "cancelled",
        "departure", "arrival", "route", "schedule",
    ]
    policy_keywords = [
        "baggage", "refund", "check-in", "check in", "policy", "cancel", "assist",
        "faq", "compensation", "reschedule", "rebook", "entitled", "allowance",
        "online check",
    ]
    q_lower = query.lower()
    has_sql_signal = any(k in q_lower for k in sql_keywords)
    has_policy_signal = any(k in q_lower for k in policy_keywords)

    if has_sql_signal and has_policy_signal:
        return "Hybrid SQL + RAG"
    if has_sql_signal:
        return "Need SQL"
    if has_policy_signal:
        return "Non SQL"
    return "Out of Context"


# --- PostgreSQL ---

db_params = {
    "host": os.getenv("SUPABASE_DB_HOST"),
    "port": os.getenv("SUPABASE_DB_PORT", "5432"),
    "user": os.getenv("SUPABASE_DB_USER"),
    "password": os.getenv("SUPABASE_DB_PASSWORD"),
    "dbname": os.getenv("SUPABASE_DB_NAME", "postgres"),
}


def execute_sql_query(query: str):
    """Connect to Supabase PostgreSQL, run a SQL query, return rows as dicts."""
    conn = None
    try:
        conn = psycopg2.connect(**db_params, sslmode="require")
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if conn:
            conn.close()


# --- SQL generation ---

FLIGHTS_TABLE_SCHEMA = (
    "Table: flights\n"
    "Columns:\n"
    "- id (BIGINT, primary key)\n"
    "- flight_no (TEXT)\n"
    "- airline_code (TEXT)\n"
    "- airline_name (TEXT)\n"
    "- origin (TEXT) -- IATA airport code\n"
    "- destination (TEXT) -- IATA airport code\n"
    "- departure_date (DATE)\n"
    "- departure_time (TIME)\n"
    "- arrival_date (DATE)\n"
    "- arrival_time (TIME)\n"
    "- status (TEXT) -- On Time, Delayed, Cancelled\n"
    "- delay_minutes (INTEGER)\n"
    "- delay_reason (TEXT)\n"
    "- terminal (TEXT)\n"
    "- gate (TEXT)\n"
    "- aircraft_type (TEXT)\n"
    "- seats_total (INTEGER)\n"
    "- seats_booked (INTEGER)\n"
    "- fare_inr (INTEGER)"
)

CITY_TO_AIRPORT = (
    "Map common city names to IATA codes when filtering origin/destination:\n"
    "Delhi->DEL, Mumbai->BOM, Bengaluru/Bangalore->BLR, Chennai->MAA,\n"
    "Hyderabad->HYD, Kolkata->CCU, Pune->PNQ, Goa->GOI, Nagpur->NAG,\n"
    "Jaipur->JAI, Varanasi->VNS, Kochi->COK, Ahmedabad->AMD"
)

sql_system_prompt = (
    "You are a PostgreSQL expert for an airline support system.\n"
    "Generate a single valid SELECT query for the flights table.\n\n"
    f"{FLIGHTS_TABLE_SCHEMA}\n\n"
    f"{CITY_TO_AIRPORT}\n\n"
    "Rules:\n"
    "- ONLY generate read-only SELECT queries\n"
    "- Always query FROM flights\n"
    "- Use exact column names from the schema\n"
    "- Return ONLY the SQL query (no markdown, no explanation)\n"
    "- Convert user dates to ISO format YYYY-MM-DD in WHERE clauses\n"
    "- Match flight_no exactly as provided (e.g., '6E477')\n"
    "- For available seats use (seats_total - seats_booked)\n"
    "- For evening flights use departure_time >= '17:00:00'\n"
    "- For fare filters use fare_inr (e.g., fare_inr < 7000)\n"
    "- For search/list queries add LIMIT 20\n"
    "- Select only columns needed to answer the question\n\n"
    "Examples:\n"
    "Q: What is the status of flight 6E815?\n"
    "SQL: SELECT flight_no, status, delay_minutes, delay_reason FROM flights "
    "WHERE flight_no = '6E815';\n\n"
    "Q: Flights from Delhi to Nagpur on 2026-11-11\n"
    "SQL: SELECT flight_no, departure_time, status, fare_inr FROM flights "
    "WHERE origin = 'DEL' AND destination = 'NAG' AND departure_date = '2026-11-11' LIMIT 20;\n\n"
    "Q: Flights under 7000 rupees from Delhi to Goa\n"
    "SQL: SELECT flight_no, origin, destination, fare_inr, departure_time FROM flights "
    "WHERE origin = 'DEL' AND destination = 'GOI' AND fare_inr < 7000 LIMIT 20;"
)

sql_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", sql_system_prompt),
    ("human", "{question}"),
])

sql_query_chain = sql_generation_prompt | llm | StrOutputParser()


def clean_sql_output(raw_sql: str) -> str:
    """Remove ```sql fences so PostgreSQL receives clean executable SQL."""
    sql_text = raw_sql.strip()
    sql_text = re.sub(r"^```sql\s*", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"^```\s*", "", sql_text)
    sql_text = re.sub(r"```$", "", sql_text)
    return sql_text.strip().rstrip(";") + ";"


FORBIDDEN_SQL_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
]


def validate_sql_query(query: str) -> tuple[bool, str]:
    """Guardrail: allow only safe read-only SELECT queries."""
    if not query or not query.strip():
        return False, "Empty SQL query."

    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        return False, "Only SELECT queries are allowed."

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalized):
            return False, f"Forbidden SQL keyword detected: {keyword}"

    return True, "OK"


@tool
def run_sql_query_tool(sql_query: str) -> str:
    """Execute a read-only SQL SELECT query on the flights table and return JSON results."""
    is_valid, reason = validate_sql_query(sql_query)
    if not is_valid:
        return json.dumps({"error": reason})

    results = execute_sql_query(sql_query)
    return json.dumps(results, default=str)


sql_tools = [run_sql_query_tool]
sql_agent = create_react_agent(llm, sql_tools)

sql_summarize_system_prompt = (
    "You are FlightAI, a polite airline customer support agent.\n"
    "Summarize database results for the customer using ONLY the provided query results.\n"
    "Do not invent flight data.\n\n"
    "If no rows were returned, clearly say no matching flight information was found\n"
    "and suggest checking the flight number, route, or date.\n\n"
    "When data is available, include relevant fields such as:\n"
    "flight_no, status, origin, destination, departure/arrival date and time,\n"
    "delay_minutes, delay_reason, terminal, gate, available seats, and fare_inr.\n\n"
    "Keep the answer concise, accurate, and customer-friendly."
)

sql_summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", sql_summarize_system_prompt),
    (
        "human",
        "Customer question: {user_query}\n"
        "SQL executed: {sql_query}\n"
        "Database results (JSON): {results}\n\n"
        "Write the customer-facing answer:",
    ),
])

sql_summarize_chain = sql_summarize_prompt | llm | StrOutputParser()


def run_sql_pipeline(user_query: str) -> str:
    """Generate SQL, validate it, execute via tool, and return a user-friendly answer."""
    generated_sql = clean_sql_output(sql_query_chain.invoke({"question": user_query}))
    is_valid, reason = validate_sql_query(generated_sql)
    if not is_valid:
        return f"Sorry, I cannot run that database request safely. Reason: {reason}"

    raw_results = run_sql_query_tool.invoke({"sql_query": generated_sql})
    return sql_summarize_chain.invoke({
        "user_query": user_query,
        "sql_query": generated_sql,
        "results": raw_results,
    })


# --- RAG setup ---

PINECONE_INDEX_NAME = "airline-faq-index"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def _ensure_pdf() -> Path:
    if PDF_PATH.exists():
        return PDF_PATH
    try:
        import urllib.request

        logger.info("Downloading FAQ PDF to %s", PDF_PATH)
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    except Exception as exc:
        raise FileNotFoundError(
            f"FAQ PDF not found at {PDF_PATH}. Download it manually from {PDF_URL}"
        ) from exc
    return PDF_PATH


def _build_retriever():
    pdf_file = _ensure_pdf()
    loader = PyMuPDFLoader(str(pdf_file))
    documents = loader.load()
    logger.info("Loaded %s pages from the knowledge base PDF.", len(documents))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = text_splitter.split_documents(documents)
    logger.info("Created %s text chunks.", len(chunks))

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing_indexes = {idx.name for idx in pc.list_indexes()}

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        logger.info("Created Pinecone index: %s", PINECONE_INDEX_NAME)

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    vector_count = pinecone_index.describe_index_stats().total_vector_count

    if vector_count == 0:
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            index_name=PINECONE_INDEX_NAME,
        )
        logger.info("Embeddings uploaded to Pinecone.")
    else:
        logger.info(
            "Pinecone index already contains %s vectors — skipping upload.", vector_count
        )

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_model,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


retriever = _build_retriever()


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


augmentation_template = (
    "You are FlightAI, a helpful airline customer support agent.\n"
    "Answer the customer question using ONLY the provided context from the\n"
    "airline knowledge base.\n\n"
    "Rules:\n"
    "- Do not guess or invent policies, fees, or rules not stated in the context.\n"
    "- If the answer is not in the context, say you do not have that information\n"
    "  in the airline knowledge base and suggest contacting support.\n"
    "- Answer all parts of multi-part questions when the context supports them.\n"
    "- Use short paragraphs or bullet points for clarity.\n"
    "- Be polite, professional, and customer-friendly.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

augmentation_prompt = ChatPromptTemplate.from_template(augmentation_template)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | augmentation_prompt
    | llm
    | StrOutputParser()
)


def retrieve_rag_context(question: str) -> str:
    """Return formatted knowledge-base context for a query."""
    docs = retriever.invoke(question)
    return format_docs(docs)


# --- Hybrid SQL + RAG ---
#
# Hybrid routing goes beyond simple either/or classification. Some customer questions
# need live flight records (PostgreSQL) and airline policy text (Pinecone RAG) in the
# same answer — for example, confirming a delay from the database and explaining
# compensation rules from the FAQ knowledge base.
#
# Processing flow:
#   1. Classifier labels the query as "Hybrid SQL + RAG"
#   2. SQL sub-pipeline generates and executes a read-only SELECT query
#   3. RAG retriever fetches relevant FAQ/policy chunks from Pinecone
#   4. A synthesis LLM merges both sources into one customer-facing response
#
# Example hybrid queries (SQL part | RAG part):
#   - "Flight 6E815 is delayed — what compensation am I entitled to?"
#       SQL: status, delay_minutes, delay_reason for flight 6E815
#       RAG: delay compensation / passenger rights policy
#   - "My flight 6E815 was cancelled. What is the refund policy?"
#       SQL: cancellation status and flight details for 6E815
#       RAG: refund policy for cancelled flights
#   - "Show Delhi to Goa flights and tell me the free baggage allowance"
#       SQL: matching flights, fares, departure times on DEL->GOI
#       RAG: free baggage allowance rules
#   - "Is flight 6E477 on time and can I check in online?"
#       SQL: on-time/delay status for 6E477
#       RAG: online check-in policy and window
#   - "I'm on flight 6E815 which is delayed — can I reschedule for free?"
#       SQL: delay details for 6E815
#       RAG: rescheduling / rebooking policy after delays

HYBRID_EXAMPLE_QUERIES = [
    {
        "query": "Flight 6E815 is delayed — what compensation am I entitled to?",
        "sql_part": "Look up status, delay_minutes, and delay_reason for flight 6E815.",
        "rag_part": "Retrieve delay compensation and passenger entitlement policy.",
    },
    {
        "query": "My flight 6E815 was cancelled. What is the refund policy?",
        "sql_part": "Confirm cancellation status and flight details for 6E815.",
        "rag_part": "Retrieve refund policy for cancelled flights.",
    },
    {
        "query": "Show flights from Delhi to Goa and tell me the free baggage allowance.",
        "sql_part": "Search DEL->GOI flights with fare and schedule fields.",
        "rag_part": "Retrieve free baggage allowance rules.",
    },
    {
        "query": "Is flight 6E477 on time and can I check in online?",
        "sql_part": "Fetch on-time/delay status and schedule for 6E477.",
        "rag_part": "Retrieve online check-in policy and time window.",
    },
    {
        "query": "I'm on flight 6E815 which is delayed — can I reschedule for free?",
        "sql_part": "Fetch delay details for flight 6E815.",
        "rag_part": "Retrieve rescheduling/rebooking policy after delays.",
    },
]

hybrid_synthesis_system_prompt = (
    "You are FlightAI, a polite airline customer support agent.\n"
    "Answer the customer using BOTH sources below:\n"
    "1. Live flight database results (structured, real-time flight records)\n"
    "2. Airline knowledge-base excerpts (policies, FAQs, rules)\n\n"
    "Rules:\n"
    "- Clearly address every part of the customer's question.\n"
    "- Use database results for live flight facts only; do not invent flight data.\n"
    "- Use knowledge-base context for policy/FAQ guidance only; do not invent policies.\n"
    "- If one source lacks information, say so and answer from the other source.\n"
    "- Organize the reply with short sections or bullet points when helpful.\n"
    "- Keep the tone concise, accurate, and customer-friendly."
)

hybrid_synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", hybrid_synthesis_system_prompt),
    (
        "human",
        "Customer question: {user_query}\n\n"
        "SQL executed: {sql_query}\n"
        "Database results (JSON): {sql_results}\n\n"
        "Knowledge-base context:\n{rag_context}\n\n"
        "Write one unified customer-facing answer:",
    ),
])

hybrid_synthesis_chain = hybrid_synthesis_prompt | llm | StrOutputParser()


def run_hybrid_pipeline(user_query: str) -> dict[str, str]:
    """Fetch SQL results and RAG context, then synthesize a single hybrid answer."""
    generated_sql = clean_sql_output(sql_query_chain.invoke({"question": user_query}))
    is_valid, reason = validate_sql_query(generated_sql)
    if not is_valid:
        sql_results = json.dumps({"error": reason})
    else:
        sql_results = run_sql_query_tool.invoke({"sql_query": generated_sql})

    rag_context = retrieve_rag_context(user_query)
    response = hybrid_synthesis_chain.invoke({
        "user_query": user_query,
        "sql_query": generated_sql,
        "sql_results": sql_results,
        "rag_context": rag_context,
    })
    return {
        "response": response,
        "sql_query": generated_sql,
        "sql_results": sql_results,
        "rag_context": rag_context,
    }


# --- Fallback ---

fallback_system_prompt = (
    "You are FlightAI, a polite airline customer support assistant.\n"
    "The user's question is outside airline support scope.\n\n"
    "Rules:\n"
    "- Do NOT answer the off-topic question itself.\n"
    "- Briefly explain that you only help with airline-related topics.\n"
    "- Mention you can help with:\n"
    "  * Live flight information (status, delays, gates, fares, seats, schedules)\n"
    "  * Airline policies and FAQs (baggage, refunds, check-in, cancellation,\n"
    "    rescheduling, special assistance)\n"
    "- Keep the response to 2-3 sentences."
)

fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", fallback_system_prompt),
    ("human", "{query}"),
])

fallback_chain = fallback_prompt | llm | StrOutputParser()


# --- Orchestrator ---

def airline_support_system(user_query: str) -> dict:
    """Main orchestration function: classify, route, and respond."""
    route = classify_user_query(user_query)

    if route == "Need SQL":
        response = run_sql_pipeline(user_query)
        path = "SQL"
        result = {
            "query": user_query,
            "route": route,
            "path": path,
            "response": response,
        }
    elif route == "Hybrid SQL + RAG":
        hybrid_result = run_hybrid_pipeline(user_query)
        result = {
            "query": user_query,
            "route": route,
            "path": "Hybrid",
            "response": hybrid_result["response"],
            "sql_query": hybrid_result["sql_query"],
        }
    elif route == "Non SQL":
        response = rag_chain.invoke(user_query)
        path = "RAG"
        result = {
            "query": user_query,
            "route": route,
            "path": path,
            "response": response,
        }
    else:
        response = fallback_chain.invoke({"query": user_query})
        path = "Fallback"
        result = {
            "query": user_query,
            "route": route,
            "path": path,
            "response": response,
        }

    return result


def get_airline_support_response(user_query: str) -> str:
    return airline_support_system(user_query)["response"]


# --- Guardrails ---

INPUT_BLOCKED_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "system prompt",
    "reveal your prompt",
    "show your prompt",
    "drop table",
    "delete from",
    "update flights",
    "insert into",
    "export the complete",
    "export complete",
    "dump database",
    "dump the database",
    "all customer records",
    "all records in the database",
    "bypass airport security",
    "bypass security",
    "api_key",
    "api key",
    "password",
    "secret key",
]

OUTPUT_BLOCKED_PATTERNS = [
    "api_key",
    "api key",
    "password",
    "secret key",
    "system prompt",
    "database password",
    "connection string",
]


def input_guardrail(user_query: str) -> tuple[bool, str]:
    """Validate user input before processing."""
    if not user_query or not user_query.strip():
        return False, "Please enter a valid question."

    if len(user_query) > 1500:
        return False, "Your message is too long. Please shorten it and try again."

    lowered = user_query.lower()
    for pattern in INPUT_BLOCKED_PATTERNS:
        if pattern in lowered:
            return False, "Your request was blocked by input safety guardrails."

    return True, user_query.strip()


def output_guardrail(response: str) -> str:
    """Validate model output before showing it to the user."""
    if not response:
        return "Sorry, I could not generate a response. Please try again."

    lowered = response.lower()
    for pattern in OUTPUT_BLOCKED_PATTERNS:
        if pattern in lowered:
            return "Sorry, I cannot share sensitive or unsafe information."

    return response.strip()


def safe_airline_support(user_query: str) -> dict:
    """End-to-end support flow with input/output guardrails."""
    allowed, checked = input_guardrail(user_query)
    if not allowed:
        return {
            "query": user_query,
            "route": "Blocked",
            "path": "Input Guardrail",
            "response": checked,
        }

    result = airline_support_system(checked)
    result["response"] = output_guardrail(result["response"])
    return result
