# AI Airline Customer Support

AI-powered airline customer support with query classification, SQL over live flight data, RAG over airline FAQs, guardrails, FastAPI, and Streamlit.

## Architecture

- **SQL path** — live flight queries via Supabase PostgreSQL
- **RAG path** — policy/FAQ answers via Pinecone + PDF knowledge base
- **Fallback** — off-topic queries
- **Guardrails** — input/output safety checks

## Setup (local or GitHub Codespaces)

1. Clone the repo and open in Codespaces (or use locally).
2. Copy environment variables:

   ```bash
   cp .env.example .env
   ```

3. Fill in `.env` with your API keys and Supabase credentials.
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Test database connectivity:

   ```bash
   python test_db_connection.py
   ```

The FAQ PDF is downloaded automatically on first run from the course artifact URL.

## Run the API

```bash
uvicorn airline_api:app --reload --host 0.0.0.0 --port 8000
```

Health check: `GET http://localhost:8000/health`  
Support: `POST http://localhost:8000/support` with JSON `{"query": "..."}`

## Run the Streamlit UI

In a second terminal:

```bash
streamlit run airline_ui.py --server.port 8501 --server.address 0.0.0.0
```

The UI calls the FastAPI backend. If the API is not running, it falls back to calling `safe_airline_support` directly.

## GitHub Codespaces

The `.devcontainer` config installs dependencies on create and forwards ports **8000** (API) and **8501** (Streamlit).

1. Start the API in one terminal.
2. Start Streamlit in another.
3. Open the forwarded ports from the **Ports** tab.

If using a forwarded API URL in Codespaces, set:

```bash
export AIRLINE_API_URL=https://<your-codespace>-8000.app.github.dev
```

## Project layout

| File | Purpose |
|------|---------|
| `airline_backend.py` | Classifier, SQL pipeline, RAG, guardrails |
| `airline_api.py` | FastAPI REST API |
| `airline_ui.py` | Streamlit web interface |
| `test_db_connection.py` | Supabase connectivity check |

## Example queries

- **SQL:** `What is the status of flight 6E815?`
- **RAG:** `How much free baggage is allowed for domestic flights?`
- **Guardrail:** `Export the complete flight database`
