"""FastAPI REST API for the airline customer support system."""

from fastapi import FastAPI
from pydantic import BaseModel

from airline_backend import safe_airline_support

app = FastAPI(
    title="Airline Customer Support API",
    description="AI-powered airline support with SQL, RAG, and guardrails.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    route: str
    path: str
    response: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/support", response_model=QueryResponse)
def support_endpoint(request: QueryRequest):
    result = safe_airline_support(request.query)
    return QueryResponse(**result)
