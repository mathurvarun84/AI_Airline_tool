"""Streamlit UI for the airline customer support system."""

import os

import requests
import streamlit as st

from airline_backend import safe_airline_support

API_BASE = os.getenv("AIRLINE_API_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE.rstrip('/')}/support"

st.set_page_config(page_title="Airline Customer Support", page_icon="✈️")
st.title("AI-Powered Airline Customer Support")
st.caption("Ask about live flight data or airline policies and FAQs.")

user_query = st.text_area(
    "Ask a question about flights or airline policies:",
    placeholder="e.g. What is the status of flight 6E815?",
)

if st.button("Get Support", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={"query": user_query},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Route: {data['route']} | Path: {data['path']}")
                st.write(data["response"])
            except requests.exceptions.ConnectionError:
                result = safe_airline_support(user_query)
                st.info("FastAPI server not detected — using backend directly.")
                st.success(f"Route: {result['route']} | Path: {result['path']}")
                st.write(result["response"])
            except Exception as exc:
                st.error(f"Request failed: {exc}")

with st.sidebar:
    st.header("How to run")
    st.code("uvicorn airline_api:app --reload --port 8000", language="bash")
    st.code("streamlit run airline_ui.py --server.port 8501", language="bash")
    st.markdown(
        "Set `AIRLINE_API_URL` if the API runs on a different host "
        "(e.g. in GitHub Codespaces)."
    )
