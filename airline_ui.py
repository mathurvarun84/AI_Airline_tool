"""Streamlit UI for the airline customer support system."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

from airline_backend import safe_airline_support

API_BASE = os.getenv("AIRLINE_API_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE.rstrip('/')}/support"
HEALTH_URL = f"{API_BASE.rstrip('/')}/health"

EXAMPLE_QUERIES = [
    ("Flight status", "What is the status of flight 6E815?"),
    ("Find flights", "Show flights from Delhi to Goa under 7000 rupees"),
    ("Baggage policy", "How much free baggage is allowed for domestic flights?"),
    ("Hybrid: delay + compensation", "Flight 6E815 is delayed — what compensation am I entitled to?"),
    ("Hybrid: cancel + refund", "My flight 6E815 was cancelled. What is the refund policy?"),
]

PATH_META = {
    "SQL": {
        "label": "Live flight data",
        "hint": "Answered from the flights database",
        "color": "#0B4F8A",
        "bg": "#E8F2FB",
    },
    "RAG": {
        "label": "Policy & FAQ",
        "hint": "Answered from the airline knowledge base",
        "color": "#9A6700",
        "bg": "#FFF8E6",
    },
    "Hybrid": {
        "label": "Hybrid SQL + RAG",
        "hint": "Combined live flight data and policy knowledge base",
        "color": "#6D28D9",
        "bg": "#F3E8FF",
    },
    "Fallback": {
        "label": "General help",
        "hint": "Outside flight data and policy scope",
        "color": "#475569",
        "bg": "#F1F5F9",
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Playfair+Display:wght@600;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.08), transparent 28%),
                radial-gradient(circle at 20% 80%, rgba(11, 79, 138, 0.06), transparent 24%),
                #F4F7FB;
        }

        .block-container {
            padding-top: 1.5rem;
            max-width: 920px;
        }

        .hero-card {
            background: linear-gradient(135deg, #0B4F8A 0%, #0E7490 100%);
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            color: #F8FAFC;
            box-shadow: 0 18px 40px rgba(11, 79, 138, 0.18);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-family: "Playfair Display", serif;
            font-size: 2rem;
            line-height: 1.1;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            font-family: "DM Sans", sans-serif;
            font-size: 0.98rem;
            opacity: 0.92;
            margin: 0;
        }

        .route-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-family: "DM Sans", sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .route-hint {
            font-family: "DM Sans", sans-serif;
            font-size: 0.82rem;
            color: #64748B;
            margin: -0.35rem 0 0.85rem 0;
        }

        .sidebar-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.85rem;
        }

        .sidebar-card h4 {
            font-family: "DM Sans", sans-serif;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748B;
            margin: 0 0 0.45rem 0;
        }

        .sidebar-card p {
            font-family: "DM Sans", sans-serif;
            margin: 0;
            color: #0F172A;
            font-size: 0.92rem;
        }

        div[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }

        div[data-testid="stChatMessage"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 0.35rem 0.2rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
            background: #EEF6FF;
            border-color: #BFDBFE;
        }

        div[data-testid="stChatInput"] {
            border-radius: 16px;
        }

        button[kind="secondary"] {
            border-radius: 999px;
        }

        #MainMenu, footer, header[data-testid="stHeader"] {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <p class="hero-title">SkyAssist Airline Support</p>
            <p class="hero-subtitle">
                Ask about live flight schedules, delays, fares, baggage rules, refunds, and more.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_api_health() -> tuple[bool, str]:
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        if response.ok:
            return True, "Connected"
        return False, f"Unavailable ({response.status_code})"
    except requests.RequestException:
        return False, "Offline"


def fetch_support(query: str) -> dict[str, Any]:
    try:
        response = requests.post(API_URL, json={"query": query}, timeout=120)
        response.raise_for_status()
        data = response.json()
        data["source"] = "api"
        return data
    except requests.exceptions.ConnectionError:
        result = safe_airline_support(query)
        result["source"] = "direct"
        return result
    except Exception as exc:
        return {
            "query": query,
            "route": "Error",
            "path": "Error",
            "response": f"Request failed: {exc}",
            "source": "error",
        }


def render_route_badge(path: str) -> None:
    meta = PATH_META.get(path, PATH_META["Fallback"])
    st.markdown(
        f"""
        <div class="route-badge" style="color:{meta['color']}; background:{meta['bg']};">
            {meta['label']}
        </div>
        <p class="route-hint">{meta['hint']}</p>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_chat() -> None:
    st.session_state.messages = []


def handle_query(query: str) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Checking flight systems and policies..."):
        result = fetch_support(query)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["response"],
            "path": result.get("path", "Fallback"),
            "source": result.get("source", "api"),
        }
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Control panel")

        api_ok, api_status = check_api_health()
        status_color = "#15803D" if api_ok else "#B45309"
        st.markdown(
            f"""
            <div class="sidebar-card">
                <h4>API status</h4>
                <p><span style="color:{status_color}; font-weight:700;">●</span> {api_status}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <h4>How answers are routed</h4>
                <p><strong>Live flight data</strong> uses SQL over the flights database.</p>
                <p style="margin-top:0.55rem;"><strong>Policy & FAQ</strong> uses the airline knowledge base.</p>
                <p style="margin-top:0.55rem;"><strong>Hybrid SQL + RAG</strong> combines both for questions that need live flight facts and policy guidance together.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Try an example")
        for label, example in EXAMPLE_QUERIES:
            if st.button(label, use_container_width=True, key=f"example_{label}"):
                handle_query(example)
                st.rerun()

        if st.button("Clear conversation", use_container_width=True):
            clear_chat()
            st.rerun()

        with st.expander("Run locally"):
            st.code("uvicorn airline_api:app --reload --port 8000", language="bash")
            st.code("streamlit run airline_ui.py --server.port 8501", language="bash")
            st.caption(
                "Set `AIRLINE_API_URL` if the API runs on another host, such as GitHub Codespaces."
            )


def render_chat_history() -> None:
    if not st.session_state.messages:
        st.info(
            "Start with a question about a flight, route, fare, delay, baggage allowance, "
            "or refund policy. Example prompts are in the sidebar."
        )
        return

    for message in st.session_state.messages:
        avatar = "✈️" if message["role"] == "assistant" else "🧳"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                render_route_badge(message.get("path", "Fallback"))
                if message.get("source") == "direct":
                    st.caption("FastAPI server not detected — answered via backend directly.")
            st.markdown(message["content"])


def main() -> None:
    st.set_page_config(
        page_title="SkyAssist Airline Support",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    init_session_state()
    render_sidebar()

    left, center, right = st.columns([0.08, 1, 0.08])
    with center:
        render_hero()
        render_chat_history()

        prompt = st.chat_input("Ask about flights, fares, delays, baggage, or refunds...")
        if prompt:
            handle_query(prompt.strip())
            st.rerun()


if __name__ == "__main__":
    main()
