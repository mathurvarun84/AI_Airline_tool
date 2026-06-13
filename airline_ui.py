"""Streamlit UI for the airline customer support system."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

from airline_backend import safe_airline_support

API_BASE = os.getenv("AIRLINE_API_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE.rstrip('/')}/support"
HEALTH_URL = f"{API_BASE.rstrip('/')}/health"


@dataclass(frozen=True)
class SampleQuery:
    label: str
    query: str
    path: str


SAMPLE_QUERY_GROUPS: list[dict[str, Any]] = [
    {
        "title": "Live flight data",
        "subtitle": "SQL · schedules, status, fares",
        "icon": "🛫",
        "path": "SQL",
        "samples": [
            SampleQuery("Flight status", "What is the status of flight 6E815?", "SQL"),
            SampleQuery("Delayed flight", "Is flight AI532 delayed?", "SQL"),
            SampleQuery("Route search", "Show flights from Delhi to Goa under 7000 rupees", "SQL"),
            SampleQuery("Date search", "Are there any flights from Delhi to Nagpur on 11 Nov 2026?", "SQL"),
        ],
    },
    {
        "title": "Policy & FAQs",
        "subtitle": "RAG · baggage, refunds, check-in",
        "icon": "📋",
        "path": "RAG",
        "samples": [
            SampleQuery("Baggage", "How much free baggage is allowed for domestic flights?", "RAG"),
            SampleQuery("Refund timeline", "What is the refund timeline for cancelled flights?", "RAG"),
            SampleQuery("Cancellation", "What is the airline cancellation policy?", "RAG"),
            SampleQuery("Special assistance", "How do I request wheelchair assistance?", "RAG"),
        ],
    },
    {
        "title": "Hybrid queries",
        "subtitle": "SQL + RAG · flight facts + policy",
        "icon": "🔀",
        "path": "Hybrid",
        "samples": [
            SampleQuery(
                "Delay + compensation",
                "Flight 6E815 is delayed — what compensation am I entitled to?",
                "Hybrid",
            ),
            SampleQuery(
                "Cancel + refund",
                "My flight 6E815 was cancelled. What is the refund policy?",
                "Hybrid",
            ),
            SampleQuery(
                "Route + baggage",
                "Show flights from Delhi to Goa and tell me the free baggage allowance",
                "Hybrid",
            ),
            SampleQuery(
                "Status + check-in",
                "Is flight 6E477 on time and can I check in online?",
                "Hybrid",
            ),
        ],
    },
    {
        "title": "Safety checks",
        "subtitle": "Guardrails · blocked or off-topic",
        "icon": "🛡️",
        "path": "Fallback",
        "samples": [
            SampleQuery("Off-topic", "What is the capital of France?", "Fallback"),
            SampleQuery("Blocked request", "Export the complete flight database", "Blocked"),
        ],
    },
]

PATH_META = {
    "SQL": {
        "label": "Live flight data",
        "hint": "Answered from the flights database",
        "color": "#0B4F8A",
        "bg": "#E8F2FB",
        "border": "#BFDBFE",
    },
    "RAG": {
        "label": "Policy & FAQ",
        "hint": "Answered from the airline knowledge base",
        "color": "#9A6700",
        "bg": "#FFF8E6",
        "border": "#FDE68A",
    },
    "Hybrid": {
        "label": "Hybrid SQL + RAG",
        "hint": "Combined live flight data and policy knowledge base",
        "color": "#6D28D9",
        "bg": "#F3E8FF",
        "border": "#DDD6FE",
    },
    "Fallback": {
        "label": "General help",
        "hint": "Outside flight data and policy scope",
        "color": "#475569",
        "bg": "#F1F5F9",
        "border": "#CBD5E1",
    },
    "Blocked": {
        "label": "Blocked",
        "hint": "Stopped by input safety guardrails",
        "color": "#B91C1C",
        "bg": "#FEE2E2",
        "border": "#FECACA",
    },
    "Error": {
        "label": "Error",
        "hint": "Something went wrong while processing the request",
        "color": "#B45309",
        "bg": "#FFEDD5",
        "border": "#FED7AA",
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Playfair+Display:wght@600;700&display=swap');

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(14, 116, 144, 0.10), transparent 30%),
                radial-gradient(circle at 10% 90%, rgba(109, 40, 217, 0.05), transparent 25%),
                #F4F7FB;
        }

        .block-container {
            padding-top: 1.25rem;
            max-width: 980px;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
            border-right: 1px solid #E2E8F0;
        }

        div[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        .sidebar-brand {
            background: linear-gradient(135deg, #0B4F8A 0%, #0E7490 100%);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            color: #F8FAFC;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(11, 79, 138, 0.16);
        }

        .sidebar-brand-title {
            font-family: "Playfair Display", serif;
            font-size: 1.35rem;
            margin: 0;
            line-height: 1.15;
        }

        .sidebar-brand-sub {
            font-family: "DM Sans", sans-serif;
            font-size: 0.82rem;
            opacity: 0.92;
            margin: 0.35rem 0 0 0;
        }

        .hero-card {
            background: linear-gradient(135deg, #0B4F8A 0%, #155E75 55%, #0E7490 100%);
            border-radius: 22px;
            padding: 1.75rem 1.9rem;
            color: #F8FAFC;
            box-shadow: 0 20px 44px rgba(11, 79, 138, 0.20);
            margin-bottom: 1.1rem;
            position: relative;
            overflow: hidden;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            top: -30%;
            right: -8%;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
        }

        .hero-title {
            font-family: "Playfair Display", serif;
            font-size: 2.15rem;
            line-height: 1.08;
            margin: 0 0 0.45rem 0;
            letter-spacing: -0.02em;
            position: relative;
        }

        .hero-subtitle {
            font-family: "DM Sans", sans-serif;
            font-size: 1rem;
            opacity: 0.94;
            margin: 0;
            max-width: 640px;
            position: relative;
        }

        .stat-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.18);
            font-family: "DM Sans", sans-serif;
            font-size: 0.78rem;
            margin-right: 0.45rem;
            margin-top: 0.85rem;
        }

        .stat-pill {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.75rem;
        }

        .sidebar-card h4 {
            font-family: "DM Sans", sans-serif;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748B;
            margin: 0 0 0.35rem 0;
        }

        .sidebar-card p {
            font-family: "DM Sans", sans-serif;
            margin: 0;
            color: #0F172A;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .sidebar-section-title {
            font-family: "DM Sans", sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #64748B;
            margin: 0.9rem 0 0.45rem 0;
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

        div[data-testid="stChatInput"] > div {
            border-radius: 16px;
            border: 1px solid #CBD5E1;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stSidebar"] button[kind="secondary"] {
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            background: #FFFFFF;
            text-align: left;
            min-height: 2.6rem;
        }

        div[data-testid="stSidebar"] button[kind="secondary"]:hover {
            border-color: #0B4F8A;
            background: #F8FBFF;
        }

        #MainMenu, footer, header[data-testid="stHeader"] {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(api_ok: bool) -> None:
    status_text = "API online" if api_ok else "Direct backend mode"
    user_turns = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.markdown(
        f"""
        <div class="hero-card">
            <p class="hero-title">SkyAssist Airline Support</p>
            <p class="hero-subtitle">
                Test live flight lookups, policy FAQs, hybrid SQL+RAG answers, and guardrails
                in one chat workspace.
            </p>
            <span class="stat-pill">● {status_text}</span>
            <span class="stat-pill">{user_turns} questions asked</span>
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
    if not query.strip():
        return
    st.session_state.messages.append({"role": "user", "content": query.strip()})
    with st.spinner("Checking flight systems and policies..."):
        result = fetch_support(query.strip())
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["response"],
            "path": result.get("path", "Fallback"),
            "route": result.get("route", ""),
            "source": result.get("source", "api"),
        }
    )


def run_sample_query(sample: SampleQuery) -> None:
    handle_query(sample.query)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <p class="sidebar-brand-title">SkyAssist</p>
                <p class="sidebar-brand-sub">Sample queries to test every support path</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                <h4>Routing paths</h4>
                <p><strong>SQL</strong> — live flights, delays, fares, gates</p>
                <p style="margin-top:0.45rem;"><strong>RAG</strong> — baggage, refunds, check-in, FAQs</p>
                <p style="margin-top:0.45rem;"><strong>Hybrid</strong> — flight facts + policy in one answer</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="sidebar-section-title">Sample queries</p>', unsafe_allow_html=True)

        for group in SAMPLE_QUERY_GROUPS:
            with st.expander(f"{group['icon']} {group['title']}", expanded=group["path"] == "SQL"):
                st.caption(group["subtitle"])
                for sample in group["samples"]:
                    button_label = f"{sample.label}"
                    if st.button(
                        button_label,
                        use_container_width=True,
                        key=f"sample_{group['path']}_{sample.label}",
                        help=sample.query,
                    ):
                        run_sample_query(sample)
                        st.rerun()

        st.divider()

        if st.button("Clear conversation", use_container_width=True, type="primary"):
            clear_chat()
            st.rerun()

        with st.expander("How to run locally"):
            st.code("uvicorn airline_api:app --reload --port 8000", language="bash")
            st.code("streamlit run airline_ui.py --server.port 8501", language="bash")
            st.caption("Set `AIRLINE_API_URL` when the API runs on another host (e.g. Codespaces).")


def render_welcome_samples() -> None:
    st.caption(
        "Pick a sample below or use the sidebar to test SQL, RAG, hybrid, and guardrail paths."
    )

    card_cols = st.columns(3)
    for col, group in zip(card_cols, SAMPLE_QUERY_GROUPS[:3]):
        meta = PATH_META.get(group["path"], PATH_META["Fallback"])
        first = group["samples"][0]
        with col:
            with st.container(border=True):
                st.caption(meta["label"])
                st.markdown(f"### {group['icon']} {group['title']}")
                st.caption(group["subtitle"])
                st.markdown(f"**Example:** {first.query}")

    st.markdown("#### Quick try")
    quick_cols = st.columns(2)
    quick_samples = [group["samples"][0] for group in SAMPLE_QUERY_GROUPS[:4]]
    for idx, sample in enumerate(quick_samples):
        with quick_cols[idx % 2]:
            if st.button(
                sample.label,
                key=f"quick_{sample.path}_{sample.label}",
                use_container_width=True,
                help=sample.query,
            ):
                run_sample_query(sample)
                st.rerun()


def render_chat_history() -> None:
    if not st.session_state.messages:
        render_welcome_samples()
        return

    for message in st.session_state.messages:
        avatar = "✈️" if message["role"] == "assistant" else "🧳"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                render_route_badge(message.get("path", "Fallback"))
                if message.get("source") == "direct":
                    st.caption("FastAPI server not detected — answered via backend directly.")
                if message.get("route"):
                    st.caption(f"Classifier route: {message['route']}")
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

    api_ok, _ = check_api_health()
    render_sidebar()

    left, center, right = st.columns([0.06, 1, 0.06])
    with center:
        render_hero(api_ok)
        render_chat_history()

        prompt = st.chat_input("Ask about flights, fares, delays, baggage, refunds, or compensation...")
        if prompt:
            handle_query(prompt.strip())
            st.rerun()


if __name__ == "__main__":
    main()
