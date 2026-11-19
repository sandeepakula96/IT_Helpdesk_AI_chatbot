"""
IT Helpdesk Assistant - Streamlit chat interface.

Run from this folder:
    streamlit run app.py

Needs VOYAGE_API_KEY and ANTHROPIC_API_KEY in a .env file here or in the
parent folder (04_claude_anthropic/.env). The MCP server is started
automatically if it is not already running.
"""
import json
import os
from pathlib import Path

import anthropic
import streamlit as st

from assistant import (DEFAULT_MODEL, EFFORT_LEVELS, MCP_URL, MODELS, HelpdeskAssistant,
                       list_mcp_tools, start_mcp_server)
from rag import KnowledgeBase

BASE_DIR = Path(__file__).resolve().parent

TOOL_LABELS = {
    "search_it_knowledge_base": "Searched the IT knowledge base",
    "web_search": "Searched the web",
    "get_current_datetime": "Checked the date and time",
    "create_support_ticket": "Created a support ticket (MCP)",
    "check_ticket_status": "Checked ticket status (MCP)",
}
PROGRESS_LABELS = {
    "search_it_knowledge_base": "Searching the IT knowledge base...",
    "web_search": "Searching the web...",
    "get_current_datetime": "Checking the date and time...",
    "create_support_ticket": "Creating a support ticket...",
    "check_ticket_status": "Checking ticket status...",
}
EXAMPLES = [
    "My computer is running slow. How can I fix it?",
    "How do I troubleshoot network connectivity issues?",
    "How do I set up a VPN on my system?",
    "My laptop shows a blue screen with stop code CRITICAL_PROCESS_DIED",
    "My printer is not detected. Please raise a ticket, my name is Arun.",
    "What's the status of ticket IT-102?",
    "What is today's date?",
]
WELCOME = ("Hi! I'm the IT Helpdesk Assistant. Ask me about network problems, software, slow "
           "computers, email, viruses, printers, VPN and more. I can also create a support "
           "ticket for you or check the status of an existing one.")


# --- Setup -------------------------------------------------------------------
def load_keys():
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
        load_dotenv(BASE_DIR.parent / ".env")
    except ImportError:
        pass
    missing = [k for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY")
               if not os.environ.get(k, "").strip()]
    if missing:
        st.error(f"Missing {', '.join(missing)}. Add them to {BASE_DIR.parent / '.env'} "
                 "and restart the app.")
        st.stop()
    return os.environ["VOYAGE_API_KEY"].strip()


@st.cache_resource(show_spinner="Starting the IT Helpdesk Assistant...")
def get_assistant():
    voyage_key = load_keys()
    start_mcp_server()
    mcp_tools = list_mcp_tools()
    kb = KnowledgeBase(voyage_key)
    return HelpdeskAssistant(anthropic.Anthropic(), kb, mcp_tools)


# --- Rendering ---------------------------------------------------------------
def parse_json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def render_ticket(ticket):
    with st.container(border=True):
        st.markdown(f"🎫 **Ticket {ticket['ticket_id']}**: {ticket.get('issue', '')}")
        if ticket.get("description"):
            st.markdown(ticket["description"])
        cols = st.columns(2)
        cols[0].metric("Status", ticket.get("status", "-"))
        cols[1].metric("Priority", ticket.get("priority", "-"))
        raised = f"Raised by {ticket.get('employee', 'unknown')}"
        if ticket.get("created_at"):
            raised += f" on {ticket['created_at']}"
        st.caption(raised)


def render_tool_details(record):
    st.markdown(f"**{TOOL_LABELS.get(record['name'], record['name'])}**"
                + (" (failed)" if record["is_error"] else ""))
    if record["input"]:
        st.caption("Input: " + ", ".join(f"{k} = {v!r}" for k, v in record["input"].items()))
    data = parse_json(record["output"])
    if record["name"] == "search_it_knowledge_base" and data:
        for hit in data.get("results", []):
            st.markdown(f"- {hit['section']} (`{hit['source']}`, relevance {hit['score']})")
    elif record["name"] == "web_search" and data:
        for r in data.get("results", []):
            st.markdown(f"- [{r['title']}]({r['url']})")
    else:
        st.code(record["output"][:1500], language="json" if data else None)


def render_assistant_body(msg):
    st.markdown(msg["text"])
    for record in msg.get("activity", []):
        if record["name"] in ("create_support_ticket", "check_ticket_status"):
            data = parse_json(record["output"])
            if data and "ticket_id" in data:
                render_ticket(data)
    model_label = msg.get("model")
    if model_label and msg.get("effort"):
        model_label += f" ({msg['effort']} effort)"
    caption = " · ".join(x for x in (model_label, msg.get("note")) if x)
    if caption:
        st.caption(caption)
    if msg.get("activity"):
        with st.expander(f"Tools used ({len(msg['activity'])})"):
            for record in msg["activity"]:
                render_tool_details(record)


def render_sidebar(assistant):
    st.sidebar.header("🛠️ IT Helpdesk")
    model_name = st.sidebar.selectbox("LLM model", list(MODELS),
                                      index=list(MODELS).index(DEFAULT_MODEL))
    st.sidebar.caption(MODELS[model_name]["about"])
    # Remember the chosen effort separately for each model. (Streamlit forgets a
    # widget's value when it isn't shown, so the choices are kept in our own dict.)
    efforts = st.session_state.setdefault("efforts", {})
    effort = st.sidebar.select_slider(
        "Effort", options=EFFORT_LEVELS,
        value=efforts.get(model_name, MODELS[model_name]["default_effort"]),
        key=f"effort_{model_name}",
        help="Higher effort = more thorough answers, but slower and more tokens. "
             "Low or medium is usually enough for IT questions.")
    efforts[model_name] = effort
    st.sidebar.divider()
    st.sidebar.markdown(
        "**I can help you with**\n"
        "- Answers from the IT support knowledge base\n"
        "- Extra troubleshooting from the web\n"
        "- Creating support tickets\n"
        "- Checking ticket status"
    )
    st.sidebar.subheader("Try asking")
    for example in EXAMPLES:
        if st.sidebar.button(example, width="stretch"):
            st.session_state.pending_prompt = example
    st.sidebar.divider()
    if st.sidebar.button("🔄 New conversation", width="stretch"):
        st.session_state.chat = []
        st.session_state.api_messages = []
        st.rerun()
    with st.sidebar.expander("System status"):
        st.markdown(f"- Knowledge base: {assistant.kb.count()} sections of it_sector.txt in ChromaDB\n"
                    f"- MCP server: `{MCP_URL}`\n"
                    f"- Tools: {', '.join(t['name'] for t in assistant.tools)}")
        st.caption("Tickets IT-101 to IT-103 are loaded from mcp_tool_data.json. New tickets "
                   "are kept in the MCP server's memory and are lost when it stops.")
    return model_name, effort


# --- Main ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="IT Helpdesk Assistant", page_icon="🛠️")
    st.title("🛠️ IT Helpdesk Assistant")

    try:
        assistant = get_assistant()
    except Exception as e:
        st.error(f"Could not start the assistant: {e}")
        st.stop()

    st.session_state.setdefault("chat", [])          # what the page shows
    st.session_state.setdefault("api_messages", [])  # full conversation sent to Claude
    model_name, effort = render_sidebar(assistant)

    with st.chat_message("assistant"):
        st.markdown(WELCOME)
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["text"])
            else:
                render_assistant_body(msg)

    prompt = st.chat_input("Describe your IT issue or ask a question...")
    prompt = prompt or st.session_state.pop("pending_prompt", None)
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    api_messages = st.session_state.api_messages
    checkpoint = len(api_messages)
    api_messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with placeholder.status("Working on it...") as status:
            def on_tool_start(name, tool_input):
                status.update(label=PROGRESS_LABELS.get(name, f"Running {name}..."))

            def on_tool_end(record):
                status.write(("❌ " if record["is_error"] else "✅ ")
                             + TOOL_LABELS.get(record["name"], record["name"]))

            try:
                reply = assistant.respond(api_messages, model_name=model_name, effort=effort,
                                          on_tool_start=on_tool_start, on_tool_end=on_tool_end)
            except anthropic.APIStatusError as e:
                del api_messages[checkpoint:]  # undo the half-finished turn
                status.update(label="Something went wrong", state="error")
                st.error(f"Claude API error {e.status_code}: {e.message}")
                return
            except anthropic.APIConnectionError:
                del api_messages[checkpoint:]
                status.update(label="Something went wrong", state="error")
                st.error("Could not reach the Claude API. Check your internet connection.")
                return
        placeholder.empty()
        render_assistant_body(reply)

    st.session_state.chat.append({"role": "user", "text": prompt})
    st.session_state.chat.append({"role": "assistant", **reply})


main()
