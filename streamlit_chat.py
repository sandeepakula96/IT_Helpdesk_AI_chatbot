"""
Streamlit chat app for Claude with model, effort, thinking and system prompt controls.

Setup (PowerShell):
    pip install streamlit anthropic python-dotenv

Key: put ANTHROPIC_API_KEY=sk-ant-... in a .env file next to this script
(the same .env the RAG chatbot uses), then run:
    streamlit run .\\streamlit_chat.py
"""
import os
from pathlib import Path

import anthropic
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# Each model has different rules for turning thinking off:
#   thinking_off = None            -> thinking is always on (toggle is locked)
#   thinking_off = "between_tools" -> Sonnet 5.5's way to turn thinking off
#   thinking_off = "disabled"      -> standard {"type": "disabled"}
# Thinking off is only accepted at effort low/medium/high on these models.
MODELS = {
    "Claude Opus 5.5": {"id": "claude-opus-5-5", "thinking_off": None,
                        "default_effort": "medium", "fallback": True},
    "Claude Sonnet 5.5": {"id": "claude-sonnet-5-5", "thinking_off": "between_tools",
                          "default_effort": "high", "fallback": True},
    "Claude Haiku 5.5": {"id": "claude-haiku-5-5", "thinking_off": "disabled",
                         "default_effort": "medium", "fallback": False},
    "Claude Fable 5.2": {"id": "claude-fable-5-2", "thinking_off": None,
                         "default_effort": "high", "fallback": True},
}
ALL_EFFORTS = ["low", "medium", "high", "xhigh", "max"]
THINKING_OFF_EFFORTS = ["low", "medium", "high"]
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 32000


# --- Client ---------------------------------------------------------------
@st.cache_resource
def get_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        st.error(f"ANTHROPIC_API_KEY is not set. Add it to {BASE_DIR / '.env'} and restart.")
        st.stop()
    return anthropic.Anthropic()


# --- Sidebar: request settings --------------------------------------------
def sidebar_settings():
    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox("Model", list(MODELS))
    model = MODELS[model_name]

    can_disable = model["thinking_off"] is not None
    thinking_on = st.sidebar.toggle(
        "Thinking", value=True, disabled=not can_disable,
        help=None if can_disable else f"{model_name} always thinks; it can't be turned off.",
    )
    if not can_disable:
        thinking_on = True
        st.sidebar.caption(f"{model_name} always thinks. Use a lower effort for faster answers.")

    efforts = ALL_EFFORTS if thinking_on else THINKING_OFF_EFFORTS
    default = model["default_effort"] if model["default_effort"] in efforts else "high"
    effort = st.sidebar.select_slider("Effort", options=efforts, value=default,
                                      help="Higher effort = more thorough, slower, more tokens.")
    if not thinking_on:
        st.sidebar.caption("With thinking off, effort is limited to low / medium / high.")

    system = st.sidebar.text_area("System instruction", value="You are a helpful assistant.",
                                  height=150)

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    return {"model": model, "thinking_on": thinking_on, "effort": effort, "system": system}


# --- Build the request dynamically ----------------------------------------
def build_request(settings, messages):
    model = settings["model"]
    if settings["thinking_on"]:
        thinking = {"type": "adaptive", "display": "summarized"}  # show a readable summary
    else:
        thinking = {"type": model["thinking_off"]}

    request = {
        "model": model["id"],
        "max_tokens": MAX_TOKENS,
        "messages": messages,
        "thinking": thinking,
        "output_config": {"effort": settings["effort"]},
    }
    if settings["system"].strip():
        request["system"] = settings["system"].strip()
    if model["fallback"]:
        # If the model declines on safety grounds, the API retries on a fallback model
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = "default"
    return request


def request_summary(request):
    """Request with the conversation shortened, for display."""
    shown = {k: v for k, v in request.items() if k != "messages"}
    shown["messages"] = f"<{len(request['messages'])} messages; last: "\
                        f"{request['messages'][-1]['content'][:80]!r}>"
    return shown


# --- Chat -------------------------------------------------------------------
def render_message(msg):
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("Thinking"):
                st.markdown(msg["thinking"])
        st.markdown(msg["text"])
        if msg.get("note"):
            st.caption(msg["note"])


def stream_reply(client, request):
    """Stream the reply into the page. Returns (text, thinking, content, note)."""
    thinking_box = st.expander("Thinking", expanded=False) if request["thinking"]["type"] == "adaptive" else None
    thinking_area = thinking_box.empty() if thinking_box else None
    text_area = st.empty()
    text, thinking = "", ""

    api = client.beta.messages if "betas" in request else client.messages
    with api.stream(**request) as stream:
        for event in stream:
            if event.type != "content_block_delta":
                continue
            if event.delta.type == "thinking_delta" and thinking_area:
                thinking += event.delta.thinking
                thinking_area.markdown(thinking)
            elif event.delta.type == "text_delta":
                text += event.delta.text
                text_area.markdown(text + "▌")
        final = stream.get_final_message()

    text = "".join(b.text for b in final.content if b.type == "text").strip()
    note = ""
    if final.stop_reason == "refusal":
        text = text or "_The model declined to answer this request._"
    elif final.stop_reason == "max_tokens":
        note = "Answer cut off: reached the max_tokens limit."
    if final.model != request["model"]:
        note = (note + " " if note else "") + f"Answered by fallback model {final.model}."
    text = text or "_(No text returned.)_"
    text_area.markdown(text)
    if note:
        st.caption(note)

    # Keep the full content (including thinking blocks) so the next turn
    # sends the conversation back unchanged.
    content = [b.to_dict() for b in final.content]
    return text, thinking, content, note


def main():
    st.set_page_config(page_title="Your assistant", page_icon="💬")
    st.title("Your assistant")

    settings = sidebar_settings()
    client = get_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input("Ask Claude anything...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "text": prompt, "content": prompt})
    render_message(st.session_state.messages[-1])

    api_messages = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages]
    request = build_request(settings, api_messages)

    with st.chat_message("assistant"):
        with st.expander("Request sent"):
            st.json(request_summary(request))
        try:
            text, thinking, content, note = stream_reply(client, request)
        except anthropic.APIStatusError as e:
            st.session_state.messages.pop()  # drop the unanswered question
            st.error(f"API error {e.status_code}: {e.message}")
            return
        except anthropic.APIConnectionError:
            st.session_state.messages.pop()
            st.error("Could not reach the Claude API. Check your internet connection.")
            return

    st.session_state.messages.append({"role": "assistant", "text": text, "thinking": thinking,
                                      "content": content, "note": note})


main()
