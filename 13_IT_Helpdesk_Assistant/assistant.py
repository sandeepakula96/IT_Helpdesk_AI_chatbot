"""
The IT Helpdesk Assistant's brain: Claude plus the tools it can choose to call.

Tools Claude can call:
  search_it_knowledge_base  local  RAG over the IT support FAQ (ChromaDB)
  web_search                local  internet search for extra troubleshooting help
  get_current_datetime      local  current date and time
  create_support_ticket     MCP    IT Helpdesk MCP Server
  check_ticket_status       MCP    IT Helpdesk MCP Server
"""
import asyncio
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

BASE_DIR = Path(__file__).resolve().parent

# Models the employee can pick in the sidebar. "fallback": whether the model
# supports the API's server-side refusal fallback (Haiku 5.5 does not).
# "default_effort" is the effort each model uses when none is chosen.
MODELS = {
    "Claude Opus 5.5": {"id": "claude-opus-5-5", "fallback": True, "default_effort": "medium",
                        "about": "Best balance of quality and cost. $4 / $20 per million tokens."},
    "Claude Sonnet 5.5": {"id": "claude-sonnet-5-5", "fallback": True, "default_effort": "high",
                          "about": "Faster and cheaper. $2 / $10 per million tokens."},
    "Claude Haiku 5.5": {"id": "claude-haiku-5-5", "fallback": False, "default_effort": "medium",
                         "about": "Fastest and cheapest; less reliable with tools. "
                                  "$0.10 / $0.50 per million tokens."},
    "Claude Fable 5.2": {"id": "claude-fable-5-2", "fallback": True, "default_effort": "high",
                         "about": "Most capable, slowest and most expensive. "
                                  "$8 / $40 per million tokens."},
}
DEFAULT_MODEL = "Claude Opus 5.5"
EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]  # supported by all models above
MAX_TOKENS = 16000
MAX_TOOL_ROUNDS = 10
FALLBACK_BETA = "server-side-fallback-2026-07-01"

MCP_HOST, MCP_PORT = "127.0.0.1", 3001
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

SYSTEM_PROMPT = """You are the IT Helpdesk Assistant, a chat assistant that helps the company's employees with IT problems such as network connectivity, software installation, slow computers, backups, email, viruses, printers and other hardware, VPN and mobile devices.

How to help:
- For IT questions, search the internal knowledge base first with search_it_knowledge_base. The knowledge base is the company's IT support guidance, so prefer it over anything else and mention the section you used.
- The knowledge base answers are short. Use web_search when it does not cover the problem, when its steps did not work, when the employee needs more detailed or product-specific steps, or when they ask for more options. Say which steps come from the internet and include the source links. Don't suggest risky fixes (disabling security software, editing the registry, unofficial "repair" tools); point the employee to a ticket instead.
- Use get_current_datetime whenever the answer depends on today's date or the current time.
- Give troubleshooting steps as a short numbered list, in the order the employee should try them.

Tickets (IT Helpdesk MCP Server):
- Create a ticket with create_support_ticket when the employee asks for one, or offer to create one when self-help steps have not solved the problem or the issue needs IT staff (hardware failure, lost device, security incident).
- You need the employee's name and a clear description of the issue. Ask for anything missing instead of inventing it. Choose the priority yourself and tell the employee which one you chose: Critical for security incidents or business stopped, High when they cannot work, Medium when they can work with difficulty, Low for questions and requests.
- Use check_ticket_status when the employee asks about an existing ticket. If they did not give a ticket ID, ask for it.
- After creating or checking a ticket, show the ticket ID, issue, status and priority. Never make up ticket IDs or statuses; only report what the tools return.
- You can only create tickets and check their status. You cannot update, reassign or close a ticket, so don't offer to; for changes, the employee can call the Service Desk.
- If the knowledge base search fails, say so briefly and continue with web search or general advice, labelled as not from company documentation.

Stay focused on IT support. If a request is not IT related, say briefly that you can only help with IT questions."""

LOCAL_TOOLS = [
    {
        "name": "search_it_knowledge_base",
        "description": (
            "Search the company's internal IT support documentation and return the most "
            "relevant sections. Use this first for any IT question. Topics covered: "
            "network connectivity, software installation and configuration, system "
            "performance (slow computer), data backup and recovery, email setup, security "
            "and virus protection, hardware and printer issues, VPN setup and mobile devices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The employee's question or problem, in plain words."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the internet for IT troubleshooting solutions that the internal "
            "knowledge base does not cover, such as specific error codes or "
            "application-specific problems. Returns titles, links and snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused web search query."},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 8,
                                "description": "How many results to return (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current local date, time, day of the week and time zone.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# --- MCP client ------------------------------------------------------------
def mcp_server_running():
    try:
        with socket.create_connection((MCP_HOST, MCP_PORT), timeout=1):
            return True
    except OSError:
        return False


def start_mcp_server(timeout=20):
    """Start mcp_server.py in the background if it is not already running."""
    if mcp_server_running():
        return
    log = open(BASE_DIR / "mcp_server.log", "a", encoding="utf-8")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen([sys.executable, str(BASE_DIR / "mcp_server.py")], cwd=BASE_DIR,
                     stdout=log, stderr=log, creationflags=flags)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mcp_server_running():
            return
        time.sleep(0.3)
    raise RuntimeError(f"The MCP server did not start. See {BASE_DIR / 'mcp_server.log'}.")


async def _with_mcp_session(fn):
    """Open a short MCP connection, run fn(session), then close it."""
    async with streamable_http_client(MCP_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await fn(session)


def list_mcp_tools():
    """Fetch the MCP server's tools and convert them to Claude tool definitions."""
    async def _list(session):
        return await session.list_tools()
    result = asyncio.run(_with_mcp_session(_list))
    return [{"name": t.name, "description": t.description or t.name,
             "input_schema": t.input_schema} for t in result.tools]


def call_mcp_tool(name, arguments):
    async def _call(session):
        return await session.call_tool(name, arguments)
    result = asyncio.run(_with_mcp_session(_call))
    text = "\n".join(b.text for b in result.content if b.type == "text")
    return text, bool(result.is_error)


# --- Local tools -----------------------------------------------------------
def web_search(query, max_results=5, attempts=3):
    max_results = max(1, min(int(max_results), 8))
    results = []
    for attempt in range(attempts):
        try:
            results = DDGS().text(query, max_results=max_results)
            if results:
                break
        except DDGSException:  # also covers rate limits and timeouts
            if attempt == attempts - 1:
                raise
        time.sleep(1.5)  # DuckDuckGo sometimes returns nothing on the first try
    if not results:
        return json.dumps({"results": [], "note": "No results found."})
    return json.dumps({"results": [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in results
    ]})


def get_current_datetime():
    now = datetime.now().astimezone()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M"),
        "day_of_week": now.strftime("%A"),
        "timezone": now.strftime("%Z (UTC%z)"),
        "iso": now.isoformat(timespec="seconds"),
    })


# --- The assistant -----------------------------------------------------------
class HelpdeskAssistant:
    def __init__(self, client, knowledge_base, mcp_tools):
        self.client = client
        self.kb = knowledge_base
        self.mcp_tool_names = {t["name"] for t in mcp_tools}
        self.tools = LOCAL_TOOLS + mcp_tools

    def run_tool(self, name, tool_input):
        """Run one tool. Returns (result_text, is_error)."""
        try:
            if name in self.mcp_tool_names:
                return call_mcp_tool(name, tool_input)
            if name == "search_it_knowledge_base":
                hits = self.kb.search(tool_input["query"])
                return json.dumps({"results": hits} if hits else
                                  {"results": [], "note": "Nothing relevant in the knowledge base."}), False
            if name == "web_search":
                return web_search(tool_input["query"], tool_input.get("max_results", 5)), False
            if name == "get_current_datetime":
                return get_current_datetime(), False
            return f"Unknown tool: {name}", True
        except Exception as e:  # report tool failures to Claude instead of crashing
            return f"Tool {name} failed: {type(e).__name__}: {e}", True

    def respond(self, messages, model_name=DEFAULT_MODEL, effort=None,
                on_tool_start=None, on_tool_end=None):
        """Run Claude, executing any tools it asks for, until it gives a final answer.

        `messages` is the full API conversation; this appends every assistant turn
        and tool result to it. Returns {"text", "activity", "note", "model", "effort"}.
        """
        model = MODELS[model_name]
        effort = effort or model["default_effort"]
        extra = {}
        if model["fallback"]:
            # If the model declines on safety grounds, the API retries on a fallback model
            extra["fallbacks"] = "default"
        activity = []
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.beta.messages.create(
                model=model["id"],
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=messages,
                output_config={"effort": effort},
                cache_control={"type": "ephemeral"},  # cache the system prompt, tools and history
                betas=[FALLBACK_BETA],
                **extra,
            )
            # Keep the full content (thinking blocks included) so history goes back unchanged
            messages.append({"role": "assistant",
                             "content": [b.to_dict() for b in response.content]})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                return {"text": self._final_text(response), "activity": activity,
                        "note": self._note(response, model["id"]), "model": model_name,
                        "effort": effort}

            results = []
            for block in tool_uses:
                if on_tool_start:
                    on_tool_start(block.name, block.input)
                output, is_error = self.run_tool(block.name, block.input)
                record = {"name": block.name, "input": block.input,
                          "output": output, "is_error": is_error}
                activity.append(record)
                if on_tool_end:
                    on_tool_end(record)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output, "is_error": is_error})
            # All tool results for one turn go back together in a single user message
            messages.append({"role": "user", "content": results})

        return {"text": "Sorry, this request needed too many steps. Please try rephrasing it.",
                "activity": activity, "note": "", "model": model_name, "effort": effort}

    @staticmethod
    def _final_text(response):
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "refusal":
            return text or "Sorry, I can't help with that request."
        return text or "(No answer was returned.)"

    @staticmethod
    def _note(response, requested_model_id):
        notes = []
        if response.stop_reason == "max_tokens":
            notes.append("The answer was cut off because it reached the length limit.")
        if response.model != requested_model_id:
            notes.append(f"Answered by fallback model {response.model}.")
        return " ".join(notes)
