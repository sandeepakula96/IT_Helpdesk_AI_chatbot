"""
IT Helpdesk MCP Server.

Exposes two tools over MCP (streamable HTTP):
  - create_support_ticket: create a new IT support ticket
  - check_ticket_status:   look up an existing ticket by ID

Tickets are kept in an in-memory Python dictionary. At startup it is loaded
with the existing tickets from 04_claude_anthropic/Docs/Day-12/mcp_tool_data.json;
tickets created later live only while this server process is running (the
JSON file is never modified).

Run on its own with:   python mcp_server.py
(The Streamlit app also starts it automatically if it is not already running.)
"""
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import uvicorn
from mcp.server import MCPServer

HOST = "127.0.0.1"
PORT = 3001

mcp_server = MCPServer(name="IT Helpdesk MCP Server", version="1.0.0")

TICKET_DATA_FILE = Path(__file__).resolve().parent.parent / "Docs" / "Day-12" / "mcp_tool_data.json"


def load_tickets(path):
    """Load existing tickets from the JSON file into a dict keyed by ticket ID."""
    if not path.exists():
        print(f"Warning: {path} not found; starting with no tickets.")
        return {}
    return {t["ticket_id"].upper(): t for t in json.loads(path.read_text(encoding="utf-8"))}


# In-memory ticket store: {"IT-101": {"ticket_id", "employee", "issue", ...}}
TICKETS: dict[str, dict] = load_tickets(TICKET_DATA_FILE)
# New tickets continue numbering after the highest existing ID (IT-103 -> IT-104)
_next_ticket_number = max(
    [int(m.group()) for t in TICKETS for m in [re.search(r"\d+", t)] if m] or [100]
) + 1


def _now():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")


@mcp_server.tool(
    name="create_support_ticket",
    description=(
        "Create a new IT support ticket for an employee. Use this when the employee "
        "asks to raise/log a ticket, or when the issue cannot be fixed with self-help "
        "steps. Arguments: employee = the employee's name, issue = a short title of "
        "the problem, description = details of what is happening. Returns the new "
        "ticket with its ticket_id (for example IT-104)."
    ),
)
def create_support_ticket(
    employee: str,
    issue: str,
    description: str,
    priority: Literal["Low", "Medium", "High", "Critical"] = "Medium",
) -> str:
    global _next_ticket_number
    ticket_id = f"IT-{_next_ticket_number}"
    _next_ticket_number += 1
    # Same fields as the tickets in mcp_tool_data.json, plus the creation time
    ticket = {
        "ticket_id": ticket_id,
        "employee": employee.strip(),
        "issue": issue.strip(),
        "description": description.strip(),
        "priority": priority,
        "status": "Open",
        "created_at": _now(),
    }
    TICKETS[ticket_id] = ticket
    return json.dumps(ticket)


@mcp_server.tool(
    name="check_ticket_status",
    description=(
        "Look up an existing IT support ticket by its ID (for example IT-101) and "
        "return its current status and details."
    ),
)
def check_ticket_status(ticket_id: str) -> str:
    key = ticket_id.strip().upper()
    if key.isdigit():
        key = f"IT-{key}"  # accept "1001" as well as "IT-1001"
    ticket = TICKETS.get(key)
    if ticket is None:
        return json.dumps({"error": f"No ticket found with ID {ticket_id}.",
                           "known_ticket_ids": sorted(TICKETS)})
    return json.dumps(ticket)


app = mcp_server.streamable_http_app()

if __name__ == "__main__":
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    # A SelectorEventLoop on Windows avoids noisy ConnectionResetError logs from
    # the client's short-lived connections (same approach as 12_MCP/01_Server).
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    print(f"IT Helpdesk MCP Server running at http://{HOST}:{PORT}/mcp")
    asyncio.run(server.serve(), loop_factory=loop_factory)
