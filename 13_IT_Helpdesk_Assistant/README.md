# IT Helpdesk Assistant

A chatbot that helps employees with common IT problems. It answers from the company's IT support document using RAG, searches the web for extra solutions, and creates and checks support tickets through an MCP server.

## How it works

| Requirement | Where | How |
|---|---|---|
| RAG | `rag.py`, `../Docs/Day-12/it_sector.txt` | The IT support document is split into one chunk per `## ` section, embedded with Voyage AI (`voyage-4-lite`) and stored in a ChromaDB vector database (`vector_db/`). It is re-embedded automatically when the document changes. |
| MCP server | `mcp_server.py`, `../Docs/Day-12/mcp_tool_data.json` | "IT Helpdesk MCP Server" (streamable HTTP on `http://127.0.0.1:3001/mcp`) with `create_support_ticket` and `check_ticket_status`. Tickets are stored in an in-memory Python dictionary that is loaded with the existing tickets from `mcp_tool_data.json` at startup. |
| Tool calling | `assistant.py` | Claude (`claude-opus-5-5`) decides which tools to call: `search_it_knowledge_base`, `web_search` (DuckDuckGo via `ddgs`), `get_current_datetime`, plus the two MCP tools, which are loaded from the server at startup. |
| Chat interface | `app.py` | A Streamlit chat. It shows answers and troubleshooting steps, ticket cards, and a "Tools used" panel under each answer. |

```
Employee -> Streamlit (app.py) -> Claude + tool loop (assistant.py)
                                    |- search_it_knowledge_base -> ChromaDB + Voyage (rag.py)
                                    |- web_search               -> DuckDuckGo
                                    |- get_current_datetime     -> system clock
                                    '- create_support_ticket / check_ticket_status
                                                                -> MCP client -> mcp_server.py (in-memory dict)
```

## Setup

1. Install the packages:
   ```powershell
   pip install -r requirements.txt
   ```
2. Put your keys in `04_claude_anthropic/.env` (the parent folder; a `.env` in this folder also works):
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   VOYAGE_API_KEY=pa-...
   ```
3. Run the app from this folder:
   ```powershell
   streamlit run app.py
   ```
   The app starts the MCP server in the background if it is not already running. To run the server yourself instead (for example, to watch its output), start `python mcp_server.py` in another terminal first.

## Things to know

- **Tickets:** IT-101 to IT-103 come from `mcp_tool_data.json` and can be checked straight away. New tickets continue from IT-104 and are kept only in memory: the JSON file is never changed, and new tickets are lost when the MCP server stops. Restarting Streamlit alone keeps them, since the server keeps running in the background.
- **After changing `mcp_server.py` or the JSON file**, stop the background server (or restart your computer) so the app starts a fresh one with the new code and data.
- **Voyage free tier:** without a payment method on your Voyage account, Voyage allows only 3 requests per minute. If you send several questions quickly, a knowledge-base search may wait up to about 40 seconds before it goes through. Adding a payment method on the Voyage dashboard removes this limit, and the free token allowance still applies.
- **Changing the knowledge base:** edit `it_sector.txt`, or add more files to `SOURCE_FILES` in `rag.py`. Each `## ` heading becomes one searchable section.
- **Model and effort:** set with `MODEL` and `EFFORT` at the top of `assistant.py`. If Claude Opus 5.5 declines a request on safety grounds, the API automatically retries it on a fallback model.
