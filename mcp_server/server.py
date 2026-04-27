"""MCP server wrapping ChartReader's agent.

Exposes a single tool: `analyze_chart_grounded(image, question)`.
Claude Desktop (or any MCP client) calls this tool to get a structured
GroundedAnswer back.

Run with: python -m mcp_server.server

Or register in claude_desktop_config.json:
{
  "mcpServers": {
    "chartreader": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "C:/Users/manka/Downloads/chartreader"
    }
  }
}
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.agents.analyst import analyze_chart
from src.retrieval.store import init_schema

load_dotenv()
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


server = Server("chartreader")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise our single tool."""
    return [
        Tool(
            name="analyze_chart_grounded",
            description=(
                "Analyze a financial chart image and answer a question about it. "
                "The agent extracts visual data, verifies against yfinance ground "
                "truth, retrieves relevant news, and returns a structured answer "
                "with citations. Refuses prediction/advice questions explicitly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": (
                            "Chart image as a base64-encoded string. Either this "
                            "or `image_path` must be provided."
                        ),
                    },
                    "image_path": {
                        "type": "string",
                        "description": (
                            "Local file path to the chart image. Either this or "
                            "`image_base64` must be provided."
                        ),
                    },
                    "image_mime": {
                        "type": "string",
                        "description": "MIME type of the image (default: image/png)",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to answer about the chart.",
                    },
                },
                "required": ["question"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool calls to ChartReader's analyst."""
    if name != "analyze_chart_grounded":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    question = (arguments.get("question") or "").strip()
    if not question:
        return [TextContent(type="text",
                            text="Error: `question` is required.")]

    image_path_arg = arguments.get("image_path")
    image_b64 = arguments.get("image_base64")
    mime = arguments.get("image_mime", "image/png")

    if not image_path_arg and not image_b64:
        return [TextContent(type="text",
                            text="Error: provide image_path or image_base64.")]

    # Resolve to a local file path
    cleanup_path = None
    if image_b64:
        suffix_map = {"image/png": ".png", "image/jpeg": ".jpg",
                      "image/gif": ".gif", "image/webp": ".webp"}
        suffix = suffix_map.get(mime, ".png")
        try:
            data = base64.b64decode(image_b64)
        except Exception as exc:
            return [TextContent(type="text",
                                text=f"Error decoding image_base64: {exc}")]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            image_path = tmp.name
            cleanup_path = image_path
    else:
        image_path = image_path_arg
        if not Path(image_path).exists():
            return [TextContent(type="text",
                                text=f"Error: image file not found: {image_path}")]

    try:
        answer = await analyze_chart(image_path, question)
    except Exception as exc:
        logger.exception("Tool call failed")
        return [TextContent(type="text", text=f"Error: {exc}")]
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except Exception:
                pass

    # Return structured JSON so the calling agent can parse it cleanly
    payload = answer.model_dump(mode="json")
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


async def main() -> None:
    # Try to init pgvector schema at startup. Failure is non-fatal — the agent
    # can still run without semantic news cache (it uses Tavily live).
    try:
        init_schema()
    except Exception as exc:
        logger.warning("pgvector init skipped: %s", exc)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
