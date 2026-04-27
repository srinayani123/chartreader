"""End-to-end test client for the ChartReader MCP server.

This script proves that the MCP server in `mcp_server/server.py` works
end-to-end by:
  1. Spawning the server as a subprocess (over stdio).
  2. Initializing the MCP protocol session.
  3. Listing available tools (verifies the tool advertises correctly).
  4. Calling `analyze_chart_grounded` with a real chart from the test set.
  5. Printing the structured response.
  6. Saving the response to mcp_test_output.json for documentation.

Why this exists: it is impractical to verify MCP integration only through
Claude Desktop, which depends on a working desktop install. This automated
test exercises the same protocol that any MCP client (Claude Desktop,
VS Code Copilot, custom agents) would use, and produces a reproducible
artifact suitable for CI or repo documentation.

Run with (from the project root, with venv active):
    python scripts/test_mcp_client.py

Expected output: a JSON-pretty-printed GroundedAnswer plus a short summary.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Resolve the project root from this script's location.
# This script lives in scripts/, so root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CHART = PROJECT_ROOT / "eval" / "test_set" / "charts" / "AAPL_6mo.png"
OUTPUT_FILE = PROJECT_ROOT / "mcp_test_output.json"

# Use the venv's python explicitly so the subprocess inherits all dependencies.
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    # Fall back to whichever python is on PATH if venv layout is different.
    VENV_PYTHON = "python"
else:
    VENV_PYTHON = str(VENV_PYTHON)


async def run_test() -> int:
    if not TEST_CHART.exists():
        print(f"[ERROR] Test chart not found: {TEST_CHART}", file=sys.stderr)
        print("        Make sure you're running from the project root.",
              file=sys.stderr)
        return 1

    print("=" * 70)
    print("ChartReader MCP server — end-to-end client test")
    print("=" * 70)
    print(f"Spawning server: {VENV_PYTHON} -m mcp_server.server")
    print(f"Test chart:      {TEST_CHART}")
    print(f"CWD:             {PROJECT_ROOT}")
    print()

    server_params = StdioServerParameters(
        command=VENV_PYTHON,
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT),
        env=None,  # inherit current environment (so .env, API keys, etc. flow)
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 1. Initialize the protocol handshake.
            print("[1/3] Initializing MCP session...")
            init_result = await session.initialize()
            server_name = init_result.serverInfo.name
            server_version = init_result.serverInfo.version
            print(f"      Connected to: {server_name} v{server_version}")
            print()

            # 2. List the tools the server advertises.
            print("[2/3] Listing available tools...")
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]
            print(f"      Tools advertised: {tool_names}")
            if "analyze_chart_grounded" not in tool_names:
                print("[ERROR] Expected 'analyze_chart_grounded' tool was "
                      "not advertised. Server may have a registration bug.",
                      file=sys.stderr)
                return 2
            print("      OK — analyze_chart_grounded registered.")
            print()

            # 3. Call the tool with a real chart and a real question.
            print("[3/3] Calling analyze_chart_grounded...")
            print(f"      image_path: {TEST_CHART}")
            print(f"      question:   'What happened to AAPL during this period?'")
            print(f"      (this will exercise the full agent — vision +")
            print(f"       yfinance verify + news retrieval — may take ~60s)")
            print()
            tool_result = await session.call_tool(
                "analyze_chart_grounded",
                arguments={
                    "image_path": str(TEST_CHART),
                    "question": "What happened to AAPL during this period?",
                },
            )

    # The server returns a list of TextContent blocks; we expect one with
    # a JSON-encoded GroundedAnswer.
    if not tool_result.content:
        print("[ERROR] Tool returned no content blocks.", file=sys.stderr)
        return 3

    text_block = tool_result.content[0]
    response_text = getattr(text_block, "text", str(text_block))

    # Try to parse as JSON; if it's an error message, print it as-is.
    try:
        parsed = json.loads(response_text)
        is_structured = True
    except json.JSONDecodeError:
        parsed = response_text
        is_structured = False

    print("=" * 70)
    print("TOOL RESPONSE")
    print("=" * 70)
    if is_structured:
        print(json.dumps(parsed, indent=2)[:2000])
        if len(json.dumps(parsed, indent=2)) > 2000:
            print(f"\n... (truncated, full response saved to {OUTPUT_FILE.name})")
    else:
        print(parsed)
    print()

    # Save the full response for the README / docs.
    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "server_name": server_name,
                "server_version": server_version,
                "tools_advertised": tool_names,
                "test_chart": str(TEST_CHART.relative_to(PROJECT_ROOT)),
                "test_question": "What happened to AAPL during this period?",
                "response_was_structured_json": is_structured,
                "response": parsed,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=" * 70)
    print(f"PASS — full response saved to {OUTPUT_FILE.name}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_test())
    sys.exit(exit_code)
    