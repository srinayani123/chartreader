"""Tests for mcp_server/server.py.

Verifies the MCP server exposes the right tool with the right schema.
We don't actually start an MCP stdio session in tests — we just verify
the tool registration and dispatch logic.
"""
import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-import")


class TestMCPToolListing:
    def test_lists_one_tool(self):
        from mcp_server.server import list_tools
        tools = asyncio.run(list_tools())
        assert len(tools) == 1
        assert tools[0].name == "analyze_chart_grounded"

    def test_tool_schema_requires_question(self):
        from mcp_server.server import list_tools
        tools = asyncio.run(list_tools())
        schema = tools[0].inputSchema
        assert "question" in schema["required"]

    def test_tool_schema_accepts_either_image_form(self):
        from mcp_server.server import list_tools
        tools = asyncio.run(list_tools())
        props = tools[0].inputSchema["properties"]
        assert "image_path" in props
        assert "image_base64" in props


class TestMCPDispatch:
    def test_unknown_tool_returns_error(self):
        from mcp_server.server import call_tool
        result = asyncio.run(call_tool("unknown_tool", {}))
        assert len(result) == 1
        assert "Unknown tool" in result[0].text

    def test_missing_question_errors(self):
        from mcp_server.server import call_tool
        result = asyncio.run(
            call_tool("analyze_chart_grounded", {"image_path": "/tmp/x.png"})
        )
        assert len(result) == 1
        assert "question" in result[0].text.lower()

    def test_missing_image_errors(self):
        from mcp_server.server import call_tool
        result = asyncio.run(
            call_tool("analyze_chart_grounded", {"question": "What happened?"})
        )
        assert len(result) == 1
        assert "image" in result[0].text.lower()

    def test_nonexistent_image_path_errors(self):
        from mcp_server.server import call_tool
        result = asyncio.run(
            call_tool("analyze_chart_grounded", {
                "question": "What happened?",
                "image_path": "/this/path/does/not/exist.png",
            })
        )
        assert len(result) == 1
        assert "not found" in result[0].text.lower()
