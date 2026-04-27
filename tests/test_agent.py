"""Tests for src/agents/analyst.py — verifies the agent assembles correctly.

We don't make actual LLM calls in these tests. We just verify:
  - the agent constructs without errors
  - the right tools are registered
  - the singleton pattern works
"""
import os
from unittest.mock import patch

import pytest

# Ensure required env vars are set for import
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-import")


class TestAgentAssembly:
    def test_agent_builds(self):
        """The agent should construct without raising."""
        from src.agents.analyst import _build_agent
        agent = _build_agent()
        assert agent is not None

    def test_singleton_returns_same_instance(self):
        """get_analyst() should return the same agent on repeated calls."""
        from src.agents.analyst import get_analyst
        a1 = get_analyst()
        a2 = get_analyst()
        assert a1 is a2


class TestAgentImports:
    """Verify that the full module graph imports cleanly."""

    def test_tools_module_imports(self):
        from src.agents import tools
        assert hasattr(tools, "check_refusal")
        assert hasattr(tools, "extract_chart_data")
        assert hasattr(tools, "verify_against_yfinance")
        assert hasattr(tools, "get_news")
        assert hasattr(tools, "get_peer_comparison")

    def test_prompts_module_imports(self):
        from src.agents.prompts import ANALYST_SYSTEM_PROMPT
        assert "ChartReader" in ANALYST_SYSTEM_PROMPT
        # Sanity: prompt mentions refusal
        assert "REFUSE" in ANALYST_SYSTEM_PROMPT.upper()

    def test_analyst_module_imports(self):
        from src.agents.analyst import analyze_chart, get_analyst
        assert callable(analyze_chart)
        assert callable(get_analyst)
