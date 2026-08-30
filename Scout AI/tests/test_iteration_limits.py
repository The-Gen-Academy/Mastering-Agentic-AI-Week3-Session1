import unittest
import os
from unittest.mock import patch

from langchain.agents.middleware import ModelCallLimitMiddleware

from agent_middleware import SharedWorkspacePathMiddleware
from analysis_agent import analysis_agent
from config import (
    AGENT_MAX_CONCURRENCY,
    ANALYSIS_AGENT_MAX_TURNS,
    GRAPH_RECURSION_LIMIT,
    ORCHESTRATOR_MAX_TURNS,
    RESEARCH_AGENT_MAX_TURNS,
    WRITER_AGENT_MAX_TURNS,
)
from research_agent import RESEARCH_AGENT_PROMPT, research_agent
from analysis_agent import ANALYSIS_AGENT_PROMPT
from writer_agent import WRITER_AGENT_PROMPT, writer_agent
from orchestrator_agent import resolve_model_roles


class IterationLimitConfigurationTests(unittest.TestCase):
    def test_each_specialist_has_its_configured_hard_limit(self):
        expected_limits = {
            "research-agent": RESEARCH_AGENT_MAX_TURNS,
            "analysis-agent": ANALYSIS_AGENT_MAX_TURNS,
            "writer-agent": WRITER_AGENT_MAX_TURNS,
        }
        self.assertEqual(expected_limits["research-agent"], 10)
        self.assertEqual(expected_limits["analysis-agent"], 7)
        self.assertEqual(expected_limits["writer-agent"], 10)

        for agent in (research_agent, analysis_agent, writer_agent):
            limits = [
                middleware
                for middleware in agent["middleware"]
                if isinstance(middleware, ModelCallLimitMiddleware)
            ]
            self.assertEqual(len(limits), 1, agent["name"])
            self.assertEqual(
                limits[0].run_limit,
                expected_limits[agent["name"]],
                agent["name"],
            )
            self.assertEqual(limits[0].exit_behavior, "error", agent["name"])
            path_guards = [
                middleware
                for middleware in agent["middleware"]
                if isinstance(middleware, SharedWorkspacePathMiddleware)
            ]
            self.assertEqual(len(path_guards), 1, agent["name"])

    def test_specialist_prompts_use_absolute_virtual_paths(self):
        self.assertIn("`/brief.md`", RESEARCH_AGENT_PROMPT)
        self.assertIn("`/research.md`", RESEARCH_AGENT_PROMPT)
        self.assertIn("10 MODEL CALLS MAXIMUM", RESEARCH_AGENT_PROMPT)
        self.assertIn("Never call `ls` first", RESEARCH_AGENT_PROMPT)
        self.assertIn("parallel tool calls", RESEARCH_AGENT_PROMPT)
        self.assertIn("`/research.md`", ANALYSIS_AGENT_PROMPT)
        self.assertIn("`/analysis.md`", ANALYSIS_AGENT_PROMPT)
        self.assertIn("7 MODEL CALLS MAXIMUM", ANALYSIS_AGENT_PROMPT)
        self.assertIn("Never call `ls`", ANALYSIS_AGENT_PROMPT)
        self.assertIn("`/analysis.md`", WRITER_AGENT_PROMPT)
        self.assertIn("`/final.md`", WRITER_AGENT_PROMPT)
        self.assertIn("10 MODEL CALLS MAXIMUM", WRITER_AGENT_PROMPT)
        self.assertNotIn("limit=1000", WRITER_AGENT_PROMPT)
        self.assertIn("limit=100", ANALYSIS_AGENT_PROMPT)
        self.assertIn("`/analysis.md` with `limit=80`", WRITER_AGENT_PROMPT)
        self.assertIn("one `grep` call", WRITER_AGENT_PROMPT)
        self.assertIn("limit=40", WRITER_AGENT_PROMPT)
        self.assertIn("at most 100 lines", RESEARCH_AGENT_PROMPT)
        self.assertIn("at most 80 lines", ANALYSIS_AGENT_PROMPT)
        self.assertIn("parallel `read_file`", WRITER_AGENT_PROMPT)
        self.assertIn("Never call `ls`", WRITER_AGENT_PROMPT)

    def test_orchestrator_and_concurrency_limits_are_bounded(self):
        self.assertEqual(ORCHESTRATOR_MAX_TURNS, 12)
        self.assertEqual(AGENT_MAX_CONCURRENCY, 5)
        self.assertGreaterEqual(
            GRAPH_RECURSION_LIMIT,
            ORCHESTRATOR_MAX_TURNS * 2 + 10,
        )
        self.assertLessEqual(GRAPH_RECURSION_LIMIT, 100)

    def test_model_roles_keep_fast_router_separate_from_synthesis(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_ROUTER_MODEL": "fast-router",
                "OPENAI_MODEL": "large-synthesis",
            },
            clear=True,
        ):
            self.assertEqual(
                resolve_model_roles(),
                ("fast-router", "large-synthesis"),
            )


if __name__ == "__main__":
    unittest.main()
