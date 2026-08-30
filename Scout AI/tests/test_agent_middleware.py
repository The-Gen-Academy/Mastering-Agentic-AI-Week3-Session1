import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import agent_middleware
from agent_middleware import (
    AgentIterationLimitError,
    SharedWorkspacePathMiddleware,
    SubagentTimeoutRetryMiddleware,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from config import (
    RESEARCH_ARTIFACT_MAX_CHARS,
    RESEARCH_ARTIFACT_MAX_LINES,
    WRITER_RESEARCH_FALLBACK_MAX_LINES,
)


class SubagentTimeoutRetryMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_inactivity_timeout_resets_while_activity_continues(self):
        heartbeat = agent_middleware._AgentActivityHeartbeat()

        async def active_handler():
            for _ in range(10):
                await asyncio.sleep(0.01)
                heartbeat.touch()
            return "completed"

        task = asyncio.create_task(active_handler())
        result = await agent_middleware._await_with_inactivity_timeout(
            task,
            heartbeat,
            timeout_seconds=0.05,
        )

        self.assertEqual(result, "completed")

    async def test_target_subagent_times_out_and_retries_twice(self):
        calls = 0

        async def slow_handler(_request):
            nonlocal calls
            calls += 1
            await asyncio.Future()

        request = SimpleNamespace(
            tool=None,
            tool_call={
                "name": "task",
                "args": {"subagent_type": "research-agent"},
            },
        )

        with (
            patch.object(agent_middleware, "AGENT_TIMEOUT_SECONDS", 0.01),
            patch.object(agent_middleware, "AGENT_MAX_RETRIES", 2),
            patch.object(
                agent_middleware.asyncio,
                "sleep",
                new_callable=AsyncMock,
            ) as sleep_mock,
        ):
            with self.assertRaisesRegex(TimeoutError, "3 attempts"):
                await SubagentTimeoutRetryMiddleware().awrap_tool_call(
                    request,
                    slow_handler,
                )

        self.assertEqual(calls, 3)
        self.assertEqual(
            [await_call.args[0] for await_call in sleep_mock.await_args_list],
            [2, 4],
        )

    async def test_non_target_tool_is_not_retried(self):
        calls = 0

        async def handler(_request):
            nonlocal calls
            calls += 1
            return "ok"

        request = SimpleNamespace(
            tool=None,
            tool_call={"name": "calculator", "args": {}},
        )
        result = await SubagentTimeoutRetryMiddleware().awrap_tool_call(
            request,
            handler,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(calls, 1)

    async def test_iteration_limit_error_preserves_subagent_name(self):
        async def limited_handler(_request):
            raise ModelCallLimitExceededError(
                thread_count=0,
                run_count=10,
                thread_limit=None,
                run_limit=10,
            )

        request = SimpleNamespace(
            tool=None,
            tool_call={
                "name": "task",
                "args": {"subagent_type": "research-agent"},
            },
        )

        with self.assertRaises(AgentIterationLimitError) as raised:
            await SubagentTimeoutRetryMiddleware().awrap_tool_call(
                request,
                limited_handler,
            )

        self.assertEqual(raised.exception.agent_name, "research-agent")
        self.assertEqual(
            str(raised.exception),
            "research-agent reached its configured iteration limit (10/10)",
        )

    async def test_workspace_guard_redirects_ls_without_calling_backend(self):
        calls = 0

        async def handler(_request):
            nonlocal calls
            calls += 1
            return "unexpected"

        request = SimpleNamespace(
            tool=None,
            tool_call={
                "id": "tool-call-1",
                "name": "ls",
                "args": {"path": "/home/ubuntu/research"},
            },
        )
        result = await SharedWorkspacePathMiddleware(
            "analysis-agent"
        ).awrap_tool_call(request, handler)

        self.assertEqual(calls, 0)
        self.assertIn("/brief.md", result.content)
        self.assertIn("/research.md", result.content)
        self.assertIn("Do not inspect /home", result.content)

    async def test_workspace_guard_clamps_reads_and_caps_stage_writes(self):
        seen_args = []

        async def handler(request):
            seen_args.append(dict(request.tool_call["args"]))
            return "ok"

        middleware = SharedWorkspacePathMiddleware("writer-agent")
        read_request = SimpleNamespace(
            tool=None,
            tool_call={
                "name": "read_file",
                "args": {"file_path": "/research.md", "limit": 1000},
            },
        )
        await middleware.awrap_tool_call(read_request, handler)

        oversized = "\n".join("evidence " + "x" * 300 for _ in range(140))
        write_request = SimpleNamespace(
            tool=None,
            tool_call={
                "name": "write_file",
                "args": {
                    "file_path": "/research.md",
                    "content": oversized,
                },
            },
        )
        await middleware.awrap_tool_call(write_request, handler)

        self.assertEqual(
            seen_args[0]["limit"],
            WRITER_RESEARCH_FALLBACK_MAX_LINES,
        )
        bounded = seen_args[1]["content"]
        self.assertLessEqual(len(bounded.splitlines()), RESEARCH_ARTIFACT_MAX_LINES)
        self.assertLessEqual(len(bounded), RESEARCH_ARTIFACT_MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
