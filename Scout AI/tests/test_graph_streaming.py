import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from graph_streaming import StreamTelemetry, stream_graph_execution


class FakeStreamingAgent:
    async def astream_events(self, _state, config=None, version="v2"):
        self.config = config
        self.version = version
        yield {
            "event": "on_chain_start",
            "name": "scout-ai-research",
            "run_id": "root",
            "parent_ids": [],
            "data": {"input": {}},
        }
        yield {
            "event": "on_tool_start",
            "name": "task",
            "run_id": "task-1",
            "parent_ids": ["root"],
            "data": {"input": {"subagent_type": "research-agent"}},
        }
        yield {
            "event": "on_chat_model_start",
            "name": "ChatOpenAI",
            "run_id": "model-1",
            "parent_ids": ["root", "task-1"],
            "data": {
                "input": {
                    "messages": [HumanMessage(content="test prompt " * 20)]
                }
            },
        }
        yield {
            "event": "on_chat_model_end",
            "name": "ChatOpenAI",
            "run_id": "model-1",
            "parent_ids": ["root", "task-1"],
            "data": {
                "output": AIMessage(
                    content="done",
                    usage_metadata={
                        "input_tokens": 123,
                        "output_tokens": 17,
                        "total_tokens": 140,
                    },
                )
            },
        }
        yield {
            "event": "on_tool_start",
            "name": "internet_search",
            "run_id": "search-1",
            "parent_ids": ["root", "task-1"],
            "data": {"input": {"query": "test evidence"}},
        }
        yield {
            "event": "on_tool_start",
            "name": "write_file",
            "run_id": "write-1",
            "parent_ids": ["root", "task-1"],
            "data": {"input": {"file_path": "/research.md"}},
        }
        yield {
            "event": "on_tool_end",
            "name": "task",
            "run_id": "task-1",
            "parent_ids": ["root"],
            "data": {"output": "done"},
        }
        yield {
            "event": "on_chain_end",
            "name": "scout-ai-research",
            "run_id": "root",
            "parent_ids": [],
            "data": {"output": {"messages": [], "files": {}}},
        }


class GraphStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_progress_and_returns_final_root_state(self):
        agent = FakeStreamingAgent()
        updates: list[StreamTelemetry] = []

        result = await stream_graph_execution(
            agent,
            {"messages": []},
            {"run_name": "test"},
            updates.append,
        )

        self.assertEqual(result, {"messages": [], "files": {}})
        self.assertEqual(agent.version, "v2")
        self.assertEqual(updates[0].agent, "research-agent")
        self.assertEqual(updates[0].progress, 35)
        model_update = next(
            update for update in updates if update.log_type == "model"
        )
        search_update = next(
            update for update in updates if "Searching the web" in update.message
        )
        self.assertEqual(model_update.progress, 35)
        self.assertEqual(model_update.model_turn_limit, 10)
        self.assertTrue(model_update.token_count_is_estimate)
        usage_update = next(
            update for update in updates if update.log_type == "model_usage"
        )
        self.assertEqual(usage_update.prompt_tokens, 123)
        self.assertEqual(usage_update.completion_tokens, 17)
        self.assertEqual(usage_update.total_tokens, 140)
        self.assertEqual(search_update.progress, 35)
        self.assertTrue(any("Writing /research.md" in update.message for update in updates))
        self.assertEqual(updates[-1].progress, 55)

    async def test_maps_attempt_and_backoff_custom_events(self):
        class RetryAgent:
            async def astream_events(self, _state, config=None, version="v2"):
                yield {
                    "event": "on_chain_start",
                    "name": "root",
                    "run_id": "root",
                    "parent_ids": [],
                    "data": {},
                }
                yield {
                    "event": "on_custom_event",
                    "name": "scout_agent_attempt",
                    "run_id": "attempt",
                    "parent_ids": ["root"],
                    "data": {
                        "agent": "analysis-agent",
                        "attempt": 2,
                        "max_attempts": 3,
                        "timeout_seconds": 60,
                    },
                }
                yield {
                    "event": "on_chat_model_start",
                    "name": "model",
                    "run_id": "model",
                    "parent_ids": ["root"],
                    "data": {},
                }
                yield {
                    "event": "on_custom_event",
                    "name": "scout_agent_backoff",
                    "run_id": "backoff",
                    "parent_ids": ["root"],
                    "data": {
                        "agent": "analysis-agent",
                        "attempt": 2,
                        "max_attempts": 3,
                        "timeout_seconds": 60,
                        "backoff_seconds": 4,
                    },
                }
                yield {
                    "event": "on_chain_end",
                    "name": "root",
                    "run_id": "root",
                    "parent_ids": [],
                    "data": {"output": {"finished": True}},
                }

        updates: list[StreamTelemetry] = []
        await stream_graph_execution(RetryAgent(), {}, {}, updates.append)

        attempt_update = next(u for u in updates if u.log_type == "attempt")
        model_update = next(u for u in updates if u.log_type == "model")
        backoff_update = next(u for u in updates if u.log_type == "retry")
        self.assertEqual(attempt_update.progress, 60)
        self.assertEqual(attempt_update.attempt, 2)
        self.assertEqual(model_update.model_turn_limit, 7)
        self.assertEqual(model_update.progress, 60)
        self.assertEqual(backoff_update.activity_state, "backoff")
        self.assertEqual(backoff_update.backoff_seconds, 4)
        self.assertEqual(
            backoff_update.message,
            "Analysis attempt timed out. Retrying in 4 seconds.",
        )

    async def test_raises_if_root_stream_has_no_final_state(self):
        class IncompleteAgent:
            async def astream_events(self, _state, config=None, version="v2"):
                yield {
                    "event": "on_chain_start",
                    "name": "root",
                    "run_id": "root",
                    "parent_ids": [],
                    "data": {},
                }

        with self.assertRaisesRegex(RuntimeError, "without a final root state"):
            await stream_graph_execution(
                IncompleteAgent(),
                {},
                {},
                lambda _update: None,
            )

    async def test_installed_event_api_returns_root_output(self):
        runnable = RunnableLambda(lambda state: {**state, "finished": True})

        result = await stream_graph_execution(
            runnable,
            {"value": 1},
            {"run_name": "local-stream-probe"},
            lambda _update: None,
        )

        self.assertEqual(result, {"value": 1, "finished": True})


if __name__ == "__main__":
    unittest.main()
