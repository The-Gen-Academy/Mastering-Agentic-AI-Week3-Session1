"""Reliability middleware for Scout AI sub-agent delegation."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import ensure_config, set_config_context
from langgraph.types import Command

from agent_files import cap_stage_artifact
from config import (
    AGENT_MAX_RETRIES,
    AGENT_TIMEOUT_SECONDS,
    ANALYSIS_ARTIFACT_MAX_LINES,
    BRIEF_READ_MAX_LINES,
    FINAL_READ_MAX_LINES,
    RESEARCH_ARTIFACT_MAX_LINES,
    WRITER_RESEARCH_FALLBACK_MAX_LINES,
)


TIMED_SUBAGENTS = {"research-agent", "analysis-agent", "writer-agent"}
SHARED_WORKSPACE_PATHS = (
    "/brief.md",
    "/research.md",
    "/analysis.md",
    "/final.md",
)
READ_LINE_LIMITS = {
    "/brief.md": BRIEF_READ_MAX_LINES,
    "/research.md": RESEARCH_ARTIFACT_MAX_LINES,
    "/analysis.md": ANALYSIS_ARTIFACT_MAX_LINES,
    "/final.md": FINAL_READ_MAX_LINES,
}


class AgentInactivityTimeoutError(TimeoutError):
    """Raised only when a delegated agent emits no activity for its timeout."""


class SubagentTimeoutError(TimeoutError):
    """Final timeout failure that preserves the responsible agent name."""

    def __init__(self, agent_name: str, attempts: int):
        self.agent_name = agent_name
        self.attempts = attempts
        super().__init__(
            f"{agent_name} produced no model or tool activity for "
            f"{AGENT_TIMEOUT_SECONDS} seconds on each of {attempts} attempts"
        )


class _AgentActivityHeartbeat(BaseCallbackHandler):
    """Track nested LangChain/LangGraph model, tool, and chain activity."""

    def __init__(self) -> None:
        self.last_activity = asyncio.get_running_loop().time()

    def touch(self, *_args: Any, **_kwargs: Any) -> None:
        self.last_activity = asyncio.get_running_loop().time()

    on_chain_start = touch
    on_chain_end = touch
    on_chain_error = touch
    on_chat_model_start = touch
    on_llm_start = touch
    on_llm_new_token = touch
    on_llm_end = touch
    on_llm_error = touch
    on_tool_start = touch
    on_tool_end = touch
    on_tool_error = touch
    on_custom_event = touch


def _config_with_heartbeat(
    request: ToolCallRequest,
    heartbeat: _AgentActivityHeartbeat,
) -> dict[str, Any]:
    runtime = getattr(request, "runtime", None)
    config = ensure_config(getattr(runtime, "config", None))
    callbacks = config.get("callbacks")
    if callbacks is None:
        monitored_callbacks: Any = [heartbeat]
    elif isinstance(callbacks, list):
        monitored_callbacks = [*callbacks, heartbeat]
    else:
        monitored_callbacks = callbacks.copy()
        monitored_callbacks.add_handler(heartbeat, inherit=True)
    return {**config, "callbacks": monitored_callbacks}


async def _await_with_inactivity_timeout(
    task: asyncio.Task[Any],
    heartbeat: _AgentActivityHeartbeat,
    timeout_seconds: float,
) -> Any:
    """Wait indefinitely while activity continues; cancel after true silence."""
    loop = asyncio.get_running_loop()
    try:
        while True:
            inactive_for = loop.time() - heartbeat.last_activity
            remaining = max(0.001, timeout_seconds - inactive_for)
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if task in done:
                return await task
            inactive_for = loop.time() - heartbeat.last_activity
            if inactive_for >= timeout_seconds:
                raise AgentInactivityTimeoutError(
                    f"no model or tool activity for {timeout_seconds:g} seconds"
                )
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


class AgentIterationLimitError(RuntimeError):
    """Iteration-limit failure that preserves the responsible agent name."""

    def __init__(self, agent_name: str, original_error: Exception):
        self.agent_name = agent_name
        self.run_count = getattr(original_error, "run_count", None)
        self.run_limit = getattr(original_error, "run_limit", None)
        limit_summary = (
            f" ({self.run_count}/{self.run_limit})"
            if self.run_count is not None and self.run_limit is not None
            else ""
        )
        super().__init__(
            f"{agent_name} reached its configured iteration limit"
            f"{limit_summary}"
        )


class SharedWorkspacePathMiddleware(AgentMiddleware):
    """Enforce fixed paths plus bounded stage reads and writes."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        tool_name = request.tool.name if request.tool else request.tool_call["name"]
        args = request.tool_call.get("args", {})
        if not isinstance(args, dict):
            return await handler(request)

        if tool_name == "read_file":
            path = args.get("file_path") or args.get("path")
            line_limit = READ_LINE_LIMITS.get(path)
            if self.agent_name == "writer-agent" and path == "/research.md":
                line_limit = WRITER_RESEARCH_FALLBACK_MAX_LINES
            if line_limit is not None:
                try:
                    requested_limit = int(args.get("limit", line_limit))
                except (TypeError, ValueError):
                    requested_limit = line_limit
                args["limit"] = max(1, min(requested_limit, line_limit))
            return await handler(request)

        if tool_name == "write_file":
            path = str(args.get("file_path") or args.get("path") or "")
            content = args.get("content")
            if isinstance(content, str):
                args["content"] = cap_stage_artifact(path, content)
            return await handler(request)

        if tool_name != "ls":
            return await handler(request)

        paths = "\n".join(SHARED_WORKSPACE_PATHS)
        return ToolMessage(
            content=(
                f"{self.agent_name} uses a fixed virtual workspace. "
                "Do not inspect /home or other directories. Available files:\n"
                f"{paths}\nUse read_file with the exact required path."
            ),
            tool_call_id=request.tool_call["id"],
            name="ls",
            status="success",
        )


class SubagentTimeoutRetryMiddleware(AgentMiddleware):
    """Apply a timeout and retry policy to Scout AI's delegated agents.

    DeepAgents delegates work through its ``task`` tool. Each invocation of a
    configured sub-agent gets one initial attempt plus ``AGENT_MAX_RETRIES``
    retries. The timeout measures inactivity rather than total stage duration:
    healthy model, tool, token, and chain events reset the 60-second clock.
    """

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest], Awaitable[ToolMessage | Command[Any]]
        ],
    ) -> ToolMessage | Command[Any]:
        tool_name = request.tool.name if request.tool else request.tool_call["name"]
        args = request.tool_call.get("args", {})
        subagent_type = args.get("subagent_type") if isinstance(args, dict) else None

        if tool_name != "task" or subagent_type not in TIMED_SUBAGENTS:
            return await handler(request)

        total_attempts = AGENT_MAX_RETRIES + 1
        for attempt in range(AGENT_MAX_RETRIES + 1):
            attempt_number = attempt + 1
            await _dispatch_retry_event(
                "scout_agent_attempt",
                {
                    "agent": subagent_type,
                    "attempt": attempt_number,
                    "max_attempts": total_attempts,
                    "timeout_seconds": AGENT_TIMEOUT_SECONDS,
                },
            )
            try:
                heartbeat = _AgentActivityHeartbeat()
                monitored_config = _config_with_heartbeat(request, heartbeat)
                with set_config_context(monitored_config) as context:
                    handler_task = context.run(
                        asyncio.create_task,
                        handler(request),
                    )
                return await _await_with_inactivity_timeout(
                    handler_task,
                    heartbeat,
                    AGENT_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                if attempt >= AGENT_MAX_RETRIES:
                    await _dispatch_retry_event(
                        "scout_agent_timeout",
                        {
                            "agent": subagent_type,
                            "attempt": attempt_number,
                            "max_attempts": total_attempts,
                            "timeout_seconds": AGENT_TIMEOUT_SECONDS,
                        },
                    )
                    raise SubagentTimeoutError(
                        subagent_type,
                        total_attempts,
                    ) from exc

                # Exponential backoff: two seconds, then four seconds.
                backoff_seconds = 2 ** attempt_number
                await _dispatch_retry_event(
                    "scout_agent_backoff",
                    {
                        "agent": subagent_type,
                        "attempt": attempt_number,
                        "next_attempt": attempt_number + 1,
                        "max_attempts": total_attempts,
                        "backoff_seconds": backoff_seconds,
                        "timeout_seconds": AGENT_TIMEOUT_SECONDS,
                    },
                )
                await asyncio.sleep(backoff_seconds)
            except ModelCallLimitExceededError as exc:
                raise AgentIterationLimitError(subagent_type, exc) from exc

        raise RuntimeError("Sub-agent retry loop exited unexpectedly")


async def _dispatch_retry_event(name: str, data: dict[str, Any]) -> None:
    """Emit retry state into LangGraph's v2 event stream when one is active."""
    try:
        await adispatch_custom_event(name, data)
    except RuntimeError:
        # Middleware unit tests and direct calls do not have a parent Runnable.
        # The timeout policy must still work when no event stream is attached.
        return
