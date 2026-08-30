"""Translate LangGraph execution events into Scout AI progress telemetry."""

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from config import (
    AGENT_MAX_RETRIES,
    AGENT_TIMEOUT_SECONDS,
    ANALYSIS_AGENT_MAX_TURNS,
    ORCHESTRATOR_MAX_TURNS,
    RESEARCH_AGENT_MAX_TURNS,
    WRITER_AGENT_MAX_TURNS,
)

STAGE_START_PROGRESS = {
    "research-agent": 35,
    "analysis-agent": 60,
    "writer-agent": 80,
}
STAGE_COMPLETE_PROGRESS = {
    "research-agent": 55,
    "analysis-agent": 75,
    "writer-agent": 92,
}
AGENT_TURN_LIMITS = {
    "orchestrator": ORCHESTRATOR_MAX_TURNS,
    "research-agent": RESEARCH_AGENT_MAX_TURNS,
    "analysis-agent": ANALYSIS_AGENT_MAX_TURNS,
    "writer-agent": WRITER_AGENT_MAX_TURNS,
}
AGENT_TITLES = {
    "orchestrator": "Orchestrator",
    "research-agent": "Research",
    "analysis-agent": "Analysis",
    "writer-agent": "Writer",
}
OUTPUT_AGENT = {
    "/research.md": "research-agent",
    "/analysis.md": "analysis-agent",
    "/final.md": "writer-agent",
}


@dataclass(frozen=True)
class StreamTelemetry:
    agent: str
    log_type: str
    message: str
    progress: int
    model_turn: int | None = None
    model_turn_limit: int | None = None
    current_tool: str | None = None
    delegation_count: int | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    timeout_seconds: int | None = None
    backoff_seconds: int | None = None
    activity_state: str = "active"
    record_log: bool = True
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    token_count_is_estimate: bool = False


def _event_input(event: Mapping[str, Any]) -> dict[str, Any]:
    value = (event.get("data") or {}).get("input", {})
    if isinstance(value, dict) and set(value) == {"input"}:
        value = value["input"]
    return value if isinstance(value, dict) else {}


def _content_character_count(value: Any) -> int:
    """Count prompt content characters without serializing runtime objects."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_content_character_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_content_character_count(item) for item in value)
    content = getattr(value, "content", None)
    return _content_character_count(content)


def _estimate_prompt_tokens(event: Mapping[str, Any]) -> int | None:
    model_input = (event.get("data") or {}).get("input")
    character_count = _content_character_count(model_input)
    return math.ceil(character_count / 4) if character_count else None


def _extract_token_usage(
    event: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None]:
    output = (event.get("data") or {}).get("output")
    usage = getattr(output, "usage_metadata", None)
    response_metadata = getattr(output, "response_metadata", None)
    if not usage and isinstance(output, Mapping):
        usage = output.get("usage_metadata")
        response_metadata = output.get("response_metadata")
    if not usage and isinstance(response_metadata, Mapping):
        usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, None

    prompt = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion = usage.get("output_tokens", usage.get("completion_tokens"))
    total = usage.get("total_tokens")
    try:
        prompt_value = int(prompt) if prompt is not None else None
        completion_value = int(completion) if completion is not None else None
        total_value = int(total) if total is not None else None
    except (TypeError, ValueError):
        return None, None, None
    if total_value is None and prompt_value is not None and completion_value is not None:
        total_value = prompt_value + completion_value
    return prompt_value, completion_value, total_value


async def stream_graph_execution(
    agent: Any,
    state: dict[str, Any],
    config: dict[str, Any],
    on_telemetry: Callable[[StreamTelemetry], None],
) -> dict[str, Any]:
    """Run a graph through its event stream and return its final root state."""
    root_run_id: str | None = None
    final_state: dict[str, Any] | None = None
    active_agent = "orchestrator"
    milestone_progress = 30
    model_turns: defaultdict[str, int] = defaultdict(int)
    delegation_counts: defaultdict[str, int] = defaultdict(int)
    delegated_runs: dict[str, str] = {}

    async for event in agent.astream_events(state, config=config, version="v2"):
        event_type = event.get("event", "")
        name = str(event.get("name", ""))
        run_id = event.get("run_id")
        parent_ids = event.get("parent_ids", [])

        # Attribute nested events by their delegated task ancestry. This keeps
        # telemetry correct even if the orchestrator overlaps sub-agent runs.
        parent_agent = next(
            (
                delegated_runs[str(parent_id)]
                for parent_id in reversed(parent_ids)
                if str(parent_id) in delegated_runs
            ),
            None,
        )
        if parent_agent is not None:
            active_agent = parent_agent

        if event_type == "on_chain_start" and not parent_ids and root_run_id is None:
            root_run_id = run_id

        if event_type == "on_tool_start":
            tool_input = _event_input(event)
            if name == "task":
                delegated_agent = tool_input.get("subagent_type")
                if delegated_agent in STAGE_START_PROGRESS:
                    active_agent = delegated_agent
                    milestone_progress = max(
                        milestone_progress,
                        STAGE_START_PROGRESS[delegated_agent],
                    )
                    if run_id is not None:
                        delegated_runs[str(run_id)] = delegated_agent
                    delegation_counts[delegated_agent] += 1
                    invocation = delegation_counts[delegated_agent]
                    on_telemetry(
                        StreamTelemetry(
                            agent=delegated_agent,
                            log_type="delegate",
                            message=(
                                f"Starting {delegated_agent} invocation "
                                f"{invocation}."
                            ),
                            progress=milestone_progress,
                            current_tool="task",
                            delegation_count=invocation,
                        )
                    )
            elif name == "internet_search":
                active_agent = "research-agent"
                query = str(tool_input.get("query", ""))[:180]
                on_telemetry(
                    StreamTelemetry(
                        agent=active_agent,
                        log_type="tool_call",
                        message=f"Searching the web: {query}",
                        progress=milestone_progress,
                        current_tool=name,
                    )
                )
            elif name == "read_file":
                path = tool_input.get("file_path") or tool_input.get("path") or "file"
                limit = tool_input.get("limit")
                limit_text = f" (up to {limit} lines)" if limit else ""
                on_telemetry(
                    StreamTelemetry(
                        agent=active_agent,
                        log_type="tool_call",
                        message=f"Reading {path}{limit_text}.",
                        progress=milestone_progress,
                        current_tool=name,
                    )
                )
            elif name == "grep":
                path = tool_input.get("path") or tool_input.get("file_path") or "files"
                pattern = str(tool_input.get("pattern", ""))[:120]
                on_telemetry(
                    StreamTelemetry(
                        agent=active_agent,
                        log_type="tool_call",
                        message=f"Selecting relevant evidence from {path}: {pattern}",
                        progress=milestone_progress,
                        current_tool=name,
                    )
                )
            elif name == "write_file":
                path = tool_input.get("file_path") or tool_input.get("path") or "file"
                file_agent = OUTPUT_AGENT.get(path, active_agent)
                active_agent = file_agent
                on_telemetry(
                    StreamTelemetry(
                        agent=file_agent,
                        log_type="file_write",
                        message=f"Writing {path}.",
                        progress=milestone_progress,
                        current_tool=name,
                    )
                )
            elif name == "calculator":
                on_telemetry(
                    StreamTelemetry(
                        agent=active_agent,
                        log_type="tool_call",
                        message="Running quantitative calculation.",
                        progress=milestone_progress,
                        current_tool=name,
                    )
                )

        elif event_type == "on_tool_end" and name == "task":
            completed_agent = delegated_runs.pop(str(run_id), active_agent)
            if completed_agent in STAGE_COMPLETE_PROGRESS:
                milestone_progress = max(
                    milestone_progress,
                    STAGE_COMPLETE_PROGRESS[completed_agent],
                )
                on_telemetry(
                    StreamTelemetry(
                        agent=completed_agent,
                        log_type="complete",
                        message=f"{completed_agent} returned control to the orchestrator.",
                        progress=milestone_progress,
                        current_tool=None,
                    )
                )
            active_agent = "orchestrator"

        elif event_type == "on_chat_model_start":
            model_turns[active_agent] += 1
            turn = model_turns[active_agent]
            estimated_prompt_tokens = _estimate_prompt_tokens(event)
            on_telemetry(
                StreamTelemetry(
                    agent=active_agent,
                    log_type="model",
                    message=f"Model turn {turn} started.",
                    progress=milestone_progress,
                    model_turn=turn,
                    model_turn_limit=AGENT_TURN_LIMITS.get(active_agent),
                    current_tool="model",
                    prompt_tokens=estimated_prompt_tokens,
                    token_count_is_estimate=estimated_prompt_tokens is not None,
                )
            )

        elif event_type == "on_chat_model_stream":
            on_telemetry(
                StreamTelemetry(
                    agent=active_agent,
                    log_type="model_activity",
                    message="",
                    progress=milestone_progress,
                    model_turn=model_turns[active_agent] or None,
                    model_turn_limit=AGENT_TURN_LIMITS.get(active_agent),
                    current_tool="model",
                    record_log=False,
                )
            )

        elif event_type == "on_chat_model_end":
            prompt_tokens, completion_tokens, total_tokens = _extract_token_usage(
                event
            )
            on_telemetry(
                StreamTelemetry(
                    agent=active_agent,
                    log_type="model_usage",
                    message="",
                    progress=milestone_progress,
                    model_turn=model_turns[active_agent] or None,
                    model_turn_limit=AGENT_TURN_LIMITS.get(active_agent),
                    current_tool="model",
                    record_log=False,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            )

        elif event_type == "on_custom_event" and name.startswith("scout_agent_"):
            event_data = event.get("data") or {}
            if not isinstance(event_data, dict):
                event_data = {}
            event_agent = str(event_data.get("agent", active_agent))
            active_agent = event_agent
            milestone_progress = max(
                milestone_progress,
                STAGE_START_PROGRESS.get(event_agent, milestone_progress),
            )
            attempt = int(event_data.get("attempt", 1))
            max_attempts = int(
                event_data.get("max_attempts", AGENT_MAX_RETRIES + 1)
            )
            timeout_seconds = int(
                event_data.get("timeout_seconds", AGENT_TIMEOUT_SECONDS)
            )
            title = AGENT_TITLES.get(event_agent, event_agent)

            if name == "scout_agent_backoff":
                backoff_seconds = int(event_data.get("backoff_seconds", 0))
                message = (
                    f"{title} attempt timed out. "
                    f"Retrying in {backoff_seconds} seconds."
                )
                activity_state = "backoff"
                current_tool = "backoff"
                log_type = "retry"
            elif name == "scout_agent_timeout":
                backoff_seconds = None
                message = (
                    f"{title} attempt {attempt}/{max_attempts} timed out."
                )
                activity_state = "timeout"
                current_tool = None
                log_type = "timeout"
            else:
                backoff_seconds = None
                message = (
                    f"{title} Agent attempt {attempt}/{max_attempts} started."
                )
                activity_state = "active"
                current_tool = "task"
                log_type = "attempt"

            on_telemetry(
                StreamTelemetry(
                    agent=event_agent,
                    log_type=log_type,
                    message=message,
                    progress=milestone_progress,
                    current_tool=current_tool,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    timeout_seconds=timeout_seconds,
                    backoff_seconds=backoff_seconds,
                    activity_state=activity_state,
                )
            )

        if (
            event_type == "on_chain_end"
            and root_run_id is not None
            and run_id == root_run_id
        ):
            output = (event.get("data") or {}).get("output")
            if isinstance(output, dict):
                final_state = output

    if final_state is None:
        raise RuntimeError("LangGraph event stream ended without a final root state")
    return final_state
