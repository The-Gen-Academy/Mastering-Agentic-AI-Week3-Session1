"""DeepAgents virtual-filesystem state helpers."""

from collections.abc import Mapping
from typing import Any

from deepagents.backends.utils import create_file_data, file_data_to_string

from config import (
    ANALYSIS_ARTIFACT_MAX_CHARS,
    ANALYSIS_ARTIFACT_MAX_LINES,
    RESEARCH_ARTIFACT_MAX_CHARS,
    RESEARCH_ARTIFACT_MAX_LINES,
)


STAGE_FILENAMES = ("research.md", "analysis.md", "final.md")
ARTIFACT_BUDGETS = {
    "research.md": (
        RESEARCH_ARTIFACT_MAX_LINES,
        RESEARCH_ARTIFACT_MAX_CHARS,
    ),
    "analysis.md": (
        ANALYSIS_ARTIFACT_MAX_LINES,
        ANALYSIS_ARTIFACT_MAX_CHARS,
    ),
}
CAP_NOTICE = "[Scout AI capped this stage artifact to its context budget.]"


def cap_stage_artifact(filename: str, content: str) -> str:
    """Bound research/analysis artifacts without splitting a Markdown line."""
    budget = ARTIFACT_BUDGETS.get(filename.lstrip("/"))
    if budget is None or not isinstance(content, str):
        return content

    max_lines, max_chars = budget
    lines = content.splitlines()
    if len(lines) <= max_lines and len(content) <= max_chars:
        return content

    kept = lines[: max_lines - 1]
    available_chars = max_chars - len(CAP_NOTICE) - 1
    bounded = "\n".join(kept)
    if len(bounded) > available_chars:
        bounded = bounded[:available_chars]
        if "\n" in bounded:
            bounded = bounded.rsplit("\n", 1)[0]
        else:
            bounded = bounded.rstrip()
    return f"{bounded}\n{CAP_NOTICE}" if bounded else CAP_NOTICE


def cap_stage_artifacts(artifacts: Mapping[str, str]) -> dict[str, str]:
    """Return a copy with every intermediate artifact inside its budget."""
    return {
        filename: cap_stage_artifact(filename, content)
        for filename, content in artifacts.items()
    }


def build_agent_file_state(
    brief: str,
    artifacts: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build canonical FileData entries for the DeepAgents StateBackend."""
    artifacts = cap_stage_artifacts(artifacts or {})
    content_by_path = {
        "/brief.md": brief,
        "/research.md": artifacts.get("research.md", ""),
        "/analysis.md": artifacts.get("analysis.md", ""),
        "/final.md": artifacts.get("final.md", ""),
    }
    return {
        path: create_file_data(content)
        for path, content in content_by_path.items()
    }


def get_agent_file_content(files: Mapping[str, Any], filename: str) -> str:
    """Read a current FileData entry or a legacy raw-string entry safely."""
    raw_value = files.get(f"/{filename}")
    if raw_value is None:
        raw_value = files.get(filename)

    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        return raw_value
    if isinstance(raw_value, dict):
        return file_data_to_string(raw_value)

    raise TypeError(
        f"Unexpected agent file value for {filename}: "
        f"{type(raw_value).__name__}"
    )
