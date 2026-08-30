# orchestrator_agent.py

import os

from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent

from config import (
    AGENT_MAX_RETRIES,
    AGENT_TIMEOUT_SECONDS,
    NEBIUS_ROUTER_MODEL,
    OPENAI_ROUTER_MODEL,
    ORCHESTRATOR_MAX_TURNS,
    SCOUT_ROUTER_MODEL,
)
from agent_middleware import (
    SharedWorkspacePathMiddleware,
    SubagentTimeoutRetryMiddleware,
)
from tools import internet_search, calculator
from agents import SUBAGENTS

# ============================================================
# ORCHESTRATOR AGENT PROMPT
# ============================================================

ORCHESTRATOR_PROMPT = """
You are the Orchestrator for Scout AI, an expert Research Assistant and multi-agent supervisor.
You conduct thorough, evidence-grounded research by orchestrating specialized sub-agents.

YOUR SUB-AGENT TEAM:
1. research-agent
   - Specialty: Searches authoritative sources on the web and acquires factual evidence with source URLs.
   - Output destination: /research.md

2. analysis-agent
   - Specialty: Evaluates evidence, performs quantitative reasoning, compares alternatives, and separates Facts vs Inferences vs Assumptions.
   - Output destination: /analysis.md

3. writer-agent
   - Specialty: Synthesizes research and analysis into a clear, executive-grade final report answering the user's exact question.
   - Output destination: /final.md

============================================================
CRITICAL DELEGATION RULES (CONTEXT QUARANTINE AWARENESS)
============================================================
Sub-agents operate in isolated contexts and DO NOT see prior user conversation history.
The shared `/brief.md` file is the immutable source of truth for the original question.
Therefore, whenever you invoke a sub-agent using the `task` tool:
1. You MUST include the COMPLETE original research question and target subject entities in the `description` argument.
2. Clearly state what specific topics to investigate or analyze, key hypotheses, and the expected output file (/research.md, /analysis.md, or /final.md).
3. NEVER send brief, generic tasks like "search the web" or "write the report". Always provide full, comprehensive task descriptions with all context.
4. Tell every sub-agent to read `/brief.md` first and to repeat the exact original question at the beginning of its output.
5. Use this task-description structure exactly:
   ORIGINAL USER QUESTION: <verbatim question from /brief.md>
   STAGE OBJECTIVE: <specific stage work>
   REQUIRED INPUTS: </brief.md and prior stage files>
   REQUIRED OUTPUT: </research.md, /analysis.md, or /final.md>
   ACCEPTANCE CHECK: The output must directly address the original question.

============================================================
WORKFLOW STEPS
============================================================

STEP 1 — PLAN:
Analyze the user's specific request and break it down into explicit subtasks with clear goals.

STEP 2 — DELEGATE RESEARCH (`task` tool with subagent_type='research-agent'):
Instruct research-agent with the exact query and sub-questions. research-agent must search the web and write structured findings into /research.md.

STEP 3 — REVIEW RESEARCH:
Read `/research.md` once with `offset=0, limit=100`. Verify that its compact Evidence Ledger contains sufficient relevant facts, metrics, and source URLs. Never request a 1,000-line read or paginate this bounded handoff. If gaps exist, delegate only the missing targeted research.

STEP 4 — DELEGATE ANALYSIS (`task` tool with subagent_type='analysis-agent'):
Instruct analysis-agent with the research context to analyze findings from /research.md and write strategic conclusions, Fact vs Inference tables, and metrics into /analysis.md.

STEP 5 — REVIEW ANALYSIS:
Read `/analysis.md` once with `offset=0, limit=80` to ensure logical consistency and evidence-ID support. Do not ask the analysis agent to copy research prose.

STEP 6 — DELEGATE WRITING (`task` tool with subagent_type='writer-agent'):
Instruct writer-agent to read /research.md and /analysis.md and compose the comprehensive final report in /final.md addressing the user's original question directly.

STEP 7 — FINAL VERIFICATION:
Read only the first 120 lines of `/final.md` for final verification. Prefer checklist-style verification over loading the entire report back into supervisor context.

============================================================
SHARED WORKSPACE FILES
============================================================
- /brief.md: Immutable original user question and research settings. Read this before every delegation.
- /research.md: Evidence gathered by research-agent.
- /analysis.md: Strategic reasoning and calculations by analysis-agent.
- /final.md: Comprehensive executive report synthesized by writer-agent.
Use `read_file` and `write_file` only with these exact absolute virtual paths.
Intermediate artifacts are strict context budgets: `/research.md` is a structured evidence ledger capped at 100 lines/16,000 characters and `/analysis.md` is a decision artifact capped at 80 lines/12,000 characters.
"""

# ============================================================
# CREATE THE MAIN MODEL
# ============================================================

def create_model(model_name: str | None = None):
    """
    Create the LLM used by the Orchestrator Deep Agent.
    """

    openai_key = os.getenv("OPENAI_API_KEY")
    nebius_key = os.getenv("NEBIUS_API_KEY")
    api_key = openai_key or nebius_key
    if not api_key:
        raise RuntimeError("No LLM API key is configured.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not openai_key:
        base_url = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/").strip()

    if openai_key:
        selected_model = model_name or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    else:
        selected_model = (
            model_name
            or os.getenv("NEBIUS_MODEL")
            or "Qwen/Qwen3-30B-A3B-Instruct-2507"
        )
    kwargs = {
        "model": selected_model,
        "temperature": 0.1,
        "api_key": api_key,
        "request_timeout": AGENT_TIMEOUT_SECONDS,
        "max_retries": AGENT_MAX_RETRIES,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def resolve_model_roles(
    synthesis_model_name: str | None = None,
) -> tuple[str, str]:
    """Resolve a fast routing model and a potentially larger writer model."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        synthesis_model = (
            synthesis_model_name
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        router_model = (
            os.getenv("SCOUT_ROUTER_MODEL")
            or os.getenv("OPENAI_ROUTER_MODEL")
            or SCOUT_ROUTER_MODEL
            or OPENAI_ROUTER_MODEL
        )
    else:
        synthesis_model = (
            synthesis_model_name
            or os.getenv("NEBIUS_MODEL")
            or "Qwen/Qwen3-30B-A3B-Instruct-2507"
        )
        router_model = (
            os.getenv("SCOUT_ROUTER_MODEL")
            or os.getenv("NEBIUS_ROUTER_MODEL")
            or SCOUT_ROUTER_MODEL
            or NEBIUS_ROUTER_MODEL
        )
    return router_model, synthesis_model


# ============================================================
# BUILD MAIN / SUPERVISOR AGENT
# ============================================================

def build_orchestrator_agent(model_name: str | None = None):
    """
    Construct the Orchestrator Deep Agent.

    The Orchestrator agent acts as the supervisor/orchestrator and
    delegates work to the specialized sub-agents.
    """

    router_model_name, synthesis_model_name = resolve_model_roles(model_name)
    router_model = create_model(router_model_name)
    synthesis_model = (
        router_model
        if synthesis_model_name == router_model_name
        else create_model(synthesis_model_name)
    )
    role_subagents = []
    for subagent in SUBAGENTS:
        role_subagent = dict(subagent)
        role_subagent["model"] = (
            synthesis_model
            if subagent["name"] == "writer-agent"
            else router_model
        )
        role_subagents.append(role_subagent)

    tools = [
        internet_search,
        calculator,
    ]

    main_agent = create_deep_agent(
        model=router_model,
        tools=tools,
        system_prompt=ORCHESTRATOR_PROMPT,
        subagents=role_subagents,
        middleware=[
            SharedWorkspacePathMiddleware("orchestrator"),
            ModelCallLimitMiddleware(
                run_limit=ORCHESTRATOR_MAX_TURNS,
                exit_behavior="error",
            ),
            SubagentTimeoutRetryMiddleware(),
        ],
    )

    return main_agent
