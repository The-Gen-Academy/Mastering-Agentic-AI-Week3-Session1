import os

from dotenv import load_dotenv
import truststore

# Use the Windows trust store for every HTTPS client initialized by the app,
# including LangSmith, Tavily, and OpenAI-compatible model providers.
truststore.inject_into_ssl()

load_dotenv()

# Provider configuration is validated when a live research task starts so the
# web application can still launch and offer an explicitly labeled simulation.
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
#OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
NEBIUS_MODEL = os.getenv("NEBIUS_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Shared reliability policy for research, analysis, and writer agents.
AGENT_TIMEOUT_SECONDS = 60
AGENT_MAX_RETRIES = 2
RESEARCH_AGENT_MAX_TURNS = 10
ANALYSIS_AGENT_MAX_TURNS = 7
WRITER_AGENT_MAX_TURNS = 10
ORCHESTRATOR_MAX_TURNS = 12
AGENT_MAX_CONCURRENCY = 5

# LangGraph counts node transitions, not just model calls. A model turn plus
# its tool node commonly consumes two graph steps, so the framework default of
# 25 conflicts with the 12-turn orchestrator once delegation overhead is added.
# Per-agent ModelCallLimitMiddleware remains the stricter loop guard.
GRAPH_RECURSION_LIMIT = 50

# Context budgets for artifacts passed between stages. Line limits make the
# contracts easy for agents to follow; character limits provide a hard safety
# net for unusually wide Markdown tables or long URLs.
RESEARCH_ARTIFACT_MAX_LINES = 100
RESEARCH_ARTIFACT_MAX_CHARS = 16_000
ANALYSIS_ARTIFACT_MAX_LINES = 80
ANALYSIS_ARTIFACT_MAX_CHARS = 12_000
BRIEF_READ_MAX_LINES = 60
FINAL_READ_MAX_LINES = 120
WRITER_RESEARCH_FALLBACK_MAX_LINES = 40
SEARCH_MAX_RESULTS_PER_QUERY = 3
SEARCH_RESULT_SNIPPET_CHARS = 600

# The supervisor, research agent, and analysis agent use the routing model.
# The writer keeps the selected primary model for final synthesis. Provider-
# specific variables can override this without changing the user's final model.
OPENAI_ROUTER_MODEL = os.getenv("OPENAI_ROUTER_MODEL", "gpt-4o-mini")
NEBIUS_ROUTER_MODEL = os.getenv("NEBIUS_ROUTER_MODEL","Qwen/Qwen3-30B-A3B-Instruct-2507")
#NEBIUS_ROUTER_MODEL = os.getenv("NEBIUS_ROUTER_MODEL", "Meta-Llama-3.1-8B-Instruct-fast")
SCOUT_ROUTER_MODEL = os.getenv("SCOUT_ROUTER_MODEL")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "scout-ai")
LANGSMITH_TRACING = os.getenv(
    "LANGSMITH_TRACING",
    "true" if LANGSMITH_API_KEY else "false",
)

if LANGSMITH_API_KEY:
    # LangChain and DeepAgents automatically emit nested model, tool, and
    # sub-agent runs when these standard LangSmith variables are present.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
