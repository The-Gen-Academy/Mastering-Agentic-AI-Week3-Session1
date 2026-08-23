# LangGraph Workshop: Building Agents, From Scratch to Multi-Agent

A hands-on, three-notebook workshop that builds up from a hand-wired ReAct agent to a composed multi-agent system, using **LangGraph** and **LangChain 1.0**. The running example throughout is an **Employee Assistant** that handles time-off requests and expense reimbursements.

## What you'll build

The notebooks are meant to be worked through in order — each one reuses the ideas (and code) from the one before it.

### 1. `1_Langgraph_React_Agent.ipynb` — LangGraph fundamentals

Build a **ReAct** (Reason + Act) agent by hand, so you understand every piece before any framework hides it:

- **Tools** — ordinary Python functions exposed to the LLM via the `@tool` decorator (the docstring *is* the prompt).
- **Graph state** — a `TypedDict` with an `add_messages` reducer that appends to the conversation.
- **Nodes & edges** — a `reason_node` (the LLM decides) and an `action_node` (Python runs the tool), wired into a loop with a conditional router.
- **Human-in-the-loop** — `interrupt()` pauses the entire graph so the agent can *ask a human* when it's missing information (e.g. leave dates), then resume with `Command(resume=...)`.
- **Checkpointers** — `InMemorySaver` snapshots state per `thread_id`, which is what makes pause/resume and state inspection possible.

### 2. `2_Langchain_CreateAgent.ipynb` — the high-level shortcut

Rebuild the same agent with LangChain's `create_agent` — a pre-assembled LangGraph that wires the model→tools loop for you. Then layer on **middleware** (`HumanInTheLoopMiddleware`) to add approval gates around tool calls.

### 3. `3_Multi_Agent_Workflow.ipynb` — composing a system

Combine both approaches into a **router-pattern multi-agent workflow**:

- An **intent classifier** (a single structured-output LLM call) routes each request.
- A **time-off agent** (reused untouched from notebook 2).
- An **expense agent** that does light **agentic RAG** — searching company policy documents (`policies/`) embedded into an in-memory vector store — and submits claims against auto-approval limits.
- A **fallback node** for off-topic requests, so the system never hallucinates.

All assembled into one parent `StateGraph` where interrupts raised deep inside a sub-agent surface cleanly through the parent.

## Project layout

```
.
├── 1_Langgraph_React_Agent.ipynb   # ReAct from scratch
├── 2_Langchain_CreateAgent.ipynb   # create_agent + middleware
├── 3_Multi_Agent_Workflow.ipynb    # multi-agent router + RAG
├── langraph_prompts.py             # shared prompt template
├── policies/                       # company policy docs used for RAG
│   ├── equipment.md
│   ├── meals.md
│   └── travel.md
├── images/                         # diagrams referenced by the notebooks
├── pyproject.toml                  # dependencies (Python >=3.12)
└── uv.lock
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

1. **Install dependencies** (creates the `.venv` and installs from the lockfile):

   ```bash
   uv sync
   ```

2. **Configure environment variables.** Create a `.env` file in the project root:

   ```bash
   # Required — used for the LLM calls (gpt-4o) and embeddings
   OPENAI_API_KEY=sk-...

   # Optional — only needed if you turn on LangSmith tracing
   LANGSMITH_API_KEY=ls-...
   ```

   The notebooks load this with `load_dotenv(override=True)`. To enable tracing, uncomment the `LANGSMITH_TRACING` / `LANGSMITH_PROJECT` lines in notebook 1.

3. **Run the notebooks.** Launch Jupyter (or open them in VS Code) using the project's virtual environment as the kernel:

   ```bash
   uv run jupyter lab
   ```

   Work through them in order: `1` → `2` → `3`.

## Requirements

- Python ≥ 3.12
- An OpenAI API key (the notebooks use `gpt-4o` and OpenAI embeddings)
- Key libraries: `langchain` 1.x, `langgraph` 1.x, `langchain-openai`, `langsmith`

## Key concepts covered

| Concept | Where |
|---|---|
| ReAct loop (reason → act → observe) | Notebook 1 |
| Tools via `@tool` | Notebook 1 |
| State + `add_messages` reducer | Notebook 1 |
| Nodes, edges, conditional routing | Notebook 1 |
| Interrupts & `Command(resume=...)` | Notebooks 1–3 |
| Checkpointers (`InMemorySaver`, `thread_id`) | Notebooks 1–3 |
| `create_agent` | Notebooks 2–3 |
| Middleware (`HumanInTheLoopMiddleware`) | Notebook 2 |
| Structured-output intent classification | Notebook 3 |
| Agentic RAG (vector store as a tool) | Notebook 3 |
| Multi-agent composition (router pattern) | Notebook 3 |

## Notes

- `.env` and `.venv/` are gitignored — don't commit your API keys.
- The vector store and checkpointers are **in-memory**, so state resets when the kernel restarts. This is intentional for a workshop; production would use a persistent vector store and a durable checkpointer.
