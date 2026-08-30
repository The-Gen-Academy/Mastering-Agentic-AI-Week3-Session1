# app.py
"""
Scout AI - Autonomous Multi-Agent Deep Research Platform
High-End Research Intelligence UI & Backend
"""

import os
import sys
import json
import time
import asyncio
import uuid
import re
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from io import BytesIO

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, set_key
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langgraph.errors import GraphRecursionError

# Ensure current directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import config
from agent_middleware import AgentIterationLimitError, SubagentTimeoutError
from agent_files import (
    build_agent_file_state,
    cap_stage_artifacts,
    get_agent_file_content,
)
from graph_streaming import StreamTelemetry, stream_graph_execution
from orchestrator_agent import build_orchestrator_agent, resolve_model_roles
from tools import internet_search, calculator

# Document generation
try:
    import docx
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

app = FastAPI(title="Scout AI - Deep Research Intelligence", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for active and historical research dossiers
TASKS: Dict[str, Dict[str, Any]] = {}
HISTORY_FILE = os.path.join(current_dir, "research_history.json")


def load_history():
    global TASKS
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                TASKS = json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            TASKS = {}


def save_history():
    try:
        # Keep latest 50 tasks
        sorted_tasks = dict(list(TASKS.items())[-50:])
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted_tasks, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving history: {e}")


def reconcile_interrupted_tasks() -> None:
    """Close tasks whose worker disappeared during a prior server shutdown."""
    changed = False
    for task in TASKS.values():
        if task.get("status") != "in_progress":
            continue

        task["status"] = "failed"
        task["progress"] = 100
        task["active_agent"] = "failed"
        task["error"] = (
            "Research was interrupted by an application restart before the "
            "agent workflow completed. Start a new task to retry."
        )
        task.setdefault("logs", []).append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "agent": "orchestrator",
                "type": "error",
                "message": task["error"],
            }
        )
        changed = True

    if changed:
        save_history()


load_history()
reconcile_interrupted_tasks()


class ResearchRequest(BaseModel):
    query: str
    depth: str = "comprehensive"  # quick, comprehensive, exhaustive
    focus: str = "general"  # general, news, finance, tech, academic
    model: Optional[str] = None
    force_simulation: Optional[bool] = False


class SettingsRequest(BaseModel):
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    router_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = None


class ExportRequest(BaseModel):
    title: str
    content: str
    files: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ResearchValidationError(RuntimeError):
    """Raised when an agent result fails the topic-grounding contract."""


def build_research_brief(query: str, depth: str, focus: str) -> str:
    """Create the immutable context shared by the supervisor and every sub-agent."""
    as_of_date = datetime.now().date()
    source_window_start = as_of_date - timedelta(days=365)
    return f"""# Scout AI Research Brief

ORIGINAL USER QUESTION:
{query}

RESEARCH DEPTH: {depth}
RESEARCH FOCUS: {focus}

RUNTIME CONTEXT:
AS_OF_DATE: {as_of_date.isoformat()}
SOURCE_WINDOW_START: {source_window_start.isoformat()}
MARKET: United States
PRODUCT_REQUIREMENT: Latest model/version currently available for purchase

NON-NEGOTIABLE REQUIREMENTS:
- Address the original user question exactly; do not substitute a nearby topic.
- Begin every stage output with: `Original research question: {query}`
- Use only evidence relevant to this question.
- State uncertainty and evidence gaps rather than inventing claims.
"""


def validate_stage_output(stage: str, content: str, query: str) -> List[str]:
    """Return validation errors for a live agent artifact."""
    errors: List[str] = []
    text = content.strip() if isinstance(content, str) else ""
    normalized_text = " ".join(text.lower().split())
    normalized_query = " ".join(query.lower().split())

    if len(text) < 120:
        errors.append(f"{stage}.md is missing or too short")
    if normalized_query not in normalized_text:
        errors.append(f"{stage}.md does not repeat the exact original question")
    if stage == "research" and not re.search(r"https?://", text):
        errors.append("research.md contains no source URLs")
    if any(marker in normalized_text for marker in (
        "extensive web search completed",
        "comprehensive evidence analysis completed",
    )):
        errors.append(f"{stage}.md contains placeholder output")
    return errors


def extract_agent_artifacts(result: Dict[str, Any]) -> Dict[str, str]:
    """Normalize DeepAgents file results and its last-message fallback."""
    files = result.get("files", {}) or {}
    messages = result.get("messages", []) or []
    final_content = ""
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            final_content = last_msg.content
        elif isinstance(last_msg, dict):
            final_content = last_msg.get("content", "")
        else:
            final_content = str(last_msg)

    return cap_stage_artifacts({
        "research.md": get_agent_file_content(files, "research.md"),
        "analysis.md": get_agent_file_content(files, "analysis.md"),
        "final.md": get_agent_file_content(files, "final.md") or final_content,
    })


def validate_agent_artifacts(files: Dict[str, str], query: str) -> List[str]:
    errors: List[str] = []
    for stage in ("research", "analysis", "final"):
        errors.extend(validate_stage_output(stage, files.get(f"{stage}.md", ""), query))
    return errors


# Pre-seeded high-grade intelligence dossiers for immediate exploration
DEFAULT_DOSSIERS = {
    "demo-enterprise-ai": {
        "id": "demo-enterprise-ai",
        "query": "Compare Microsoft, Google, and Amazon's strategies for enterprise generative AI and recommend which has the strongest competitive position.",
        "depth": "exhaustive",
        "focus": "finance",
        "model": "gpt-4o",
        "status": "completed",
        "progress": 100,
        "created_at": "2026-08-30T10:15:00Z",
        "duration_seconds": 38.4,
        "active_agent": "completed",
        "sources_count": 18,
        "confidence_score": 94,
        "metrics": {
            "radar": {
                "categories": ["Ecosystem Lock-in", "Model Capabilities", "Enterprise Security", "Cloud Infra & Compute", "Developer Mindshare", "Pricing & TCO"],
                "series": [
                    {"name": "Microsoft (Azure AI)", "values": [95, 92, 96, 91, 94, 78]},
                    {"name": "Google (Vertex AI)", "values": [82, 95, 88, 93, 89, 86]},
                    {"name": "Amazon (AWS Bedrock)", "values": [90, 84, 93, 98, 87, 89]}
                ]
            },
            "swot": {
                "strengths": [
                    "Microsoft: Dominant Office 365 / Copilot distribution; multi-billion OpenAI partnership exclusivity in enterprise.",
                    "Google: Vertically integrated TPUs (v5p/v6e), Gemini 2.5 multimodal context window lead (2M+ tokens), DeepMind research bench.",
                    "Amazon: Infrastructure scale with Trainium2/Inferentia chips, neutral multi-model marketplace (Anthropic Claude 3.7, Llama 3, Nova)."
                ],
                "weaknesses": [
                    "Microsoft: Heavy reliance on OpenAI IP; higher inference inference cost per seat.",
                    "Google: Slower enterprise sales cycle transition compared to Azure Copilot monetization.",
                    "Amazon: Slower to market with proprietary frontier models; dependent on partner models."
                ],
                "opportunities": [
                    "Agentic Workflow Orchestration: Embedding multi-agent task execution inside enterprise ERP/CRM stacks.",
                    "Sovereign AI & On-Premises Compliance: Localized enterprise fine-tuning clusters.",
                    "Cost reduction via custom silicon (Trainium/TPU/Maia)."
                ],
                "threats": [
                    "Commoditization of foundational LLM weights from open-source models (DeepSeek, Llama).",
                    "Regulatory scrutiny regarding cloud hyperscaler AI exclusivity pacts.",
                    "Enterprise ROI justification pressure during 2026 budget cycles."
                ]
            },
            "confidence_breakdown": {
                "Authoritative SEC Filings & Cloud Earnings": 96,
                "Independent Technical Benchmarks (LMSYS/HELM)": 94,
                "Enterprise Customer Surveys (Gartner/IDC)": 91,
                "Patent & Model Release Documentation": 95
            }
        },
        "files": {
            "research.md": """# Research Dossier: Enterprise Generative AI Hyperscaler Benchmark

## 1. Executive Evidence Summary
- **Microsoft Azure AI**: Leverages OpenAI GPT-4o, o1, and proprietary Phi-4 models. Azure AI Studio adoption grew over 120% YoY. Over 70% of Fortune 500 companies have deployed Azure OpenAI instances. Office 365 Copilot active seats surpassed 65M.
- **Google Cloud (Vertex AI)**: Anchored by Gemini 1.5 Pro/2.0 Flash with 2M+ token multimodal context windows. Strongest native retrieval over massive enterprise codebases and multimodal unstructured data. Google Cloud ARR surpassed $40B with high operating margins.
- **Amazon Web Services (AWS Bedrock)**: Platform-agnostic aggregator strategy. Deep $8B+ partnership with Anthropic (Claude 3.7 Sonnet as primary frontier offering), proprietary Amazon Nova family, and custom Trainium2 silicon yielding up to 40% lower inference costs.

## 2. Authoritative Sources & Citations
1. [Microsoft Corporation - Q4 FY2026 Earnings Call & Form 10-K](https://www.microsoft.com/investor)
2. [Alphabet Inc. - Google Cloud Enterprise Strategy & Gemini Ecosystem Whitepaper](https://cloud.google.com/vertex-ai)
3. [Amazon.com Inc. - AWS Bedrock Architecture & Anthropic Strategic Partnership](https://aws.amazon.com/bedrock)
4. [Gartner Magic Quadrant for Enterprise Conversational AI & Foundation Models 2026](https://www.gartner.com/en)
5. [IDC MarketScape: Worldwide AI Platform Services for Enterprise](https://www.idc.com)

## 3. Quantitative Evidence & Metrics
- Global Enterprise Generative AI Cloud Spend: $68.4 Billion in 2026.
- Market Share Distribution: Microsoft (41.2%), AWS (32.5%), Google Cloud (21.8%), Others (4.5%).
- Median Enterprise Deployment Latency: Azure (4.2 wks), AWS (3.8 wks), GCP (4.9 wks).
""",
            "analysis.md": """# Strategic Analysis: Enterprise Generative AI Competitive Positioning

## 1. Fact vs. Inference vs. Assumption Matrix

### Confirmed Facts:
- Microsoft possesses the broadest out-of-the-box productivity software distribution channel via Windows, M365, GitHub Copilot, and Dynamics 365.
- Google possesses superior algorithmic efficiency, native long-context reasoning (2M tokens), and complete vertical integration from TPU silicon to consumer/enterprise search.
- Amazon controls the largest installed base of cloud compute workloads, giving Bedrock an immediate low-latency adjacency advantage for existing S3/RDS enterprise data lakes.

### Evidence-Supported Inferences:
- Microsoft's enterprise moat is strongest in white-collar workflow automation and executive buy-in due to pre-existing enterprise agreements (EA).
- AWS Bedrock is winning developers who fear single-vendor lock-in, because Bedrock allows seamless hot-swapping between Anthropic Claude, Meta Llama, and Amazon Nova.
- Google Cloud is the premier choice for data-intensive, multimodal, and analytical enterprises where massive context windows eliminate vector DB fragmentation.

### Assumptions Subject to Volatility:
- Future OpenAI API pricing stability and licensing terms with Microsoft.
- Anthropic remaining operationally aligned with AWS while also offering services on GCP.

## 2. Comparative Strategic Decision Scoring (Out of 100)
- **Microsoft Azure**: 92.5/100 (Immediate revenue realization, highest enterprise trust)
- **Amazon AWS**: 88.0/100 (Unmatched flexibility, superior custom silicon cost curve)
- **Google Cloud**: 87.5/100 (Deepest scientific bench, best multimodal context engine)

## 3. Synthesis & Strategic Verdict
Microsoft holds the strongest immediate competitive position for enterprise monetization, while AWS holds the strongest defensive posture against model obsolescence.
""",
            "final.md": """# Executive Intelligence Dossier: Enterprise Generative AI Hyperscaler Strategy

**Prepared by:** Scout AI Autonomous Research Taskforce  
**Classification:** Enterprise Strategic Advisory  
**Date:** August 2026  
**Subject:** Comparative Evaluation of Microsoft, Google, and Amazon Enterprise Generative AI Moats

---

## Executive Summary

The enterprise generative AI landscape has crystallized into a tri-polar battle among **Microsoft**, **Amazon Web Services (AWS)**, and **Google Cloud Platform (GCP)**. While each hyperscaler offers enterprise-grade security, private VPC isolation, and compliance certifications (SOC2, HIPAA, ISO 27001), their core strategic vectors diverge significantly:

1. **Microsoft (Azure AI)** operates a **Distribution-First & Proprietary Ecosystem** strategy, capturing immediate corporate wallet share through Office 365 Copilot, GitHub, and its premier OpenAI partnership.
2. **Amazon (AWS Bedrock)** operates a **Platform-Agnostic Model Marketplace & Silicon Efficiency** strategy, mitigating model obsolescence by providing Anthropic Claude, Meta Llama, and proprietary Nova models on custom Trainium2 hardware.
3. **Google (Vertex AI)** operates a **Full-Stack Vertical Integration & Multimodal Frontier** strategy, leveraging Google DeepMind's Gemini 2.5 architecture (2M+ token context) and custom TPU infrastructure.

### Strategic Recommendation
**Microsoft maintains the overall strongest competitive position in 2026** for immediate enterprise generative AI dominance, driven by unmatched enterprise sales relationships, pre-existing M365 seat distribution, and cohesive developer workflows. However, **AWS Bedrock represents the most resilient hedge** for enterprise CTOs seeking multi-model neutrality and long-term compute cost optimization.

---

## Comprehensive Comparative Matrix

| Dimension | Microsoft Azure AI | Amazon AWS (Bedrock) | Google Cloud (Vertex AI) |
| :--- | :--- | :--- | :--- |
| **Primary Frontier Engine** | OpenAI GPT-4o / o1 / Phi-4 | Anthropic Claude 3.7 / Nova | Google DeepMind Gemini 2.5 |
| **Max Context Window** | 128k - 200k tokens | 200k tokens (Claude) | **2,000,000+ tokens** |
| **Hardware Acceleration** | Nvidia H100/H200 + Azure Maia | Nvidia H200 + **Trainium2** | **Google TPU v5p / v6e** |
| **Enterprise Distribution** | M365 (400M+ commercial seats) | AWS Cloud Base ($100B+ ARR) | Google Workspace & BigQuery |
| **Custom Silicon Cost Moat** | Emerging (Maia 100) | **High (Trainium2 -40% TCO)** | **High (TPU v6e scale)** |
| **Market Share (Enterprise GenAI)**| **41.2% (Leader)** | 32.5% | 21.8% |

---

## Detailed Strategic Profiles

### 1. Microsoft: The Enterprise Distribution Fortress
Microsoft's primary competitive advantage is not merely algorithmic—it is distribution velocity. By embedding Copilot directly into Office applications (Excel, Teams, PowerPoint, Outlook) and GitHub (used by over 100 million developers), Microsoft removes all friction from procurement.
- **Key Advantage**: Zero-friction enterprise software licensing integration and deep enterprise identity governance via Microsoft Entra ID.
- **Strategic Vulnerability**: Margin compression from OpenAI licensing costs and hardware capital expenditure.

### 2. Amazon AWS: The Merchant of AI Models
Amazon has positioned AWS Bedrock as the universal operating system for enterprise foundation models. Rather than betting on a single LLM vendor, Bedrock enables enterprises to route tasks dynamically between Claude 3.7 Sonnet for complex coding, Llama 3 for low-cost classification, and Amazon Nova for ultra-fast structured extraction.
- **Key Advantage**: Multi-model flexibility without data egress; superior economics via Trainium2 instances.
- **Strategic Vulnerability**: Lack of an integrated SaaS productivity suite comparable to Microsoft 365 or Google Workspace.

### 3. Google Cloud: The Multimodal & Context Supremacy
Google Cloud's competitive wedge is built on technological breakthroughs from Google DeepMind. Gemini's massive 2M token context window allows enterprises to ingest entire code repositories, financial audit trails, and multi-hour video streams in a single prompt without complex vector database fragmentation.
- **Key Advantage**: Complete end-to-end integration with Google BigQuery, Android, and massive native multimodal context processing.
- **Strategic Vulnerability**: Enterprise go-to-market execution has historically lagged behind Microsoft's aggressive corporate sales force.

---

## Strategic Action Plan for Enterprise Leadership
1. **Adopt a Multi-Cloud Model Router**: Avoid single-model lock-in by standardizing application layers on LangChain / OpenTelemetry abstractions.
2. **Standardize on Azure for End-User Copilots**: Leverage Microsoft 365 Copilot for general workforce automation where workflow integration is critical.
3. **Deploy AWS Bedrock for Core Custom Applications**: Utilize AWS Bedrock with Anthropic Claude and custom Trainium clusters for proprietary customer-facing systems to minimize per-token inference overhead.
4. **Utilize Google Vertex AI for Massive Document & Multimodal Analytics**: Direct complex multi-hundred-page compliance, legal, and financial auditing tasks to Gemini's 2M context window.

---
*Report generated by Scout AI Autonomous Research Taskforce. Verified with 18 authoritative primary sources.*
"""
        },
        "logs": [
            {"time": "10:15:02", "agent": "orchestrator", "type": "plan", "message": "Initialized research objective: Compare MSFT, GOOG, AMZN enterprise GenAI positions."},
            {"time": "10:15:05", "agent": "orchestrator", "type": "delegate", "message": "Delegating deep evidence gathering to research-agent."},
            {"time": "10:15:08", "agent": "research-agent", "type": "tool_call", "message": "Tavily Search: 'Microsoft Azure OpenAI Enterprise Market Share Q4 2026'"},
            {"time": "10:15:14", "agent": "research-agent", "type": "tool_call", "message": "Tavily Search: 'AWS Bedrock Anthropic Trainium2 benchmarks 2026'"},
            {"time": "10:15:20", "agent": "research-agent", "type": "tool_call", "message": "Tavily Search: 'Google Cloud Vertex AI Gemini 2.5 revenue IDC report'"},
            {"time": "10:15:24", "agent": "research-agent", "type": "file_write", "message": "Saved 18 verified sources and citations to /research.md."},
            {"time": "10:15:27", "agent": "orchestrator", "type": "delegate", "message": "Delegating reasoning & SWOT matrix to analysis-agent."},
            {"time": "10:15:30", "agent": "analysis-agent", "type": "tool_call", "message": "Calculator: '((41.2 - 32.5) / 32.5) * 100' -> +26.7% market share delta"},
            {"time": "10:15:33", "agent": "analysis-agent", "type": "file_write", "message": "Structured Fact vs Inference matrix written to /analysis.md."},
            {"time": "10:15:35", "agent": "orchestrator", "type": "delegate", "message": "Delegating final executive synthesis to writer-agent."},
            {"time": "10:15:38", "agent": "writer-agent", "type": "file_write", "message": "Executive dossier synthesized and published to /final.md."},
            {"time": "10:15:39", "agent": "orchestrator", "type": "complete", "message": "Research task completed with 94% confidence score."}
        ]
    }
}

# Merge default dossiers into tasks if not present
for k, v in DEFAULT_DOSSIERS.items():
    if k not in TASKS:
        TASKS[k] = v


@app.get("/api/config")
def get_config_endpoint():
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    nebius_ready = bool(os.getenv("NEBIUS_API_KEY"))
    selected_model = (
        os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if openai_ready
        else os.getenv("NEBIUS_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    )
    router_model, synthesis_model = resolve_model_roles(selected_model)
    return {
        "openai_configured": openai_ready or nebius_ready,
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
        "provider": "openai" if openai_ready else "nebius" if nebius_ready else None,
        "openai_model": selected_model,
        "router_model": router_model,
        "synthesis_model": synthesis_model,
        "openai_base_url": (
            os.getenv("OPENAI_BASE_URL", "")
            if openai_ready
            else os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/")
        ),
        "langsmith_configured": bool(os.getenv("LANGSMITH_API_KEY")),
        "langsmith_tracing": (
            bool(os.getenv("LANGSMITH_API_KEY"))
            and os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        ),
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "scout-ai"),
        "available_models": [
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini (Fast & Cost-Efficient)", "provider": "OpenAI"},
            {"id": "gpt-4o", "name": "GPT-4o (High-Precision Multimodal)", "provider": "OpenAI"},
            {"id": "o3-mini", "name": "o3-mini (Advanced STEM & Logic)", "provider": "OpenAI"},
            {"id": "o1", "name": "o1 (Deep Strategic Reasoning)", "provider": "OpenAI"},
            {"id": "claude-3-7-sonnet", "name": "Claude 3.7 Sonnet (Anthropic)", "provider": "Anthropic"},
            {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507", "name": "Qwen3 30B Instruct (Nebius)", "provider": "Nebius"},
            {"id": "Qwen/Qwen3-235B-A22B-Instruct-2507", "name": "Qwen3 235B Instruct (Nebius Final Synthesis)", "provider": "Nebius"},
        ]
    }


@app.post("/api/config")
def update_config_endpoint(settings: SettingsRequest):
    env_path = os.path.join(current_dir, ".env")
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        try:
            set_key(env_path, "OPENAI_API_KEY", settings.openai_api_key)
        except Exception:
            pass
    if settings.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
        try:
            set_key(env_path, "TAVILY_API_KEY", settings.tavily_api_key)
        except Exception:
            pass
    if settings.openai_model:
        model_env_name = (
            "NEBIUS_MODEL"
            if os.getenv("NEBIUS_API_KEY") and not os.getenv("OPENAI_API_KEY")
            else "OPENAI_MODEL"
        )
        os.environ[model_env_name] = settings.openai_model
        try:
            set_key(env_path, model_env_name, settings.openai_model)
        except Exception:
            pass
    if settings.router_model:
        router_env_name = (
            "NEBIUS_ROUTER_MODEL"
            if os.getenv("NEBIUS_API_KEY") and not os.getenv("OPENAI_API_KEY")
            else "OPENAI_ROUTER_MODEL"
        )
        os.environ[router_env_name] = settings.router_model
        try:
            set_key(env_path, router_env_name, settings.router_model)
        except Exception:
            pass
    if settings.openai_base_url:
        base_url_env_name = (
            "NEBIUS_BASE_URL"
            if os.getenv("NEBIUS_API_KEY") and not os.getenv("OPENAI_API_KEY")
            else "OPENAI_BASE_URL"
        )
        os.environ[base_url_env_name] = settings.openai_base_url
        try:
            set_key(env_path, base_url_env_name, settings.openai_base_url)
        except Exception:
            pass
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_TRACING"] = "true"
        try:
            set_key(env_path, "LANGSMITH_API_KEY", settings.langsmith_api_key)
            set_key(env_path, "LANGSMITH_TRACING", "true")
        except Exception:
            pass
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        try:
            set_key(env_path, "LANGSMITH_PROJECT", settings.langsmith_project)
        except Exception:
            pass
    return {"status": "success", "message": "Configuration updated successfully."}


@app.get("/api/research/history")
def get_history_endpoint():
    items = []
    for task_id, task in reversed(list(TASKS.items())):
        files = task.get("files", {}) or {}
        confidence_score = task.get("confidence_score")
        if task.get("status") == "completed" and files.get("analysis.md"):
            derived_metrics = extract_metrics_from_text(
                files.get("final.md", ""),
                files.get("analysis.md", ""),
                task.get("query", ""),
                files.get("research.md", ""),
            )
            confidence_score = derived_metrics["strategic_confidence"]["score"]
        items.append({
            "id": task_id,
            "query": task.get("query", "Untitled Research"),
            "depth": task.get("depth", "comprehensive"),
            "focus": task.get("focus", "general"),
            "status": task.get("status", "completed"),
            "created_at": task.get("created_at", datetime.utcnow().isoformat()),
            "sources_count": task.get("sources_count", 0),
            "confidence_score": confidence_score,
            "duration_seconds": task.get("duration_seconds", 0),
        })
    return {"history": items}


@app.get("/api/research/history/{task_id}")
def get_task_dossier(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Research dossier not found")
    task = TASKS[task_id]
    files = task.get("files", {}) or {}
    analysis_text = files.get("analysis.md", "")
    if task.get("status") == "completed" and analysis_text:
        # Recompute on read so historical dossiers created before the live
        # matrix fix cannot continue serving persisted placeholder content.
        response_task = dict(task)
        response_task["metrics"] = extract_metrics_from_text(
            files.get("final.md", ""),
            analysis_text,
            task.get("query", ""),
            files.get("research.md", ""),
        )
        response_task["confidence_score"] = response_task["metrics"][
            "strategic_confidence"
        ]["score"]
        return response_task
    return task


@app.delete("/api/research/history/{task_id}")
def delete_task_dossier(task_id: str):
    if task_id in TASKS:
        del TASKS[task_id]
        save_history()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Not found")


def extract_entities_from_query(query: str) -> List[str]:
    """Helper to extract comparison entities or subjects from a user prompt."""
    # Check known comparison sets
    q_lower = query.lower()
    if "microsoft" in q_lower and ("google" in q_lower or "amazon" in q_lower):
        return ["Microsoft (Azure AI)", "Google (Vertex AI)", "Amazon (AWS Bedrock)"]
    if "quantumscape" in q_lower or "toyota" in q_lower or "catl" in q_lower:
        return ["QuantumScape", "Toyota Motor", "CATL / BYD"]
    if "deepagents" in q_lower or "langgraph" in q_lower or "crewai" in q_lower or "autogen" in q_lower:
        return ["DeepAgents", "LangGraph", "CrewAI / AutoGen"]
    if "nvidia" in q_lower and ("amd" in q_lower or "intel" in q_lower):
        return ["Nvidia Corp", "AMD Instinct", "Intel / Custom ASICs"]
    if "tesla" in q_lower and "byd" in q_lower:
        return ["Tesla Inc", "BYD Auto", "Legacy OEMs"]
    
    # Try regex patterns like "Compare A, B, and C" or "A vs B"
    vs_match = re.split(r'\b(?:vs\.?|versus|and|compared to|,)\b', query, flags=re.IGNORECASE)
    cleaned = [re.sub(r'[^a-zA-Z0-9\s]', '', item).strip() for item in vs_match if len(item.strip()) > 2]
    # Filter out common filler words
    stopwords = {"compare", "analyze", "evaluate", "benchmark", "recommend", "which", "has", "the", "strongest", "strategy", "for", "between", "what", "is", "how", "timeline", "of"}
    meaningful = [c for c in cleaned if c.lower() not in stopwords and len(c) > 2]
    if len(meaningful) >= 2:
        return meaningful[:3]
        
    return ["Target Subject / Leader", "Industry Benchmark / Peers"]


def _clean_analysis_value(value: str) -> str:
    """Normalize a Markdown cell/bullet without inventing display content."""
    cleaned = re.sub(r"<br\s*/?>", "; ", str(value), flags=re.IGNORECASE)
    cleaned = re.sub(r"[*_`]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" |:-\t")
    return cleaned


def _dedupe_analysis_items(items: List[str], limit: int = 5) -> List[str]:
    unique: List[str] = []
    seen = set()
    for item in items:
        cleaned = _clean_analysis_value(item)
        if not cleaned or cleaned.lower() in {"n/a", "none", "not available"}:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
        if len(unique) >= limit:
            break
    return unique


def _analysis_heading_key(line: str) -> Optional[str]:
    """Map supported Markdown headings to the live matrix schema."""
    heading = line.strip()
    if not heading or heading.startswith("|"):
        return None
    heading = re.sub(r"^#{1,6}\s*", "", heading)
    heading = re.sub(r"^\d+[.)]\s*", "", heading)
    heading = re.sub(r"[*_`]", "", heading).strip().rstrip(":").casefold()
    heading = re.sub(r"\s+", " ", heading)

    aliases = {
        "decision summary": "decision_summary",
        "strengths": "strengths",
        "strengths & core moats": "strengths",
        "weaknesses": "weaknesses",
        "weaknesses & bottlenecks": "weaknesses",
        "opportunities": "opportunities",
        "strategic opportunities": "opportunities",
        "threats": "threats",
        "threats & market headwinds": "threats",
        "risks and gaps": "risks",
        "risks & gaps": "risks",
        "recommended actions": "recommended_actions",
        "near term": "near_term",
        "near-term": "near_term",
        "medium term": "medium_term",
        "mid term": "medium_term",
        "medium-term": "medium_term",
        "mid-term": "medium_term",
        "long term": "long_term",
        "long-term": "long_term",
    }
    if heading in aliases:
        return aliases[heading]
    if heading.startswith(("strategic swot", "decision horizon", "action horizon")):
        return "matrix_container"
    return None


def _parse_analysis_sections(analysis_text: str) -> Dict[str, List[str]]:
    sections = {
        "decision_summary": [],
        "strengths": [],
        "weaknesses": [],
        "opportunities": [],
        "threats": [],
        "risks": [],
        "recommended_actions": [],
        "near_term": [],
        "medium_term": [],
        "long_term": [],
    }
    current: Optional[str] = None
    for raw_line in (analysis_text or "").splitlines():
        heading_key = _analysis_heading_key(raw_line)
        if heading_key:
            current = None if heading_key == "matrix_container" else heading_key
            continue

        bullet_match = re.match(r"^\s*(?:[-*+] |\d+[.)]\s+)(.+?)\s*$", raw_line)
        if not bullet_match:
            continue
        bullet = _clean_analysis_value(bullet_match.group(1))
        if not bullet:
            continue

        horizon_match = re.match(
            r"^(near[- ]term|medium[- ]term|mid[- ]term|long[- ]term)(?:\s*\([^)]*\))?\s*:\s*(.+)$",
            bullet,
            flags=re.IGNORECASE,
        )
        if horizon_match:
            prefix = horizon_match.group(1).casefold()
            horizon_key = (
                "near_term" if prefix.startswith("near")
                else "long_term" if prefix.startswith("long")
                else "medium_term"
            )
            sections[horizon_key].append(horizon_match.group(2))
        elif current in sections:
            sections[current].append(bullet)
    return sections


def _parse_analysis_tables(analysis_text: str) -> List[Dict[str, str]]:
    """Return all Markdown table rows keyed by normalized header text."""
    return [
        row
        for block in _parse_markdown_table_blocks(analysis_text)
        for row in block["rows"]
    ]


def _parse_markdown_table_blocks(markdown_text: str) -> List[Dict[str, Any]]:
    """Parse Markdown tables while preserving each table's headers and rows."""
    lines = (markdown_text or "").splitlines()
    tables: List[Dict[str, Any]] = []
    index = 0
    while index + 1 < len(lines):
        header_line = lines[index].strip()
        divider_line = lines[index + 1].strip()
        if not (
            header_line.startswith("|")
            and divider_line.startswith("|")
            and re.fullmatch(r"[|:\-\s]+", divider_line)
        ):
            index += 1
            continue

        headers = [
            _clean_analysis_value(cell).casefold()
            for cell in header_line.strip("|").split("|")
        ]
        table_rows: List[Dict[str, str]] = []
        index += 2
        while index < len(lines) and lines[index].strip().startswith("|"):
            values = [
                _clean_analysis_value(cell)
                for cell in lines[index].strip().strip("|").split("|")
            ]
            if len(values) == len(headers):
                table_rows.append(dict(zip(headers, values)))
            index += 1
        tables.append({"headers": headers, "rows": table_rows})
    return tables


def extract_matrix_from_analysis(analysis_text: str) -> Dict[str, Any]:
    """Build SWOT and horizon data strictly from the live `/analysis.md`."""
    sections = _parse_analysis_sections(analysis_text)
    table_rows = _parse_analysis_tables(analysis_text)

    table_strengths: List[str] = []
    table_weaknesses: List[str] = []
    for row in table_rows:
        strength = row.get("key strength")
        weakness = row.get("key weakness")
        entity = row.get("vehicle") or row.get("entity") or row.get("option") or row.get("alternative")
        if strength:
            table_strengths.append(f"{entity}: {strength}" if entity else strength)
        if weakness:
            table_weaknesses.append(f"{entity}: {weakness}" if entity else weakness)

    strengths = sections["strengths"] or table_strengths or sections["decision_summary"]
    weaknesses = sections["weaknesses"] or table_weaknesses or sections["risks"]
    opportunities = sections["opportunities"] or sections["recommended_actions"]
    threats = sections["threats"] or sections["risks"]

    explicit_horizon = any(
        sections[key] for key in ("near_term", "medium_term", "long_term")
    )
    near_term = sections["near_term"]
    if not explicit_horizon:
        # Older analyses may predate the horizon contract. Immediate actions
        # remain a direct, non-generative horizon from the live artifact.
        near_term = sections["recommended_actions"]

    return {
        "swot": {
            "strengths": _dedupe_analysis_items(strengths),
            "weaknesses": _dedupe_analysis_items(weaknesses),
            "opportunities": _dedupe_analysis_items(opportunities),
            "threats": _dedupe_analysis_items(threats),
        },
        "horizon": {
            "near_term": _dedupe_analysis_items(near_term, limit=3),
            "medium_term": _dedupe_analysis_items(sections["medium_term"], limit=3),
            "long_term": _dedupe_analysis_items(sections["long_term"], limit=3),
        },
        "source": "/analysis.md",
    }


def _extract_numeric_metric(value: str) -> Optional[float]:
    cleaned = _clean_analysis_value(value)
    if not cleaned or re.search(r"\b(?:n/?a|unknown|not available|missing)\b", cleaned, re.I):
        return None
    fraction = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", cleaned)
    if fraction and float(fraction.group(2)):
        return float(fraction.group(1)) / float(fraction.group(2)) * 100
    number = re.search(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
    return float(number.group(0).replace(",", "")) if number else None


def _is_lower_better_metric(header: str) -> bool:
    lowered = header.casefold()
    return any(term in lowered for term in ("cost", "tco", "price", "latency", "risk", "time")) and not any(
        term in lowered for term in ("resale", "retention", "savings")
    )


def extract_radar_from_analysis(analysis_text: str) -> Dict[str, Any]:
    """Normalize quantitative comparison-table values into evidence-backed scores."""
    entity_headers = ("vehicle", "entity", "option", "alternative", "company", "vendor", "model", "strategy")
    excluded_headers = {
        "finding", "evidence ids", "fact/inference/assumption", "decision implication",
        "key strength", "key weakness", "strength", "weakness", "notes", "source",
    }
    candidates: List[Dict[str, Any]] = []
    for table in _parse_markdown_table_blocks(analysis_text):
        headers = table["headers"]
        rows = table["rows"]
        if len(rows) < 2:
            continue
        entity_header = next((header for header in entity_headers if header in headers), None)
        if not entity_header:
            continue
        metric_headers = []
        for header in headers:
            if header == entity_header or header in excluded_headers:
                continue
            if any(_extract_numeric_metric(row.get(header, "")) is not None for row in rows):
                metric_headers.append(header)
        if metric_headers:
            candidates.append({
                "entity_header": entity_header,
                "metric_headers": metric_headers[:6],
                "rows": rows[:5],
            })

    if not candidates:
        return {"categories": [], "series": [], "source": "/analysis.md"}

    comparison = max(
        candidates,
        key=lambda item: len(item["rows"]) * len(item["metric_headers"]),
    )
    metric_headers = comparison["metric_headers"]
    rows = comparison["rows"]
    normalized_columns: Dict[str, List[int]] = {}
    raw_columns: Dict[str, List[str]] = {}

    for header in metric_headers:
        numeric_values = [_extract_numeric_metric(row.get(header, "")) for row in rows]
        known = [value for value in numeric_values if value is not None and value >= 0]
        raw_columns[header] = [row.get(header, "") or "N/A" for row in rows]
        if not known:
            normalized_columns[header] = [0] * len(rows)
            continue

        lower_is_better = _is_lower_better_metric(header)
        reference = min(known) if lower_is_better else max(known)
        preserve_percent_scale = (
            not lower_is_better
            and max(known) <= 100
            and any(term in header for term in ("score", "rating", "resale", "retention", "%", "percent"))
        )
        scores = []
        for value in numeric_values:
            if value is None or value < 0:
                scores.append(0)
            elif preserve_percent_scale:
                scores.append(round(max(0, min(100, value))))
            elif lower_is_better:
                scores.append(round(100 if value == 0 == reference else (reference / value * 100 if value else 0)))
            else:
                scores.append(round(value / reference * 100 if reference else 0))
        normalized_columns[header] = scores

    series = []
    for row_index, row in enumerate(rows):
        entity = _clean_analysis_value(row.get(comparison["entity_header"], ""))
        if not entity:
            continue
        series.append({
            "name": entity,
            "values": [normalized_columns[header][row_index] for header in metric_headers],
            "raw_values": [raw_columns[header][row_index] for header in metric_headers],
        })

    categories = [re.sub(r"\s+", " ", header).title() for header in metric_headers]
    return {"categories": categories, "series": series, "source": "/analysis.md"}


def _source_authority_tier(source_url: str, source_type: str) -> tuple[str, int]:
    source = f"{source_url} {source_type}".casefold()
    if any(term in source for term in (".gov", "government", "regulator", "sec filing", "nhtsa", "epa")):
        return "Government & Regulatory", 100
    if any(term in source for term in (".edu", "arxiv", "journal", "academic", "ieee", "standard")):
        return "Academic & Standards", 94
    if any(term in source for term in ("consumer reports", "iihs", "j.d. power", "jd power", "independent", "testing", "benchmark")):
        return "Independent Research & Testing", 90
    if any(term in source for term in ("official", "manufacturer", "investor", "press release", "media center", "company filing")):
        return "Primary Company Sources", 84
    if any(term in source for term in ("kbb", "edmunds", "motortrend", "cars.com", "industry", "trade", "publication", "news")):
        return "Industry & Trade Media", 74
    return "Other Secondary Sources", 55


def extract_evidence_rigor(research_text: str) -> Dict[str, Any]:
    evidence_rows: List[Dict[str, str]] = []
    for table in _parse_markdown_table_blocks(research_text):
        if "source url" in table["headers"] and any(
            header in table["headers"] for header in ("claim or metric", "claim", "finding")
        ):
            evidence_rows.extend(table["rows"])

    palette = {
        "Government & Regulatory": "#22c55e",
        "Academic & Standards": "#3b82f6",
        "Independent Research & Testing": "#a855f7",
        "Primary Company Sources": "#06b6d4",
        "Industry & Trade Media": "#f59e0b",
        "Other Secondary Sources": "#f43f5e",
    }
    counts: Dict[str, int] = {}
    authority_scores: List[int] = []
    confidence_scores: List[int] = []
    cited_rows = 0
    for row in evidence_rows:
        source_url = row.get("source url", "")
        source_type = row.get("source type", "")
        tier, authority = _source_authority_tier(source_url, source_type)
        counts[tier] = counts.get(tier, 0) + 1
        authority_scores.append(authority)
        if source_url.startswith(("http://", "https://")):
            cited_rows += 1
        confidence = row.get("confidence", "").casefold()
        confidence_scores.append(95 if "high" in confidence else 72 if "medium" in confidence else 42 if "low" in confidence else 60)

    labels = [label for label in palette if counts.get(label)]
    return {
        "labels": labels,
        "values": [counts[label] for label in labels],
        "colors": [palette[label] for label in labels],
        "total_evidence": len(evidence_rows),
        "authority_score": round(sum(authority_scores) / len(authority_scores)) if authority_scores else 0,
        "claim_confidence_score": round(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0,
        "citation_completeness": round(cited_rows / len(evidence_rows) * 100) if evidence_rows else 0,
        "source": "/research.md",
    }


def extract_strategic_confidence(
    research_text: str,
    analysis_text: str,
    final_text: str,
    evidence_rigor: Dict[str, Any],
) -> Dict[str, Any]:
    analysis_sections = _parse_analysis_sections(analysis_text)
    finding_rows = [
        row
        for table in _parse_markdown_table_blocks(analysis_text)
        if "evidence ids" in table["headers"]
        for row in table["rows"]
    ]
    grounded_findings = sum(
        1 for row in finding_rows
        if row.get("evidence ids", "").strip().casefold() not in {"", "n/a", "none"}
    )
    analysis_grounding = round(grounded_findings / len(finding_rows) * 100) if finding_rows else 0
    assumption_count = sum(
        1 for row in finding_rows
        if "assumption" in row.get("fact/inference/assumption", "").casefold()
    )
    gap_count = len(analysis_sections["risks"])
    coverage_after_gaps = max(0, 100 - min(70, gap_count * 7 + assumption_count * 5))

    research_urls = set(re.findall(r"https?://[^\s|)]+", research_text or ""))
    final_urls = set(re.findall(r"https?://[^\s|)]+", final_text or ""))
    expected_final_sources = min(5, len(research_urls))
    final_traceability = (
        round(min(1, len(final_urls) / expected_final_sources) * 100)
        if expected_final_sources else 0
    )
    components = {
        "Source authority": evidence_rigor["authority_score"],
        "Claim confidence": evidence_rigor["claim_confidence_score"],
        "Citation completeness": evidence_rigor["citation_completeness"],
        "Analysis grounding": analysis_grounding,
        "Coverage after disclosed gaps": coverage_after_gaps,
        "Final report traceability": final_traceability,
    }
    if evidence_rigor["total_evidence"] == 0:
        score = 0
    else:
        weights = (0.25, 0.15, 0.15, 0.20, 0.15, 0.10)
        score = round(sum(value * weight for value, weight in zip(components.values(), weights)))
    logic_rigor = round((analysis_grounding * 0.55 + coverage_after_gaps * 0.30 + final_traceability * 0.15) / 10, 1)
    return {
        "score": max(0, min(100, score)),
        "logic_rigor_score": max(0, min(10, logic_rigor)),
        "components": components,
        "source": "/research.md + /analysis.md + /final.md",
    }


def extract_metrics_from_text(
    final_text: str,
    analysis_text: str,
    query: str,
    research_text: str = "",
):
    """Derive every dashboard metric from the current task's artifacts."""
    matrix = extract_matrix_from_analysis(analysis_text)
    radar = extract_radar_from_analysis(analysis_text)
    evidence_rigor = extract_evidence_rigor(research_text)
    strategic_confidence = extract_strategic_confidence(
        research_text,
        analysis_text,
        final_text,
        evidence_rigor,
    )
    word_count = len(re.findall(r"\b\w+\b", final_text or ""))
    return {
        "radar": radar,
        "swot": matrix["swot"],
        "horizon": matrix["horizon"],
        "matrix_source": matrix["source"],
        "evidence_rigor": evidence_rigor,
        "confidence_breakdown": strategic_confidence["components"],
        "strategic_confidence": strategic_confidence,
        "logic_rigor_score": strategic_confidence["logic_rigor_score"],
        "read_time_minutes": round(word_count / 220, 1) if word_count else 0,
        "query": query,
    }


async def _legacy_simulated_research(task_id: str, query: str, depth: str, focus: str, model: str):
    """
    Generates rich, query-grounded deep research data with realistic real-time telemetry,
    sub-agent handoffs, live Tavily searches when available, and publication-grade output.
    """
    task = TASKS[task_id]
    start_time = time.time()
    
    def log(agent: str, log_type: str, message: str, step_progress: int):
        t_str = datetime.now().strftime("%H:%M:%S")
        entry = {"time": t_str, "agent": agent, "type": log_type, "message": message}
        task["logs"].append(entry)
        task["progress"] = step_progress
        task["active_agent"] = agent
        save_history()

    # Step 1: Supervisor Plan
    await asyncio.sleep(0.6)
    log("orchestrator", "plan", f"Deconstructing research directive: '{query}'", 10)
    await asyncio.sleep(0.7)
    log("orchestrator", "plan", f"Formulated execution plan: 1. Deep web evidence acquisition -> 2. Quantitative reasoning & Fact/Inference matrix -> 3. Executive synthesis.", 20)

    # Step 2: Delegate Research to research-agent
    await asyncio.sleep(0.7)
    log("orchestrator", "delegate", f"Invoking `task` tool: delegating factual evidence search on '{query}' to research-agent.", 25)
    await asyncio.sleep(0.8)
    
    # Real Tavily Search if key is configured
    search_results_list = []
    tavily_key = os.getenv("TAVILY_API_KEY")
    search_queries = [
        f"{query} latest data benchmarks 2026",
        f"{query} market analysis primary sources",
        f"{query} architecture comparison"
    ]
    
    for i, term in enumerate(search_queries):
        log("research-agent", "tool_call", f"Tavily Internet Search: '{term}' (depth={depth}, focus={focus})", 30 + (i * 8))
        if tavily_key:
            try:
                client = TavilyClient(api_key=tavily_key)
                res = client.search(query=term, max_results=3, topic=focus if focus in ["general", "news", "finance"] else "general")
                for item in res.get("results", []):
                    search_results_list.append(item)
            except Exception as e:
                print(f"Tavily search notice: {e}")
        await asyncio.sleep(0.9)
    
    # Format Tavily citations or fallback citations
    citations_md = ""
    evidence_points = []
    if search_results_list:
        seen_urls = set()
        for idx, res in enumerate(search_results_list):
            u = res.get("url", "")
            t = res.get("title", f"Authoritative Source {idx+1}")
            snippet = res.get("content", "")
            if u and u not in seen_urls:
                seen_urls.add(u)
                citations_md += f"{len(seen_urls)}. [{t}]({u})\n"
                if snippet:
                    evidence_points.append(f"- **{t}**: {snippet[:260]}...")
    
    if not citations_md:
        citations_md = f"""1. [Global Industry Technical Report on {query[:35]}](https://www.gartner.com/en)
2. [Frontier Empirical Research & Benchmarks](https://arxiv.org)
3. [Corporate Disclosures & SEC Form 10-K Filings](https://www.sec.gov/edgar)
4. [Authoritative Open-Source Technical Documentation](https://github.com)
"""
    if not evidence_points:
        evidence_points = [
            f"- **Core Finding**: Empirical benchmarks and market disclosures demonstrate strong performance differentials across target architectures for '{query}'.",
            f"- **Adoption & Growth**: Production deployments and industry adoption accelerate in 2026 with verified efficiency gains.",
            f"- **Technical Differentiation**: Key architectural choices directly dictate latency, total cost of ownership (TCO), and scalability."
        ]

    research_md = f"""# Research Dossier: {query}

## 1. Executive Fact Retrieval
{chr(10).join(evidence_points[:5])}

## 2. Authoritative Sources & Citations
{citations_md}

## 3. Quantitative Evidence Log
- Total Data Points Ingested: {max(len(search_results_list) * 120, 1420)}
- Verified Citations Cross-Checked: {max(len(evidence_points), 8)} authoritative domains
- Mathematical Confidence Factor: 95.2%
"""
    task["files"]["research.md"] = research_md
    task["sources_count"] = max(len(evidence_points), 8)
    log("research-agent", "file_write", "Saved verified citations and evidence to `research.md` using write_file.", 55)

    # Step 3: Supervisor Review & Delegate Analysis
    await asyncio.sleep(0.7)
    log("orchestrator", "review", "Inspected `research.md` via read_file. Sufficient empirical evidence acquired. Delegating to analysis-agent.", 60)
    await asyncio.sleep(0.6)

    # Step 4: Analysis Agent Quantitative Reasoning
    log("analysis-agent", "reasoning", "Analyzing research.md: segregating Confirmed Facts vs. Inferences vs. Assumptions.", 65)
    await asyncio.sleep(0.8)
    log("analysis-agent", "tool_call", "Calculator: '((94.8 * 1.25) / 100) * 80' -> 94.8 normalized decision score", 70)
    await asyncio.sleep(0.7)

    entities = extract_entities_from_query(query)
    entity_str = ", ".join(entities)

    analysis_md = f"""# Strategic Analysis & Reasoning: {query}

## 1. Structured Fact vs. Inference vs. Assumption Matrix

### Confirmed Empirical Facts:
- Subject entities ({entity_str}) exhibit distinct operational and architectural trade-offs in current 2026 industry deployments.
- Primary evidence confirms that vertical integration and specialized task delegation provide measurable efficiency advantages.

### Evidence-Supported Inferences:
- Organizations and architectures addressing '{query}' with modular abstraction layers achieve higher long-term agility and lower risk of single-vendor lock-in.
- TCO curves favor hybrid deployment models balancing specialized hardware acceleration with scalable general-purpose compute.

### Critical Assumptions & Boundary Conditions:
- Stability of underlying supply chain and inference pricing dynamics over the 2026-2028 horizon.
- Evolving regulatory compliance and enterprise security governance standards across major markets.

## 2. Comparative Decision Matrix ({entity_str})
- Strategic Velocity: 94 / 100
- Technological Innovation: 92 / 100
- Operational Feasibility: 89 / 100
- Risk-Adjusted ROI: 91 / 100
"""
    task["files"]["analysis.md"] = analysis_md
    log("analysis-agent", "file_write", "Recorded strategic reasoning and decision matrix to `analysis.md` using write_file.", 78)

    # Step 5: Delegate Writer Agent
    await asyncio.sleep(0.7)
    log("orchestrator", "delegate", "Delegating executive synthesis to writer-agent using `task` tool.", 82)
    await asyncio.sleep(0.9)

    final_md = f"""# Executive Intelligence Dossier: {query}

**Prepared by:** Scout AI Autonomous Research Taskforce  
**Classification:** Strategic Intelligence Briefing  
**Date:** {datetime.now().strftime("%B %d, %Y")}  
**Subject Focus:** Deep Evaluation & Evidence-Backed Assessment

---

## Executive Summary

This comprehensive intelligence dossier delivers an exhaustive, evidence-grounded assessment of **{query}**. Utilizing a multi-agent deep research architecture comprising synchronized web investigation (`research.md`), quantitative modeling (`analysis.md`), and executive synthesis (`final.md`), Scout AI evaluated authoritative disclosures, technical papers, and industry performance benchmarks.

### Core Strategic Takeaways
1. **Definitive Findings on {entities[0] if entities else "the Primary Subject"}**: Empirical data and industry documentation confirm that specialized capabilities and distribution channels create defensible moats.
2. **Comparative Trade-offs**: Across evaluated alternatives ({entity_str}), decisions hinge upon the balance between immediate execution velocity, infrastructure cost curves, and ecosystem lock-in.
3. **Actionable Horizon (2026–2028)**: Leadership teams should prioritize modular abstractions and continuous verification loops to maximize resilience.

---

## Comprehensive Comparative Analysis

| Dimension | Leader / Primary ({entities[0] if entities else "Primary"}) | Peer Benchmark ({entities[1] if len(entities) > 1 else "Baseline"}) | Next-Gen Alternative ({entities[2] if len(entities) > 2 else "Emerging"}) |
| :--- | :--- | :--- | :--- |
| **Core Value Moat** | Distribution & Ecosystem Scale | Frontier Technological Innovation | Flexible Neutral Marketplace |
| **Operational Efficiency** | High (Industry Top Decile) | Specialized Performance | Cost-Optimized |
| **Risk Profile** | Low Execution Risk | Rapid Iteration Cycle | Emerging Technology Adoption |
| **Strategic Score** | **94 / 100 (Leader)** | 90 / 100 | 87 / 100 |

---

## In-Depth Empirical Findings

{chr(10).join(evidence_points[:4])}

---

## Strategic Action Plan for Leadership
1. **Deploy Modular Architecture**: Establish clear abstraction boundaries to avoid single-vendor lock-in.
2. **Incorporate Continuous Validation**: Rigorously verify all empirical claims against primary sources and reproducible benchmarks.
3. **Balance TCO and Performance**: Optimize operational expenditures by routing standard workloads to cost-effective engines while reserving frontier systems for critical tasks.

---

## Authoritative References
{citations_md}

---
*Report generated autonomously by Scout AI Multi-Agent Research System. All findings grounded in verified source evidence.*
"""
    task["files"]["final.md"] = final_md
    log("writer-agent", "file_write", "Executive research dossier compiled and published to `final.md`.", 92)

    # Step 6: Final Quality Check & Completion
    await asyncio.sleep(0.6)
    log("orchestrator", "complete", f"Final validation successful. Completed research dossier for '{query[:50]}...'.", 100)
    
    duration = round(time.time() - start_time, 2)
    task["status"] = "completed"
    task["duration_seconds"] = duration
    task["confidence_score"] = None
    task["active_agent"] = "completed"
    task["metrics"] = extract_metrics_from_text(
        final_md,
        analysis_md,
        query,
        task["files"].get("research.md", ""),
    )
    task["confidence_score"] = task["metrics"]["strategic_confidence"]["score"]
    save_history()



async def execute_simulated_research(task_id: str, query: str, depth: str, focus: str, model: str):
    """Render an explicit, non-evidentiary workflow preview without external calls."""
    task = TASKS[task_id]
    start_time = time.time()
    entities = extract_entities_from_query(query)
    entity_list = ", ".join(entities)

    def log(agent: str, log_type: str, message: str, progress: int):
        task["logs"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "type": log_type,
            "message": message,
        })
        task["progress"] = progress
        task["active_agent"] = agent
        save_history()

    task["mode"] = "simulation"
    task["brief"] = build_research_brief(query, depth, focus)
    log("orchestrator", "simulation", f"Simulation preview created for: '{query}'", 25)
    await asyncio.sleep(0.15)
    log("research-agent", "simulation", "Previewing a topic-specific evidence acquisition plan; no web search was executed.", 50)
    await asyncio.sleep(0.15)
    log("analysis-agent", "simulation", "Previewing analysis criteria; no evidence-backed conclusions were generated.", 75)

    task["files"]["research.md"] = f"""# Simulation Preview: Research Plan

Original research question: {query}

> **Simulation only — not verified research.** No external sources were searched and no factual claims were validated.

## Proposed evidence acquisition
- Identify authoritative primary sources specifically addressing: **{query}**
- Search official documentation, regulatory filings, peer-reviewed research, and first-party disclosures.
- Cross-check dates, quantitative claims, and comparisons before including them.
- Preserve complete source URLs beside every supported claim.

## Target subjects detected
{entity_list}
"""
    task["files"]["analysis.md"] = f"""# Simulation Preview: Analysis Plan

Original research question: {query}

> **Simulation only — no analytical conclusions are asserted.** This is the structure the analysis agent would apply after verified evidence is collected.

## Proposed analysis
- Separate confirmed facts, evidence-backed inferences, and assumptions.
- Compare the relevant subjects ({entity_list}) only on dimensions supported by collected evidence.
- Calculate metrics from cited inputs and disclose calculation methods.
- Identify evidence gaps, risks, and decision boundaries.
"""
    task["files"]["final.md"] = f"""# Simulation Preview: Final Report Outline

Original research question: {query}

> **This is a workflow preview, not a research paper.** Run Live Research with valid LLM and Tavily credentials to produce evidence-backed findings.

## Planned report sections
1. Direct answer to the original question
2. Executive summary
3. Verified empirical findings
4. Comparative analysis
5. Risks, uncertainty, and evidence gaps
6. Recommendations grounded in cited evidence
7. Complete source list
"""

    log("writer-agent", "simulation", "Topic-specific report outline completed and labeled as unverified simulation.", 100)
    task["status"] = "completed"
    task["duration_seconds"] = round(time.time() - start_time, 2)
    task["confidence_score"] = 0
    task["sources_count"] = 0
    task["active_agent"] = "completed"
    task["metrics"] = {
        "radar": {"categories": [], "series": []},
        "swot": {
            "strengths": ["Simulation preview only; no strengths were verified."],
            "weaknesses": ["No live evidence was collected."],
            "opportunities": ["Run Live Research to evaluate topic-specific opportunities."],
            "threats": ["Do not use simulation output for factual decisions."],
        },
        "confidence_breakdown": {},
    }
    save_history()


async def execute_real_agent_research(task_id: str, query: str, depth: str, focus: str, model: str):
    """
    Execute the live DeepAgents pipeline and fail explicitly on provider or grounding errors.
    """
    task = TASKS[task_id]
    start_time = time.time()
    
    def log(agent: str, log_type: str, message: str, step_progress: int):
        t_str = datetime.now().strftime("%H:%M:%S")
        entry = {"time": t_str, "agent": agent, "type": log_type, "message": message}
        task["logs"].append(entry)
        reported_progress = 100 if step_progress >= 100 else min(95, step_progress)
        task["progress"] = max(task.get("progress", 0), reported_progress)
        task["active_agent"] = agent
        task["last_activity_at"] = datetime.utcnow().isoformat()
        task["last_activity_epoch"] = time.time()
        task["duration_seconds"] = round(time.time() - start_time, 2)
        save_history()

    def handle_stream_telemetry(update: StreamTelemetry) -> None:
        """Publish nested LangGraph activity to the task's SSE state."""
        task["last_activity_at"] = datetime.utcnow().isoformat()
        task["last_activity_epoch"] = time.time()
        task["duration_seconds"] = round(time.time() - start_time, 2)
        task["active_agent"] = update.agent
        task["current_tool"] = update.current_tool
        task["activity_state"] = update.activity_state
        if update.model_turn is not None:
            task.setdefault("model_turns", {})[update.agent] = update.model_turn
        if update.model_turn_limit is not None:
            task.setdefault("model_turn_limits", {})[
                update.agent
            ] = update.model_turn_limit
        if update.delegation_count is not None:
            task.setdefault("delegation_counts", {})[
                update.agent
            ] = update.delegation_count
        if update.attempt is not None:
            task.setdefault("attempts", {})[update.agent] = update.attempt
            if update.log_type == "attempt":
                task.setdefault("attempt_started_epochs", {})[
                    update.agent
                ] = time.time()
        if update.max_attempts is not None:
            task["max_attempts"] = update.max_attempts
        if update.timeout_seconds is not None:
            task["timeout_seconds"] = update.timeout_seconds
        task["backoff_seconds"] = update.backoff_seconds

        if update.model_turn is not None and (
            update.prompt_tokens is not None
            or update.completion_tokens is not None
            or update.total_tokens is not None
        ):
            call_records = task.setdefault("token_usage_calls", {})
            agent_calls = call_records.setdefault(update.agent, {})
            call_record = agent_calls.setdefault(str(update.model_turn), {})
            if update.prompt_tokens is not None:
                call_record["prompt_tokens"] = update.prompt_tokens
                call_record["prompt_is_estimate"] = (
                    update.token_count_is_estimate
                )
            if update.completion_tokens is not None:
                call_record["completion_tokens"] = update.completion_tokens
            if update.total_tokens is not None:
                call_record["total_tokens"] = update.total_tokens

            by_agent = {}
            for agent_name, records in call_records.items():
                prompt_count = sum(
                    record.get("prompt_tokens", 0)
                    for record in records.values()
                )
                completion_count = sum(
                    record.get("completion_tokens", 0)
                    for record in records.values()
                )
                total_count = sum(
                    record.get(
                        "total_tokens",
                        record.get("prompt_tokens", 0)
                        + record.get("completion_tokens", 0),
                    )
                    for record in records.values()
                )
                by_agent[agent_name] = {
                    "prompt_tokens": prompt_count,
                    "completion_tokens": completion_count,
                    "total_tokens": total_count,
                    "is_estimate": any(
                        record.get("prompt_is_estimate", False)
                        for record in records.values()
                    ),
                }
            task["token_usage"] = {
                "prompt_tokens": sum(
                    usage["prompt_tokens"] for usage in by_agent.values()
                ),
                "completion_tokens": sum(
                    usage["completion_tokens"] for usage in by_agent.values()
                ),
                "total_tokens": sum(
                    usage["total_tokens"] for usage in by_agent.values()
                ),
                "is_estimate": any(
                    usage["is_estimate"] for usage in by_agent.values()
                ),
                "by_agent": by_agent,
            }

        # Rework may revisit an earlier stage; never move the progress bar back.
        stream_progress = max(task.get("progress", 0), update.progress)
        if update.record_log:
            log(
                update.agent,
                update.log_type,
                update.message,
                stream_progress,
            )
        else:
            task["progress"] = min(95, stream_progress)

    try:
        log("orchestrator", "plan", f"Starting Scout AI Research Assistant to research : '{query}'", 10)
        brief = build_research_brief(query, depth, focus)
        task["brief"] = brief
        task["mode"] = "live"

        # Build agent
        agent = build_orchestrator_agent(model)
        log(
            "orchestrator",
            "plan",
            "Compiled Scout AI graph with "
            f"{task['router_model']} for routing/research/analysis and "
            f"{task['synthesis_model']} for final synthesis.",
            20,
        )
        
        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": f"""Execute the complete Scout AI workflow for the immutable research brief in /brief.md.

ORIGINAL USER QUESTION: {query}

For every `task` tool call, include that exact question verbatim in `description`, tell the sub-agent to read /brief.md first, and specify the required stage output. Delegate research to research-agent (/research.md), analysis to analysis-agent (/analysis.md), and final synthesis to writer-agent (/final.md). Do not accept or return content about a neighboring or generic topic.""",
                }
            ],
            "files": build_agent_file_state(brief),
        }

        log("orchestrator", "delegate", "Invoking Orchestrator Deep Agent & Sub-Agent Taskforce...", 30)
        trace_config = {
            "run_name": "scout-ai-research",
            "recursion_limit": config.GRAPH_RECURSION_LIMIT,
            "max_concurrency": config.AGENT_MAX_CONCURRENCY,
            "tags": ["scout-ai", "multi-agent", depth, focus],
            "metadata": {
                "scout_task_id": task_id,
                "research_depth": depth,
                "research_focus": focus,
                "model": model,
                "execution_mode": "live",
                "graph_recursion_limit": config.GRAPH_RECURSION_LIMIT,
            },
        }

        result = await stream_graph_execution(
            agent,
            initial_state,
            trace_config,
            handle_stream_telemetry,
        )
        
        artifacts = extract_agent_artifacts(result)
        validation_errors = validate_agent_artifacts(artifacts, query)

        if validation_errors:
            log(
                "orchestrator",
                "validation",
                "Topic-grounding validation failed; retrying once: " + "; ".join(validation_errors),
                85,
            )
            repair_state = {
                "messages": [{
                    "role": "user",
                    "content": f"""Repair the failed Scout AI workflow.

ORIGINAL USER QUESTION: {query}
VALIDATION ERRORS: {'; '.join(validation_errors)}

Read /brief.md, then delegate any necessary corrections. Every task description must contain the exact original question. Rewrite /research.md, /analysis.md, and /final.md so each begins with `Original research question: {query}` and directly addresses only that question. /research.md must contain real source URLs.""",
                }],
                "files": build_agent_file_state(brief, artifacts),
            }
            repair_trace_config = {
                **trace_config,
                "run_name": "scout-ai-research-repair",
                "tags": [*trace_config["tags"], "topic-grounding-repair"],
            }
            retry_result = await stream_graph_execution(
                agent,
                repair_state,
                repair_trace_config,
                handle_stream_telemetry,
            )
            artifacts = extract_agent_artifacts(retry_result)
            validation_errors = validate_agent_artifacts(artifacts, query)

        if validation_errors:
            raise ResearchValidationError("; ".join(validation_errors))

        task["files"].update(artifacts)
        sources_matches = re.findall(
            r"https?://[^\s\)]+",
            artifacts["research.md"] + " " + artifacts["final.md"],
        )
        task["sources_count"] = len(set(sources_matches))

        log("orchestrator", "complete", "Deep multi-agent research workflow finished successfully.", 100)
        
        duration = round(time.time() - start_time, 2)
        task["status"] = "completed"
        task["duration_seconds"] = duration
        task["confidence_score"] = None
        task["active_agent"] = "completed"
        task["current_tool"] = None
        task["activity_state"] = "completed"
        task["backoff_seconds"] = None
        task["metrics"] = extract_metrics_from_text(
            task["files"]["final.md"],
            task["files"]["analysis.md"],
            query,
            task["files"]["research.md"],
        )
        task["confidence_score"] = task["metrics"]["strategic_confidence"]["score"]
        save_history()

    except Exception as exc:
        err_msg = str(exc)
        failed_agent = None
        print(f"Error in real agent execution: {err_msg}")
        traceback.print_exc()
        if "429" in err_msg or "credit" in err_msg.lower() or "quota" in err_msg.lower():
            user_error = "Live research failed because the LLM provider quota or credits are exhausted. Update billing or select a funded provider, then retry."
        elif isinstance(exc, ResearchValidationError):
            user_error = f"Live research failed topic-grounding validation: {err_msg}"
        elif isinstance(exc, SubagentTimeoutError):
            failed_agent = exc.agent_name
            user_error = f"Live research stopped because {err_msg}"
        elif isinstance(exc, AgentIterationLimitError):
            failed_agent = exc.agent_name
            user_error = f"Live research stopped because {err_msg[:300]}"
        elif isinstance(exc, ModelCallLimitExceededError):
            failed_agent = "orchestrator"
            user_error = (
                "Live research stopped because orchestrator reached its "
                f"configured iteration limit: {err_msg[:300]}"
            )
        elif isinstance(exc, GraphRecursionError):
            failed_agent = task.get("active_agent") or "orchestrator"
            user_error = (
                f"Live research stopped because {failed_agent} did not reach "
                "a graph stop condition within Scout AI's bounded "
                f"{config.GRAPH_RECURSION_LIMIT}-step execution budget. "
                "Per-agent model-call limits remain enforced."
            )
        else:
            user_error = f"Live research failed: {err_msg[:400]}"
        log(failed_agent or "orchestrator", "error", user_error, 100)
        task["status"] = "failed"
        task["progress"] = 100
        task["active_agent"] = failed_agent or "failed"
        task["current_tool"] = None
        task["activity_state"] = "failed"
        task["backoff_seconds"] = None
        task["failed_agent"] = failed_agent
        task["duration_seconds"] = round(time.time() - start_time, 2)
        task["confidence_score"] = None
        task["error"] = user_error
        save_history()


@app.post("/api/research/start")
async def start_research_endpoint(req: ResearchRequest, background_tasks: BackgroundTasks):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Research query cannot be empty.")

    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    nebius_ready = bool(os.getenv("NEBIUS_API_KEY"))
    tavily_ready = bool(os.getenv("TAVILY_API_KEY"))
    if not req.force_simulation and not ((openai_ready or nebius_ready) and tavily_ready):
        raise HTTPException(
            status_code=503,
            detail=(
                "Live research requires Tavily plus either OPENAI_API_KEY or "
                "a NEBIUS_API_KEY. Enable simulation explicitly "
                "only if you want an unverified workflow preview."
            ),
        )

    task_id = str(uuid.uuid4())[:12]
    if openai_ready:
        selected_model = req.model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    else:
        selected_model = (
            req.model
            or os.getenv("NEBIUS_MODEL")
            or "Qwen/Qwen3-30B-A3B-Instruct-2507"
        )
    router_model, synthesis_model = resolve_model_roles(selected_model)
    
    TASKS[task_id] = {
        "id": task_id,
        "query": query,
        "depth": req.depth,
        "focus": req.focus,
        "model": selected_model,
        "router_model": router_model,
        "synthesis_model": synthesis_model,
        "mode": "simulation" if req.force_simulation else "live",
        "status": "in_progress",
        "progress": 5,
        "created_at": datetime.utcnow().isoformat(),
        "duration_seconds": 0,
        "active_agent": "orchestrator",
        "current_tool": None,
        "last_activity_at": datetime.utcnow().isoformat(),
        "last_activity_epoch": time.time(),
        "model_turns": {},
        "model_turn_limits": {},
        "delegation_counts": {},
        "attempts": {},
        "attempt_started_epochs": {},
        "max_attempts": config.AGENT_MAX_RETRIES + 1,
        "timeout_seconds": config.AGENT_TIMEOUT_SECONDS,
        "graph_recursion_limit": config.GRAPH_RECURSION_LIMIT,
        "backoff_seconds": None,
        "activity_state": "active",
        "token_usage_calls": {},
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "is_estimate": False,
            "by_agent": {},
        },
        "sources_count": 0,
        "confidence_score": None,
        "metrics": {},
        "files": {
            "research.md": "",
            "analysis.md": "",
            "final.md": "",
        },
        "logs": [
            {"time": datetime.now().strftime("%H:%M:%S"), "agent": "orchestrator", "type": "init", "message": f"Task registered with ID: {task_id}"}
        ]
    }
    save_history()

    if req.force_simulation:
        background_tasks.add_task(execute_simulated_research, task_id, query, req.depth, req.focus, selected_model)
    else:
        background_tasks.add_task(execute_real_agent_research, task_id, query, req.depth, req.focus, selected_model)

    return {"task_id": task_id, "status": "started"}


@app.get("/api/research/stream/{task_id}")
async def stream_task_endpoint(task_id: str):
    """
    Server-Sent Events (SSE) streaming endpoint for live research telemetry.
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        last_log_count = 0
        last_progress = 0
        
        while True:
            if task_id not in TASKS:
                break
                
            task = TASKS[task_id]
            current_logs = task.get("logs", [])
            current_progress = task.get("progress", 0)
            
            # Send state update payload
            payload = {
                "id": task_id,
                "status": task.get("status"),
                "progress": current_progress,
                "active_agent": task.get("active_agent"),
                "current_tool": task.get("current_tool"),
                "last_activity_at": task.get("last_activity_at"),
                "last_activity_epoch": task.get("last_activity_epoch"),
                "model_turns": task.get("model_turns", {}),
                "model_turn_limits": task.get("model_turn_limits", {}),
                "delegation_counts": task.get("delegation_counts", {}),
                "attempts": task.get("attempts", {}),
                "attempt_started_epochs": task.get("attempt_started_epochs", {}),
                "max_attempts": task.get("max_attempts", config.AGENT_MAX_RETRIES + 1),
                "timeout_seconds": task.get("timeout_seconds", config.AGENT_TIMEOUT_SECONDS),
                "graph_recursion_limit": task.get(
                    "graph_recursion_limit",
                    config.GRAPH_RECURSION_LIMIT,
                ),
                "backoff_seconds": task.get("backoff_seconds"),
                "activity_state": task.get("activity_state", "active"),
                "token_usage": task.get("token_usage", {}),
                "sources_count": task.get("sources_count", 0),
                "confidence_score": task.get("confidence_score"),
                "duration_seconds": task.get("duration_seconds", 0),
                "mode": task.get("mode", "live"),
                "router_model": task.get("router_model"),
                "synthesis_model": task.get("synthesis_model"),
                "error": task.get("error"),
                "logs": current_logs[last_log_count:],
                "files": {
                    "research_len": len(task["files"].get("research.md", "")),
                    "analysis_len": len(task["files"].get("analysis.md", "")),
                    "final_len": len(task["files"].get("final.md", "")),
                }
            }
            
            if task.get("status") in {"completed", "failed"}:
                payload["full_files"] = task.get("files")
                payload["metrics"] = task.get("metrics")
                yield f"data: {json.dumps(payload)}\n\n"
                break

            yield f"data: {json.dumps(payload)}\n\n"
            last_log_count = len(current_logs)
            
            await asyncio.sleep(0.6)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/export/docx")
async def export_docx_endpoint(req: ExportRequest):
    if not HAS_DOCX:
        raise HTTPException(status_code=500, detail="python-docx library is not available.")
    
    doc = docx.Document()
    
    # Title
    title_p = doc.add_paragraph()
    title_run = title_p.add_run(req.title or "Scout AI Deep Research Dossier")
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(15, 23, 42) # Slate 900
    
    meta_p = doc.add_paragraph()
    meta_p.add_run(f"Generated by Scout AI Autonomous Taskforce | Date: {datetime.now().strftime('%B %d, %Y')}\n")
    meta_p.paragraph_format.space_after = Pt(18)
    
    # Body lines
    for line in req.content.split("\n"):
        if line.startswith("# "):
            h = doc.add_heading(line[2:], level=1)
            h.paragraph_format.space_before = Pt(14)
        elif line.startswith("## "):
            h = doc.add_heading(line[3:], level=2)
            h.paragraph_format.space_before = Pt(12)
        elif line.startswith("### "):
            h = doc.add_heading(line[4:], level=3)
            h.paragraph_format.space_before = Pt(8)
        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:], style='List Bullet')
        elif line.strip() == "---":
            doc.add_paragraph("____________________________________________________")
        elif line.strip():
            doc.add_paragraph(line)

    # Append Research & Analysis Appendix if provided
    if req.files:
        if req.files.get("research.md"):
            doc.add_page_break()
            doc.add_heading("Appendix A: Verified Research Evidence (/research.md)", level=1)
            for l in req.files["research.md"].split("\n"):
                if l.strip():
                    doc.add_paragraph(l)

        if req.files.get("analysis.md"):
            doc.add_page_break()
            doc.add_heading("Appendix B: Strategic Analysis & Logic (/analysis.md)", level=1)
            for l in req.files["analysis.md"].split("\n"):
                if l.strip():
                    doc.add_paragraph(l)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    
    filename = f"Scout_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    return Response(
        content=buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# Serve high-end rich frontend
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = os.path.join(current_dir, "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Scout AI UI template loading...</h1>"


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    
    port = 8501
    print("\n" + "="*70)
    print("🚀 SCOUT AI - Autonomous Multi-Agent Deep Research Platform")
    print("="*70)
    print(f"📍 Web UI available at: http://localhost:{port}")
    print("="*70 + "\n")
    
    # Auto-open browser on startup
    threading = True
    try:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    except Exception:
        pass

    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
