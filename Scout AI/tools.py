# tools.py

import time
from typing import Literal, List, Dict, Any

import truststore

# Requests/certifi does not automatically use the Windows certificate store.
# Inject the OS trust store so corporate/intermediate CAs remain verified.
truststore.inject_into_ssl()

from langchain_core.tools import tool
from tavily import TavilyClient
from config import (
    AGENT_MAX_RETRIES,
    AGENT_TIMEOUT_SECONDS,
    SEARCH_MAX_RESULTS_PER_QUERY,
    SEARCH_RESULT_SNIPPET_CHARS,
    TAVILY_API_KEY,
)


@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> List[Dict[str, Any]]:
    """Search the internet for current information."""

    client = TavilyClient(api_key=TAVILY_API_KEY)
    bounded_max_results = max(1, min(max_results, SEARCH_MAX_RESULTS_PER_QUERY))

    last_error: Exception | None = None
    for attempt in range(AGENT_MAX_RETRIES + 1):
        try:
            response = client.search(
                query=query,
                max_results=bounded_max_results,
                topic=topic,
                include_raw_content=False,
                timeout=AGENT_TIMEOUT_SECONDS,
                time_range="year"
            )
            compact_results = []
            for result in response.get("results", []):
                compact_results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": str(result.get("content", ""))[
                            :SEARCH_RESULT_SNIPPET_CHARS
                        ],
                        "published_date": result.get("published_date"),
                        "score": result.get("score"),
                    }
                )
            return compact_results
        except Exception as exc:
            last_error = exc
            if attempt >= AGENT_MAX_RETRIES:
                raise
            # Do not immediately hit Tavily again after a transient failure.
            # The two retries wait two seconds and four seconds respectively.
            time.sleep(2 ** (attempt + 1))

    raise RuntimeError("Tavily search failed after retries") from last_error


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression (e.g., '120 * 1.15', '45000 / 12', '((50 - 35) / 35) * 100')."""
    allowed = set("0123456789+-*/(). %")
    if not set(expression).issubset(allowed):
        return "Invalid mathematical expression. Only standard arithmetic characters are permitted."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as exc:
        return f"Calculation failed: {exc}"




