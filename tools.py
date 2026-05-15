from __future__ import annotations
import logging
import os
from typing import Optional

from tavily import TavilyClient

from schemas import TavilyResult, TavilySearchOutput

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    topic: Optional[str] = None,
) -> TavilySearchOutput:
    """Call Tavily and return a structured TavilySearchOutput."""
    if not TAVILY_API_KEY:
        return TavilySearchOutput(
            results=[],
            query=query,
            tool_status="error",
            error_message="TAVILY_API_KEY not set",
        )

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        kwargs: dict = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        }
        if topic is not None:
            kwargs["topic"] = topic

        raw = client.search(**kwargs)

        results = [
            TavilyResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content_snippet=item.get("content", "")[:1000],
                source_type=item.get("type", "web"),
            )
            for item in raw.get("results", [])
        ]

        return TavilySearchOutput(results=results, query=query, tool_status="ok")

    except Exception as exc:
        logger.error("Tavily search failed for query=%r: %s", query, exc)
        return TavilySearchOutput(
            results=[],
            query=query,
            tool_status="error",
            error_message=str(exc),
        )


def tripadvisor_search(
    query: str,
    max_results: int = 5,
) -> TavilySearchOutput:
    """Search for rated venues using TripAdvisor-biased queries via Tavily advanced mode."""
    targeted_query = f"{query} tripadvisor"
    return tavily_search(query=targeted_query, search_depth="advanced", max_results=max_results)
