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
                content_snippet=item.get("content", "")[:500],
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


def tavily_search_advanced(
    query: str,
    max_results: int = 5,
) -> TavilySearchOutput:
    """Tavily search with advanced depth — deeper content extraction from page bodies."""
    return tavily_search(query=query, search_depth="advanced", max_results=max_results)


def tripadvisor_search(
    query: str,
    max_results: int = 5,
) -> TavilySearchOutput:
    """Search for rated attractions using TripAdvisor-biased queries via Tavily advanced mode.
    Use for attractions/hotels only — not restaurants (TripAdvisor restaurant pages are JS-rendered)."""
    targeted_query = f"{query} tripadvisor"
    return tavily_search(query=targeted_query, search_depth="advanced", max_results=max_results)


def expedia_hotel_search(
    destination: str,
    check_in: str,
    budget_level: str = "mid_range",
) -> TavilySearchOutput:
    """Search for hotels via targeted booking-site queries.
    Results are tagged source_type='hotel_search' to separate them from activity results."""
    budget_labels = {
        "budget": "budget hostel cheap",
        "mid_range": "hotel",
        "luxury": "luxury 5-star hotel",
    }
    label = budget_labels.get(budget_level, "hotel")
    query = f"best {label} {destination} {check_in} recommended booking"
    result = tavily_search(query=query, search_depth="advanced", max_results=5)

    if result.tool_status != "ok":
        return result

    tagged_results = [
        TavilyResult(
            title=r.title,
            url=r.url,
            content_snippet=r.content_snippet,
            source_type="hotel_search",
        )
        for r in result.results
    ]
    return TavilySearchOutput(results=tagged_results, query=query, tool_status="ok")
