from __future__ import annotations
import logging
from datetime import date, timedelta

from langchain_openai import ChatOpenAI

from schemas import FlightResultList, HotelResultList
from tools import tavily_search

logger = logging.getLogger(__name__)

_BUDGET_LABELS = {
    "budget": "budget cheap",
    "mid_range": "mid-range",
    "luxury": "luxury 5-star",
}

_FLIGHT_PARSE_PROMPT = """\
Extract flight offers from the search results below. For each distinct flight option found, extract:
- airline (carrier name, e.g. "Iberia")
- flight_number (e.g. "IB 6251", empty string if not found)
- origin (departure airport or city code, e.g. "JFK")
- destination (arrival airport or city code, e.g. "MAD")
- duration (e.g. "7h 45m", empty string if not found)
- stops (e.g. "Non-stop" or "1 stop (LIS)", empty string if not found)
- departure_time (e.g. "22:30", empty string if not found)
- price (total price as string including currency symbol, e.g. "$487", empty string if not found)
- url (the source URL for this flight offer)

Return up to 5 results. Use empty string for any field that cannot be determined.

Search results:
{context}"""

_HOTEL_PARSE_PROMPT = """\
Extract hotel offers from the search results below. For each distinct hotel found, extract:
- name (hotel name, e.g. "Hotel Vincci Soho")
- stars (integer star rating 1-5, use null if unknown)
- neighborhood (area or district, e.g. "Gran Vía", empty string if not found)
- amenities (comma-separated amenities string, e.g. "Free WiFi, Breakfast", empty string if not found)
- rating (numeric guest score as string, e.g. "4.3", empty string if not found)
- rating_label (e.g. "Excellent", "Very Good", empty string if not found)
- price_per_night (nightly rate with currency symbol, e.g. "€118", empty string if not found)
- price_total (total stay price with symbol, e.g. "€708", empty string if not found)
- url (the source URL for this hotel offer)

Return up to 5 results. Use empty string for any field that cannot be determined; use null for stars if unknown.

Search results:
{context}"""


def search_flights(origin: str, destination: str, start_date: str) -> list[dict]:
    """Search Tavily for flights and parse with gpt-4o-mini. Returns list of dicts."""
    try:
        month_year = date.fromisoformat(start_date).strftime("%B %Y")
    except (ValueError, TypeError):
        month_year = start_date

    query = (
        f"flights from {origin} to {destination} {month_year} price booking "
        "site:kayak.com OR site:google.com/travel OR site:skyscanner.com"
    )
    result = tavily_search(query, max_results=5)
    if result.tool_status != "ok" or not result.results:
        return []

    context = "\n\n".join(
        f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.content_snippet}"
        for r in result.results
    )
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured = llm.with_structured_output(FlightResultList)
        parsed: FlightResultList = structured.invoke(
            _FLIGHT_PARSE_PROMPT.format(context=context)
        )
        return [f.model_dump() for f in parsed.results]
    except Exception as exc:
        logger.warning("Flight parse failed: %s", exc)
        return []


def search_hotels(
    destination: str, start_date: str, duration_days: int, budget_level: str
) -> list[dict]:
    """Search Tavily for hotels and parse with gpt-4o-mini. Returns list of dicts."""
    try:
        check_in = date.fromisoformat(start_date)
        check_out = check_in + timedelta(days=duration_days)
        check_in_str = check_in.strftime("%b %d")
        check_out_str = check_out.strftime("%b %d, %Y")
    except (ValueError, TypeError):
        check_in_str = start_date
        check_out_str = None

    budget_label = _BUDGET_LABELS.get(budget_level, "mid-range")
    date_range = f"{check_in_str} to {check_out_str}" if check_out_str else check_in_str
    query = (
        f"hotels in {destination} {date_range} {budget_label} "
        "price per night booking site:booking.com OR site:hotels.com OR site:tripadvisor.com"
    )
    result = tavily_search(query, max_results=5)
    if result.tool_status != "ok" or not result.results:
        return []

    context = "\n\n".join(
        f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.content_snippet}"
        for r in result.results
    )
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured = llm.with_structured_output(HotelResultList)
        parsed: HotelResultList = structured.invoke(
            _HOTEL_PARSE_PROMPT.format(context=context)
        )
        return [h.model_dump() for h in parsed.results]
    except Exception as exc:
        logger.warning("Hotel parse failed: %s", exc)
        return []
