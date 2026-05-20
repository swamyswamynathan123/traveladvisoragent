from unittest.mock import patch, MagicMock
from schemas import (
    FlightResult, FlightResultList,
    HotelResult, HotelResultList,
    TavilySearchOutput, TavilyResult,
)


def _tavily_ok(*results):
    return TavilySearchOutput(
        results=[
            TavilyResult(title=r["title"], url=r["url"], content_snippet=r["snippet"], source_type="web")
            for r in results
        ],
        query="test",
        tool_status="ok",
    )


def _tavily_error():
    return TavilySearchOutput(results=[], query="test", tool_status="error", error_message="timeout")


def _mock_llm(return_value):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = return_value
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


def test_search_flights_returns_list_of_dicts():
    from search import search_flights
    fake_tavily = _tavily_ok({"title": "Iberia JFK-MAD $487", "url": "https://kayak.com/f1", "snippet": "Iberia IB6251 JFK to MAD nonstop $487"})
    fake_parsed = FlightResultList(results=[
        FlightResult(airline="Iberia", flight_number="IB 6251", origin="JFK", destination="MAD",
                     duration="7h 45m", stops="Non-stop", departure_time="22:30",
                     price="$487", url="https://kayak.com/f1"),
    ])
    with patch("search.tavily_search", return_value=fake_tavily), \
         patch("search.ChatOpenAI", return_value=_mock_llm(fake_parsed)):
        results = search_flights("New York", "Madrid", "2026-06-01")
    assert len(results) == 1
    assert results[0]["airline"] == "Iberia"
    assert results[0]["price"] == "$487"
    assert results[0]["url"] == "https://kayak.com/f1"


def test_search_flights_returns_empty_on_tavily_error():
    from search import search_flights
    with patch("search.tavily_search", return_value=_tavily_error()):
        results = search_flights("New York", "Madrid", "2026-06-01")
    assert results == []


def test_search_flights_returns_empty_on_llm_failure():
    from search import search_flights
    fake_tavily = _tavily_ok({"title": "Flights", "url": "https://x.com", "snippet": "text"})
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("LLM error")
    mock_llm.with_structured_output.return_value = mock_structured
    with patch("search.tavily_search", return_value=fake_tavily), \
         patch("search.ChatOpenAI", return_value=mock_llm):
        results = search_flights("New York", "Madrid", "2026-06-01")
    assert results == []


def test_search_hotels_returns_list_of_dicts():
    from search import search_hotels
    fake_tavily = _tavily_ok({"title": "Hotel Vincci Soho Madrid", "url": "https://booking.com/vincci", "snippet": "4-star hotel Gran Via €118/night"})
    fake_parsed = HotelResultList(results=[
        HotelResult(name="Hotel Vincci Soho", stars=4, neighborhood="Gran Via", amenities="Free WiFi",
                    rating="4.3", rating_label="Excellent", price_per_night="€118",
                    price_total="€708", url="https://booking.com/vincci"),
    ])
    with patch("search.tavily_search", return_value=fake_tavily), \
         patch("search.ChatOpenAI", return_value=_mock_llm(fake_parsed)):
        results = search_hotels("Madrid", "2026-06-01", 6, "mid_range")
    assert len(results) == 1
    assert results[0]["name"] == "Hotel Vincci Soho"
    assert results[0]["price_per_night"] == "€118"
    assert results[0]["url"] == "https://booking.com/vincci"


def test_search_hotels_returns_empty_on_tavily_error():
    from search import search_hotels
    with patch("search.tavily_search", return_value=_tavily_error()):
        results = search_hotels("Madrid", "2026-06-01", 6, "mid_range")
    assert results == []


def test_search_hotels_returns_empty_on_llm_failure():
    from search import search_hotels
    fake_tavily = _tavily_ok({"title": "Hotels", "url": "https://x.com", "snippet": "text"})
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = RuntimeError("LLM error")
    mock_llm.with_structured_output.return_value = mock_structured
    with patch("search.tavily_search", return_value=fake_tavily), \
         patch("search.ChatOpenAI", return_value=mock_llm):
        results = search_hotels("Madrid", "2026-06-01", 6, "mid_range")
    assert results == []


def test_search_hotels_query_uses_luxury_label():
    from search import search_hotels
    with patch("search.tavily_search", return_value=_tavily_error()) as mock_tav:
        search_hotels("Tokyo", "2026-07-01", 3, "luxury")
    query_used = mock_tav.call_args[0][0]
    assert "luxury" in query_used.lower()


def test_search_hotels_query_uses_budget_label():
    from search import search_hotels
    with patch("search.tavily_search", return_value=_tavily_error()) as mock_tav:
        search_hotels("Tokyo", "2026-07-01", 3, "budget")
    query_used = mock_tav.call_args[0][0]
    assert "budget" in query_used.lower()
