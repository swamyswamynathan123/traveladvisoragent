# Live Flight & Hotel Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auto-triggering ✈️ Flights and 🏨 Hotels tabs to the Streamlit main panel, powered by Tavily search + gpt-4o-mini structured output, showing booking cards after an itinerary is generated.

**Architecture:** New `search.py` module holds the testable search + parse logic. Four Pydantic models added to `schemas.py`. Card renderers and tab UI added to `app.py`. The existing itinerary+packing display is wrapped in a `st.tabs()` call and two new tabs are added alongside it.

**Tech Stack:** Streamlit `st.tabs()`, Tavily (existing `tools.py`), LangChain OpenAI gpt-4o-mini, Pydantic BaseModel.

---

## File Map

| File | Change |
|---|---|
| `schemas.py` | Add `FlightResult`, `FlightResultList`, `HotelResult`, `HotelResultList` after `PackingListResponse` |
| `search.py` | **New file** — `search_flights()`, `search_hotels()`, parse prompts |
| `app.py` | Import from `search.py`; add `_render_flight_card()`, `_render_hotel_card()`; add 4 session state fields; add reset logic in 2 places; wrap main display in `st.tabs()` |
| `tests/test_schemas.py` | Add 4 model validation tests |
| `tests/test_search.py` | **New file** — 7 search function tests + 3 renderer tests |

---

## Task 1: Pydantic models for flight and hotel results

**Files:**
- Modify: `schemas.py` — add 4 models after `PackingListResponse` (after line 165)
- Modify: `tests/test_schemas.py` — add 4 model validation tests

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_schemas.py`:

```python
def test_flight_result_defaults():
    from schemas import FlightResult
    f = FlightResult(airline="Iberia", origin="JFK", destination="MAD", price_usd="$487", url="https://kayak.com")
    assert f.flight_number == ""
    assert f.stops == ""
    assert f.duration == ""
    assert f.departure_time == ""


def test_flight_result_list_wraps_results():
    from schemas import FlightResult, FlightResultList
    frl = FlightResultList(results=[
        FlightResult(airline="Delta", origin="JFK", destination="MAD", price_usd="$512", url="https://delta.com")
    ])
    assert len(frl.results) == 1
    assert frl.results[0].airline == "Delta"


def test_hotel_result_defaults():
    from schemas import HotelResult
    h = HotelResult(name="Hotel Madrid", price_per_night="€118", url="https://booking.com")
    assert h.stars == 0
    assert h.neighborhood == ""
    assert h.amenities == ""
    assert h.rating == ""
    assert h.rating_label == ""
    assert h.price_total == ""


def test_hotel_result_list_wraps_results():
    from schemas import HotelResult, HotelResultList
    hrl = HotelResultList(results=[
        HotelResult(name="Ibis Madrid", price_per_night="€89", url="https://hotels.com")
    ])
    assert len(hrl.results) == 1
    assert hrl.results[0].name == "Ibis Madrid"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_schemas.py::test_flight_result_defaults tests/test_schemas.py::test_flight_result_list_wraps_results tests/test_schemas.py::test_hotel_result_defaults tests/test_schemas.py::test_hotel_result_list_wraps_results -v
```

Expected: FAIL with `ImportError: cannot import name 'FlightResult' from 'schemas'`

- [ ] **Step 3: Add 4 models to schemas.py**

In `schemas.py`, add after the `PackingListResponse` class (after line 165):

```python
class FlightResult(BaseModel):
    airline: str
    flight_number: str = ""
    origin: str
    destination: str
    duration: str = ""
    stops: str = ""
    departure_time: str = ""
    price_usd: str
    url: str


class FlightResultList(BaseModel):
    results: List[FlightResult] = Field(default_factory=list)


class HotelResult(BaseModel):
    name: str
    stars: int = 0
    neighborhood: str = ""
    amenities: str = ""
    rating: str = ""
    rating_label: str = ""
    price_per_night: str
    price_total: str = ""
    url: str


class HotelResultList(BaseModel):
    results: List[HotelResult] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_schemas.py -v
```

Expected: All pass (including the 4 new tests)

- [ ] **Step 5: Commit**

```bash
git add schemas.py tests/test_schemas.py
git commit -m "feat: add FlightResult and HotelResult Pydantic models"
```

---

## Task 2: Search + parse functions in search.py

**Files:**
- Create: `search.py`
- Create: `tests/test_search.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_search.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from schemas import (
    FlightResult, FlightResultList,
    HotelResult, HotelResultList,
    TavilySearchOutput, TavilyResult,
)

FAKE_KEY = "tvly-fake-key"


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
                     price_usd="$487", url="https://kayak.com/f1"),
    ])
    with patch("search.tavily_search", return_value=fake_tavily), \
         patch("search.ChatOpenAI", return_value=_mock_llm(fake_parsed)):
        results = search_flights("New York", "Madrid", "2026-06-01")
    assert len(results) == 1
    assert results[0]["airline"] == "Iberia"
    assert results[0]["price_usd"] == "$487"
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_search.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'search'`

- [ ] **Step 3: Create search.py**

Create `search.py` at the project root:

```python
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
- price_usd (total price as string including currency symbol, e.g. "$487", empty string if not found)
- url (the source URL for this flight offer)

Return up to 5 results. Use empty string for any field that cannot be determined.

Search results:
{context}"""

_HOTEL_PARSE_PROMPT = """\
Extract hotel offers from the search results below. For each distinct hotel found, extract:
- name (hotel name, e.g. "Hotel Vincci Soho")
- stars (integer star rating 1-5, use 0 if unknown)
- neighborhood (area or district, e.g. "Gran Vía", empty string if not found)
- amenities (comma-separated list, e.g. "Free WiFi, Breakfast", empty string if not found)
- rating (numeric guest score as string, e.g. "4.3", empty string if not found)
- rating_label (e.g. "Excellent", "Very Good", empty string if not found)
- price_per_night (nightly rate with currency symbol, e.g. "€118", empty string if not found)
- price_total (total stay price with symbol, e.g. "€708", empty string if not found)
- url (the source URL for this hotel offer)

Return up to 5 results. Use empty string for any field that cannot be determined; use 0 for stars.

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
        check_out_str = ""

    budget_label = _BUDGET_LABELS.get(budget_level, "mid-range")
    query = (
        f"hotels in {destination} {check_in_str} to {check_out_str} {budget_label} "
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_search.py -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Run full suite to check for regressions**

```
pytest tests/ -v
```

Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add search.py tests/test_search.py
git commit -m "feat: add flight and hotel Tavily search + LLM parse module"
```

---

## Task 3: Card renderer functions in app.py

**Files:**
- Modify: `app.py` — add `_render_flight_card()` and `_render_hotel_card()` after `_generate_packing_list` (before `load_dotenv()` on line 321), and add `from search import search_flights, search_hotels` to the imports block
- Modify: `tests/test_search.py` — add 3 renderer tests

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/test_search.py`:

```python
def test_render_flight_card_contains_airline_and_price():
    import sys, types
    sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
    from app import _render_flight_card
    flight = {
        "airline": "Iberia", "flight_number": "IB 6251", "origin": "JFK",
        "destination": "MAD", "duration": "7h 45m", "stops": "Non-stop",
        "departure_time": "22:30", "price_usd": "$487", "url": "https://kayak.com",
    }
    html = _render_flight_card(flight, highlight=True)
    assert "Iberia" in html
    assert "$487" in html
    assert "https://kayak.com" in html
    assert "#4c5bd4" in html


def test_render_flight_card_grey_border_when_not_highlighted():
    import sys, types
    sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
    from app import _render_flight_card
    flight = {
        "airline": "Delta", "flight_number": "", "origin": "JFK",
        "destination": "MAD", "duration": "", "stops": "",
        "departure_time": "", "price_usd": "$512", "url": "https://delta.com",
    }
    html = _render_flight_card(flight, highlight=False)
    assert "#374151" in html
    assert "#4c5bd4" not in html


def test_render_hotel_card_contains_name_and_price():
    import sys, types
    sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
    from app import _render_hotel_card
    hotel = {
        "name": "Hotel Vincci Soho", "stars": 4, "neighborhood": "Gran Via",
        "amenities": "Free WiFi", "rating": "4.3", "rating_label": "Excellent",
        "price_per_night": "€118", "price_total": "€708", "url": "https://booking.com/vincci",
    }
    html = _render_hotel_card(hotel, highlight=True)
    assert "Hotel Vincci Soho" in html
    assert "€118" in html
    assert "https://booking.com/vincci" in html
    assert "#4c5bd4" in html
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_search.py::test_render_flight_card_contains_airline_and_price tests/test_search.py::test_render_flight_card_grey_border_when_not_highlighted tests/test_search.py::test_render_hotel_card_contains_name_and_price -v
```

Expected: FAIL with `ImportError: cannot import name '_render_flight_card' from 'app'`

- [ ] **Step 3: Add import and renderer functions to app.py**

In `app.py`, add to the imports block (after `from schemas import PackingListResponse` on line 19):

```python
from search import search_flights, search_hotels
```

In `app.py`, add after the `_generate_packing_list` function (before `load_dotenv()` on line 321):

```python
def _render_flight_card(f: dict, highlight: bool) -> str:
    border = "#4c5bd4" if highlight else "#374151"
    price_color = "#4ade80" if highlight else "#f59e0b"
    meta_parts = [p for p in [
        f.get("duration"),
        f.get("stops"),
        f"Departs {f['departure_time']}" if f.get("departure_time") else None,
    ] if p]
    meta = " · ".join(meta_parts) if meta_parts else ""
    airline_line = f.get("airline", "")
    if f.get("flight_number"):
        airline_line += f" · {f['flight_number']}"
    route = f"{f.get('origin', '')} → {f.get('destination', '')}"
    return (
        f'<div style="background:#1e2130;border-radius:6px;padding:10px;display:flex;'
        f'align-items:center;gap:12px;border-left:3px solid {border};margin-bottom:8px">'
        f'<div style="font-size:20px">🛫</div>'
        f'<div style="flex:1">'
        f'<div style="font-weight:bold;color:#e2e8f0;font-size:13px">{airline_line}</div>'
        f'<div style="color:#94a3b8;font-size:11px">{route}{(" · " + meta) if meta else ""}</div>'
        f'</div>'
        f'<div style="text-align:right;margin-right:8px">'
        f'<div style="color:{price_color};font-size:15px;font-weight:bold">{f.get("price_usd", "N/A")}</div>'
        f'<div style="color:#64748b;font-size:10px">per person</div>'
        f'</div>'
        f'<a href="{f.get("url", "#")}" target="_blank" style="background:{border};color:white;'
        f'border-radius:4px;padding:5px 12px;font-size:11px;text-decoration:none;white-space:nowrap">Book →</a>'
        f'</div>'
    )


def _render_hotel_card(h: dict, highlight: bool) -> str:
    border = "#4c5bd4" if highlight else "#374151"
    price_color = "#4ade80" if highlight else "#f59e0b"
    stars_str = "★" * h.get("stars", 0) if h.get("stars") else ""
    name_line = h.get("name", "") + (f" {stars_str}" if stars_str else "")
    sub_parts = [p for p in [h.get("neighborhood"), h.get("amenities")] if p]
    sub = " · ".join(sub_parts) if sub_parts else ""
    rating_str = ""
    if h.get("rating"):
        rating_str = f"★ {h['rating']}"
        if h.get("rating_label"):
            rating_str += f" · {h['rating_label']}"
    total_str = f" (~{h['price_total']} total)" if h.get("price_total") else ""
    rating_div = (
        f'<div style="color:#fbbf24;font-size:11px;margin-top:2px">{rating_str}</div>'
        if rating_str else ""
    )
    return (
        f'<div style="background:#1e2130;border-radius:6px;padding:10px;display:flex;'
        f'gap:12px;border-left:3px solid {border};margin-bottom:8px">'
        f'<div style="font-size:26px;align-self:center">🏨</div>'
        f'<div style="flex:1">'
        f'<div style="font-weight:bold;color:#e2e8f0;font-size:13px">{name_line}</div>'
        f'<div style="color:#94a3b8;font-size:11px">{sub}</div>'
        f'{rating_div}'
        f'</div>'
        f'<div style="text-align:right;align-self:center;margin-right:8px">'
        f'<div style="color:{price_color};font-size:14px;font-weight:bold">'
        f'{h.get("price_per_night", "N/A")}'
        f'<span style="font-size:10px;color:#94a3b8">/night</span></div>'
        f'<div style="color:#64748b;font-size:10px">{total_str}</div>'
        f'<a href="{h.get("url", "#")}" target="_blank" style="background:{border};color:white;'
        f'border-radius:4px;padding:4px 10px;font-size:10px;text-decoration:none;'
        f'display:inline-block;margin-top:4px">Book →</a>'
        f'</div>'
        f'</div>'
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_search.py -v
```

Expected: All 10 tests PASS

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_search.py
git commit -m "feat: add flight and hotel card renderer functions"
```

---

## Task 4: Session state fields, reset logic, and tabs UI

**Files:**
- Modify: `app.py`
  - Session state bootstrap block (around line 348, after `geocode_cache` entry)
  - `generate_clicked` block (around line 856, after `_save_to_history` call)
  - History-load block (around line 803, after `weather_data = {}`)
  - Main display block (lines 904–949 replaced with tabbed version)

- [ ] **Step 1: Add 4 session state fields to the bootstrap block**

In `app.py`, find the session state bootstrap section. After this existing line:

```python
if "geocode_cache" not in st.session_state:
    st.session_state.geocode_cache = {}  # location string → (lat, lon) or None
```

Add:

```python
if "flight_results" not in st.session_state:
    st.session_state.flight_results = []
if "hotel_results" not in st.session_state:
    st.session_state.hotel_results = []
if "flight_search_done" not in st.session_state:
    st.session_state.flight_search_done = False
if "hotel_search_done" not in st.session_state:
    st.session_state.hotel_search_done = False
```

- [ ] **Step 2: Add reset logic in the generate_clicked block**

In `app.py`, find this exact sequence in the `generate_clicked` block:

```python
        result_state = _run_graph(new_state)
        st.session_state.agent_state = result_state
        if result_state.get("itinerary_response"):
            _save_to_history(result_state["itinerary_response"], result_state.get("trip_request"))
        st.rerun()
```

Replace with:

```python
        result_state = _run_graph(new_state)
        st.session_state.agent_state = result_state
        if result_state.get("itinerary_response"):
            _save_to_history(result_state["itinerary_response"], result_state.get("trip_request"))
        st.session_state.flight_results = []
        st.session_state.hotel_results = []
        st.session_state.flight_search_done = False
        st.session_state.hotel_search_done = False
        st.rerun()
```

- [ ] **Step 3: Add reset logic in the history-load block**

In `app.py`, find this exact sequence in the history-load block:

```python
                st.session_state.agent_state["itinerary_response"] = _it
                st.session_state.agent_state["trip_request"] = _req
                st.session_state.packing_list = None
                st.session_state.weather_data = {}
                if _req:
                    st.session_state.pending_prefill = _req
                st.rerun()
```

Replace with:

```python
                st.session_state.agent_state["itinerary_response"] = _it
                st.session_state.agent_state["trip_request"] = _req
                st.session_state.packing_list = None
                st.session_state.weather_data = {}
                st.session_state.flight_results = []
                st.session_state.hotel_results = []
                st.session_state.flight_search_done = False
                st.session_state.hotel_search_done = False
                if _req:
                    st.session_state.pending_prefill = _req
                st.rerun()
```

- [ ] **Step 4: Replace the main itinerary display with tabbed layout**

In `app.py`, find this entire block (lines 904–949):

```python
if itinerary_response:
    st.divider()
    _render_itinerary(itinerary_response.get("data", {}), weather=st.session_state.weather_data or None, weather_label=st.session_state.weather_label)

    # ── action buttons below itinerary ────────────────────────────────────────
    btn_a, btn_b, _spacer = st.columns([1, 1, 3])
    with btn_a:
        if st.button("🎒 Generate Packing List", use_container_width=True):
            with st.spinner("Generating packing list..."):
                result = _generate_packing_list(itinerary_response.get("data", {}))
            if result:
                st.session_state.packing_list = result
                st.rerun()
            else:
                st.error("Could not generate packing list. Please try again.")

    _trip_req = agent_state.get("trip_request") or {}
    _start_date = _trip_req.get("start_date")
    # Fall back to the sidebar date widget for trips loaded from history without a stored start_date
    if not _start_date and "sb_start_date" in st.session_state:
        _sb_date = st.session_state["sb_start_date"]
        if _sb_date:
            _start_date = str(_sb_date)
    with btn_b:
        _weather_label = "🌤️ Refresh Weather" if st.session_state.weather_data else "🌤️ Add Weather Forecast"
        _weather_disabled = not bool(_start_date)
        _weather_help = "Set a Start Date in the sidebar to enable weather forecasts." if _weather_disabled else None
        if st.button(_weather_label, use_container_width=True,
                     disabled=_weather_disabled, help=_weather_help):
            _itin_data = itinerary_response.get("data", {})
            _dest_for_geo = _itin_data.get("destination", "")
            _duration = _trip_req.get("duration_days", 7)
            with st.spinner("Fetching weather (per city)..." if "," in _dest_for_geo or "+" in _dest_for_geo else "Fetching weather forecast..."):
                _weather_map, _wlabel = _build_weather_map(_dest_for_geo, _start_date, int(_duration), _itin_data)
            if _weather_map:
                st.session_state.weather_data = _weather_map
                st.session_state.weather_label = _wlabel
                if _wlabel == "typical":
                    st.info("Showing typical weather from the same dates last year — your trip is more than 16 days away.")
                st.rerun()
            else:
                st.warning("Could not fetch weather data for this destination and date range.")

    if st.session_state.packing_list:
        st.divider()
        _render_packing_list(st.session_state.packing_list)
```

Replace with:

```python
if itinerary_response:
    st.divider()
    _trip_req = agent_state.get("trip_request") or {}
    _start_date = _trip_req.get("start_date")
    if not _start_date and "sb_start_date" in st.session_state:
        _sb_date = st.session_state["sb_start_date"]
        if _sb_date:
            _start_date = str(_sb_date)

    tab_itin, tab_flights, tab_hotels = st.tabs(["🗺️ Itinerary", "✈️ Flights", "🏨 Hotels"])

    with tab_itin:
        _render_itinerary(
            itinerary_response.get("data", {}),
            weather=st.session_state.weather_data or None,
            weather_label=st.session_state.weather_label,
        )
        btn_a, btn_b, _spacer = st.columns([1, 1, 3])
        with btn_a:
            if st.button("🎒 Generate Packing List", use_container_width=True):
                with st.spinner("Generating packing list..."):
                    result = _generate_packing_list(itinerary_response.get("data", {}))
                if result:
                    st.session_state.packing_list = result
                    st.rerun()
                else:
                    st.error("Could not generate packing list. Please try again.")
        with btn_b:
            _weather_label = "🌤️ Refresh Weather" if st.session_state.weather_data else "🌤️ Add Weather Forecast"
            _weather_disabled = not bool(_start_date)
            _weather_help = "Set a Start Date in the sidebar to enable weather forecasts." if _weather_disabled else None
            if st.button(_weather_label, use_container_width=True,
                         disabled=_weather_disabled, help=_weather_help):
                _itin_data = itinerary_response.get("data", {})
                _dest_for_geo = _itin_data.get("destination", "")
                _duration = _trip_req.get("duration_days", 7)
                with st.spinner(
                    "Fetching weather (per city)..."
                    if "," in _dest_for_geo or "+" in _dest_for_geo
                    else "Fetching weather forecast..."
                ):
                    _weather_map, _wlabel = _build_weather_map(
                        _dest_for_geo, _start_date, int(_duration), _itin_data
                    )
                if _weather_map:
                    st.session_state.weather_data = _weather_map
                    st.session_state.weather_label = _wlabel
                    if _wlabel == "typical":
                        st.info("Showing typical weather from the same dates last year — your trip is more than 16 days away.")
                    st.rerun()
                else:
                    st.warning("Could not fetch weather data for this destination and date range.")
        if st.session_state.packing_list:
            st.divider()
            _render_packing_list(st.session_state.packing_list)

    with tab_flights:
        _dest = (_trip_req.get("destination") or "").strip()
        _origin = (_trip_req.get("origin") or "").strip()
        if not _origin:
            _origin = st.text_input(
                "✈️ Flying from (city or airport code)",
                key="flight_origin_input",
                placeholder="e.g. New York",
            )
        if _origin and _dest and not st.session_state.flight_search_done:
            with st.spinner(f"Searching flights from {_origin} to {_dest}..."):
                st.session_state.flight_results = search_flights(_origin, _dest, _start_date or "")
            st.session_state.flight_search_done = True
        if st.session_state.flight_search_done:
            flights = st.session_state.flight_results
            if not flights:
                st.info("No flight results found. Try refining your origin or destination.")
            else:
                st.markdown(
                    "".join(_render_flight_card(f, highlight=(i == 0)) for i, f in enumerate(flights)),
                    unsafe_allow_html=True,
                )
                st.caption("⚠️ Prices sourced from web search — verify on the booking site before purchasing. Powered by Tavily.")
            if st.button("🔄 Refresh Flight Search", key="refresh_flights"):
                st.session_state.flight_search_done = False
                st.session_state.flight_results = []
                st.rerun()

    with tab_hotels:
        _dest = (_trip_req.get("destination") or "").strip()
        _duration = int(_trip_req.get("duration_days") or 5)
        _budget = _trip_req.get("budget_level", "mid_range")
        if _dest and not st.session_state.hotel_search_done:
            with st.spinner(f"Searching hotels in {_dest}..."):
                st.session_state.hotel_results = search_hotels(_dest, _start_date or "", _duration, _budget)
            st.session_state.hotel_search_done = True
        if st.session_state.hotel_search_done:
            hotels = st.session_state.hotel_results
            if not hotels:
                st.info("No hotel results found. Try refining your destination or dates.")
            else:
                st.markdown(
                    "".join(_render_hotel_card(h, highlight=(i == 0)) for i, h in enumerate(hotels)),
                    unsafe_allow_html=True,
                )
                st.caption("⚠️ Prices sourced from web search — verify on the booking site before purchasing. Powered by Tavily.")
            if st.button("🔄 Refresh Hotel Search", key="refresh_hotels"):
                st.session_state.hotel_search_done = False
                st.session_state.hotel_results = []
                st.rerun()
```

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```

Expected: All tests pass

- [ ] **Step 6: Manually verify the UI**

```
streamlit run app.py
```

Check:
1. Generate a trip with an origin (e.g. "New York") and destination (e.g. "Madrid") — three tabs appear: 🗺️ Itinerary, ✈️ Flights, 🏨 Hotels
2. Click ✈️ Flights — spinner fires, then 3–5 flight cards appear with airline, price, Book link
3. Click 🏨 Hotels — spinner fires, then hotel cards appear with name, price, rating, Book link
4. The first card in each tab has a blue left border; remaining cards have grey
5. Click 🔄 Refresh — search re-fires and cards update
6. Generate a second trip — tab click triggers a fresh search (not cached results from the first trip)
7. Load a trip from history — same as above, fresh search on next tab click
8. Generate a trip WITHOUT setting origin — Flights tab shows the "Flying from" text input; entering a city there triggers the search

- [ ] **Step 7: Commit**

```bash
git add app.py
git commit -m "feat: add Flights and Hotels tabs with Tavily-powered auto-search"
```
