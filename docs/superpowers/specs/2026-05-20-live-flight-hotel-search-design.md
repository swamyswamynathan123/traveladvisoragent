# Live Flight & Hotel Search — Design Spec

**Goal:** Add Tavily-powered live flight and hotel search results inside the Streamlit app, surfaced as two new tabs in the main panel after an itinerary is generated.

**Architecture:** Extend `app.py` only. Two new helper functions search Tavily and parse results via `gpt-4o-mini` structured output. Results render as HTML cards in new `✈ Flights` and `🏨 Hotels` tabs. No changes to the graph, schemas, tools, or prompts layers.

**Tech Stack:** Streamlit, Tavily (existing), LangChain OpenAI gpt-4o-mini (existing), Python `st.markdown` with `unsafe_allow_html=True`.

---

## UI Placement

Two new tabs — `✈ Flights` and `🏨 Hotels` — are added to the existing `st.tabs()` row in `app.py`, alongside the current Itinerary, Map, and Packing tabs. The tabs are only shown when a valid `itinerary_response` exists in session state (same condition that gates the existing tabs).

---

## Trigger Behaviour

- Search fires **automatically** the first time each tab renders after a new itinerary is generated.
- Session state flags `flight_search_done` and `hotel_search_done` (both `bool`, default `False`) gate the search so it does not re-fire on every Streamlit rerender.
- Both flags are reset to `False` whenever a new itinerary is written to `st.session_state` — ensuring a new trip always triggers a fresh search.

---

## Data Sources

**Flights tab**
- Reads `trip_request["origin"]`, `trip_request["destination"]`, `trip_request["start_date"]` from session state.
- If `origin` is `None` or empty, the tab renders a `st.text_input("Flying from")` instead of auto-searching. Once the user enters an origin and presses Enter, the search fires.
- Tavily query template: `"flights from {origin} to {destination} {month} {year} price booking site:kayak.com OR site:google.com/travel OR site:skyscanner.com"`

**Hotels tab**
- Reads `trip_request["destination"]`, `trip_request["start_date"]`, `trip_request["duration_days"]`, `trip_request["budget_level"]` from session state.
- Derives `check_out_date = start_date + duration_days days`.
- Tavily query template: `"hotels in {destination} {check_in} to {check_out} {budget_label} price per night booking site:booking.com OR site:hotels.com OR site:tripadvisor.com"`
- `budget_label` mapping: `budget` → "budget cheap", `mid_range` → "mid-range", `luxury` → "luxury 5-star".

---

## Search → Parse → Display Pipeline

### Step 1 — Tavily search
Call the existing `tavily_search(query)` from `tools.py`. Use `max_results=5`. On error, surface a `st.warning` and stop.

### Step 2 — LLM parse
Pass the raw Tavily results (title + url + snippet, concatenated) to `gpt-4o-mini`. Each search function instantiates its own local `ChatOpenAI(model="gpt-4o-mini", temperature=0)` and calls `llm.with_structured_output(FlightResultList | HotelResultList)` — matching the pattern used by the packing list feature. Returns a list of structured dicts.

**FlightResult fields** (Pydantic `BaseModel`, defined at module level in `app.py`):
```
airline: str        # e.g. "Iberia"
flight_number: str  # e.g. "IB 6251" (empty string if unknown)
origin: str         # e.g. "JFK"
destination: str    # e.g. "MAD"
duration: str       # e.g. "7h 45m"
stops: str          # e.g. "Non-stop" or "1 stop (LIS)"
departure_time: str # e.g. "22:30" (empty string if unknown)
price_usd: str      # e.g. "$487" — keep as string to preserve formatting
url: str            # direct booking link from Tavily result
```

**HotelResult fields:**
```
name: str           # e.g. "Hotel Vincci Soho"
stars: int          # 1–5, 0 if unknown
neighborhood: str   # e.g. "Centro / Gran Vía"
amenities: str      # comma-separated, e.g. "Free WiFi, Breakfast"
rating: str         # e.g. "4.3" (empty string if unknown)
rating_label: str   # e.g. "Excellent" (empty string if unknown)
price_per_night: str # e.g. "€118" — keep as string
price_total: str    # e.g. "€708" — keep as string (empty if uncalculable)
url: str            # direct booking link
```

### Step 3 — Render cards
Results are rendered using `st.markdown(..., unsafe_allow_html=True)`.

**Flight card layout:**
- Left border: blue (`#4c5bd4`) for cheapest, grey for others
- Shows: airline + flight number, route + duration + stops + departure time, price (large green for cheapest, amber for others), "Book →" link button

**Hotel card layout:**
- Left border: blue for highest-rated, grey for others
- Shows: name + stars, neighborhood + amenities, guest rating, price per night + total, "Book →" link button

**Disclaimer** (below all cards):
> ⚠ Prices sourced from web search results — verify on the booking site before purchasing. Powered by Tavily.

### Step 4 — Caching
Parsed results are saved to `st.session_state.flight_results` and `st.session_state.hotel_results` so switching away from and back to a tab does not re-run the search.

---

## New Session State Fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `flight_results` | `list[dict]` | `[]` | Parsed flight cards |
| `hotel_results` | `list[dict]` | `[]` | Parsed hotel cards |
| `flight_search_done` | `bool` | `False` | Gate: prevents re-search on rerender |
| `hotel_search_done` | `bool` | `False` | Gate: prevents re-search on rerender |

All four fields are added to the session state bootstrap block. `flight_search_done` and `hotel_search_done` are reset to `False` in the same block that writes a new `itinerary_response`.

---

## New Functions in `app.py`

```python
def _search_flights(origin: str, destination: str, start_date: str) -> list[dict]: ...
def _search_hotels(destination: str, start_date: str, duration_days: int, budget_level: str) -> list[dict]: ...
def _render_flight_card(f: dict, highlight: bool) -> str: ...  # returns HTML string
def _render_hotel_card(h: dict, highlight: bool) -> str: ...   # returns HTML string
```

---

## Error Handling

- Tavily returns no results → `st.info("No results found. Try refining your destination or dates.")`
- Tavily raises an exception → `st.warning("Search temporarily unavailable.")` + log the error
- LLM parse fails → fall back to displaying raw Tavily snippets as plain text links, no card styling
- Origin missing (flights) → show `st.text_input` prompt instead of auto-searching

---

## Files Changed

| File | Change |
|---|---|
| `app.py` | Add 4 helper functions, 4 session state fields, 2 new tab branches |

**No changes to:** `graph.py`, `schemas.py`, `tools.py`, `prompts.py`.

---

## Assumptions

- `tavily_search()` in `tools.py` accepts a plain string query and returns a `TavilySearchOutput` with up to `max_results` results.
- `gpt-4o-mini` is instantiated locally inside each search function (same pattern as packing list, which uses a local `ChatOpenAI` instance).
- Booking links come from Tavily result URLs — no affiliate links or additional tracking.
- All prices are display-only; no booking transaction happens inside the app.
