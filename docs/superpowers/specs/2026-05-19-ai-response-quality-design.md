# AI Response Quality — Design Spec
**Date:** 2026-05-19
**Status:** Approved

## Problem

Users experience four quality failures in generated itineraries and travel answers:
1. **Generic output** — activities like "visit a local market" with no specific venue names
2. **Poor personalization** — interests, budget, and pace from the sidebar are ignored or inconsistently applied
3. **Unreliable facts** — hotel/restaurant suggestions invented from training data without verification
4. **Shallow detail** — one-sentence activity notes, no practical logistics (how to get there, when to book, cost estimate)

## Approach

Approach C — Full Quality Pipeline:
- Switch generation LLM from GPT-4o to Claude Sonnet 4.6
- Run all Tavily searches in parallel (ThreadPoolExecutor)
- Integrate Expedia MCP for real hotel listings alongside Tavily
- Rewrite `ITINERARY_GENERATION_PROMPT` with explicit personalization contract, depth rules, and budget weaving
- Fix `collect_requirements_node` and `ask_clarification_node` to use `with_structured_output` instead of `json.loads()`
- Add a new `personalization_check_node` (Claude Haiku) as a post-generation quality gate
- Make follow-up questions clickable buttons in the UI

## Architecture

### Graph Changes

```
collect_requirements  (structured output — no raw json.loads)
  → validate_inputs
  → ask_clarification  (if needed — structured output)
  → search_with_tavily  (parallel via ThreadPoolExecutor)
    + expedia_hotel_search  (runs in same parallel pool when start_date is set)
  → generate_response  (Claude Sonnet 4.6, rewritten prompts)
  → personalization_check  (Claude Haiku — itinerary only)
  → respond_to_user
```

### New `personalization_check` Node

- Runs **only** when `intent == "planning"` (skipped for Q&A)
- Uses `claude-haiku-4-5-20251001` — cheap and fast (~1s)
- Reads the generated `ItineraryResponse` + original `trip_request`
- Checks four things: interest alignment, budget-tier consistency, pace (activities/day), constraint adherence
- Returns the same `ItineraryResponse` schema with violations patched
- On failure (timeout or parse error), passes through the original itinerary unchanged

**Graph wiring:** Replace the direct edge `generate_response → respond_to_user` with a conditional edge via `should_check_personalization()`:
- `intent == "planning"` → `personalization_check` → `respond_to_user`
- any other intent → `respond_to_user` directly

### Structured Output for All LLM Calls

`collect_requirements_node` and `ask_clarification_node` currently do `json.loads(response.content)` — this breaks silently when the model adds preamble text. Both are migrated to `llm.with_structured_output(PydanticModel)`.

New Pydantic model needed: `IntentDetectionOutput` with fields `intent`, `trip_request`, `travel_question`, `clarification_needed`.

## Prompts

### `ITINERARY_GENERATION_PROMPT` — Key Changes

**Personalization contract block (new, prepended):**
```
Before writing anything, confirm you will honor ALL of the following:
• Budget: {budget_level} — every hotel, restaurant, and activity must match this tier
• Pace: {pace} — {pace_description}
• Interests: {interests} — at least 60% of activities must directly reflect these
• Constraints: {constraints} — these are hard rules, never violate them
```

`pace_description` values injected at runtime:
- `relaxed` → "max 2–3 activities per day, long lunches, afternoon rest"
- `moderate` → "3–4 activities per day with breathing room between"
- `packed` → "5–6 activities per day, efficient transitions, no dead time"

**Activity depth rules (strengthened):**
- Always use specific venue name — never "a local market", always "Mercado de San Miguel"
- Every sightseeing/cultural block: 2–3 sentences (what it is, why it suits this traveler, one practical tip)
- Each block must include: how to get there, recommended duration, best arrival time, book-ahead flag (yes/no)

**Budget weaving (new injection per tier):**
- `budget` → free/cheap attractions, street food, hostels, public transit everywhere
- `mid_range` → paid attractions, sit-down restaurants ($$), 3-star hotels, occasional taxi
- `luxury` → skip-the-line tickets, tasting menus ($$$), 4–5 star hotels, private transfers

**Logistics block (strengthened):**
`logistics_notes` must cover: exact transport between each day's areas, which activities need advance booking and how far ahead, rough daily spend estimate at the stated budget level.

### New `PERSONALIZATION_CHECK_PROMPT`

```
You are a quality reviewer for a travel itinerary.

The traveler requested:
• Interests: {interests}
• Budget: {budget_level}
• Pace: {pace} ({pace_description})
• Constraints: {constraints}

Review the itinerary below. For each day, check:
1. Do activities reflect the stated interests? (flag and fix mismatches)
2. Are hotel/restaurant tiers consistent with budget_level?
3. Does the number of activities per day match the stated pace?
4. Are all constraints honored without exception?

Return the same itinerary JSON with violations fixed.
Only change fields that violate the above — do not rewrite the whole plan.
```

### `QUESTION_ANSWER_PROMPT` — Minor Tightening

Existing instruction strengthened: if search results lack specific names, use training knowledge to provide named recommendations and mark each with `[General knowledge — verify current status]`. Remove the hedge "search results did not return names" pattern entirely.

## Data Layer

### Parallel Search (`search_with_tavily_node`)

Replace the sequential `for query, search_fn in search_tasks` loop with `ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _run_search(args):
    query, search_fn = args
    return search_fn(query=query, max_results=5)

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(_run_search, task): task for task in search_tasks}
    for future in as_completed(futures):
        result = future.result()
        ...
```

Expected speedup: 5–6 serial calls (~8–10s) → parallel (~2–3s).

### Expedia Hotel Search (`tools.py`)

New function `expedia_hotel_search(destination, check_in, check_out, adults=2)`:
- Calls `mcp__claude_ai_Expedia__search_hotels` with destination + dates
- Maps results to a list of `HotelSuggestion`-compatible dicts (name, neighborhood, price_night, star_rating, description)
- Returns empty list on any failure — never raises
- Called only when `trip_request.start_date` is set; falls back to Tavily hotel queries when absent

Results are injected into `search_with_tavily_node` as a separate `expedia_context` key on state, then merged into the hotel section of the generation prompt.

## UI Changes (`app.py`)

### Clickable Follow-up Questions

In `_render_answer()` and `_render_itinerary()`, replace `st.markdown(f"- {q}")` for follow-up/open questions with `st.button()`:

```python
for i, q in enumerate(follow_up_questions):
    if st.button(f"💬 {q}", key=f"followup_{i}"):
        # inject into chat state and re-invoke graph
        state = dict(st.session_state.agent_state)
        msgs = list(state.get("messages", []))
        msgs.append({"role": "user", "content": q})
        state["messages"] = msgs
        state["intent"] = "unknown"
        state["tavily_context"] = []
        state["tool_call_count"] = 0
        state["final_response"] = None
        state["needs_clarification"] = False
        result_state = _run_graph(state)
        st.session_state.agent_state = result_state
        st.rerun()
```

Same pattern applies to `open_questions` in `_render_itinerary()`.

## Error Handling

- `personalization_check_node`: any exception → log warning, return original itinerary unchanged
- `expedia_hotel_search`: any exception → return `[]`, log warning, Tavily hotel queries fill the gap
- Parallel search: each future wrapped in try/except — one failing search does not abort the rest

## Testing

- Existing test suite continues to pass — no test-visible interface changes
- `personalization_check_node` tested with a fixture itinerary that has a known violation (luxury hotel in a budget plan) — verify the node patches it
- `expedia_hotel_search` mocked in tests (same pattern as `TavilyClient` mocking)
- Parallel search: mock all search functions, verify all queries were issued and results accumulated

## Out of Scope

- Streaming responses (separate initiative)
- Multi-language support
- Export to PDF / email
- Flight search via Expedia (hotel-only for now)
