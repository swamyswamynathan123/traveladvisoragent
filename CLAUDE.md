# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands assume the virtual environment is active. Activate it first:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

**Run the app:**
```bash
streamlit run app.py
```

**Run all tests (no API keys needed — all external calls are mocked):**
```bash
pytest tests/ -v
```

**Run a single test file:**
```bash
pytest tests/test_graph_nodes.py -v
```

**Run a single test:**
```bash
pytest tests/test_graph_nodes.py::test_validate_inputs_complete_planning -v
```

**Install/sync dependencies:**
```bash
pip install -r requirements.txt
```

**Windows one-click launch (creates venv, installs deps, starts app):**
```cmd
run.bat
```

## Architecture

The app has a strict layered dependency order — no circular imports:

```
schemas.py  ←  tools.py  ←  search.py  ─┐
                                          ├─  app.py
schemas.py  ←  tools.py  ←  prompts.py  ←  graph.py  ─┘
```

- **`schemas.py`** — All Pydantic models. No project imports. Defines both the data contracts (`TripRequest`, `ItineraryResponse`, `QuestionResponse`, `ClarificationResponse`, `TavilySearchOutput`, etc.) and the agent state model (`TravelAgentState`).

- **`tools.py`** — Wraps the Tavily Python SDK into a single `tavily_search()` function that returns a `TavilySearchOutput`. Reads `TAVILY_API_KEY` at module level; tests patch `tools.TAVILY_API_KEY` to bypass the early-return guard.

- **`search.py`** — `search_flights()` and `search_hotels()`. Each builds a Tavily query, calls `tavily_search()`, passes results to `gpt-4o-mini` via `with_structured_output(FlightResultList | HotelResultList)`, and returns a plain `list[dict]`. Returns `[]` on any error.

- **`prompts.py`** — Six string constants (`SYSTEM_PROMPT`, `INTENT_DETECTION_PROMPT`, `CLARIFICATION_PROMPT`, `ITINERARY_GENERATION_PROMPT`, `QUESTION_ANSWER_PROMPT`, `SCHEMA_REPAIR_PROMPT`). Pure strings, no logic.

- **`graph.py`** — The LangGraph graph. Key exports: `build_graph()` (returns a compiled graph) and `build_initial_state()` (returns a plain `dict`). Six nodes wired in order:
  1. `collect_requirements_node` — calls `gpt-4o-mini` to parse intent and extract `trip_request` / `travel_question` from the latest message
  2. `validate_inputs_node` — pure logic, sets `needs_clarification`
  3. conditional edge via `should_clarify()` → either `ask_clarification_node` or `search_with_tavily_node`
  4. `search_with_tavily_node` — calls `tavily_search()` up to `MAX_TOOL_CALLS=9` times, accumulates results into `state["tavily_context"]`
  5. `generate_response_node` — calls `gpt-4o` with `llm.with_structured_output(PydanticModel)`; on failure retries once via `_retry_with_repair()`
  6. `respond_to_user_node` — appends the assistant message to `state["messages"]`

- **`app.py`** — Streamlit UI. Uses `st.session_state.agent_state` (plain dict) and `st.session_state.compiled_graph`. Every user action (Generate / Refine / chat input) merges new data into the state dict, calls `compiled_graph.invoke(state)`, saves the returned state, and calls `st.rerun()`. Three rendering helpers (`_render_itinerary`, `_render_answer`, `_render_clarification`) read from `state["final_response"]["data"]`.

## State Shape

The graph state is a plain `dict` (not a TypedDict at runtime). Key fields:

| Field | Set by | Purpose |
|---|---|---|
| `intent` | `collect_requirements_node` | `"planning"`, `"question"`, or `"unknown"` |
| `trip_request` | `collect_requirements_node` / sidebar form | Dict matching `TripRequest` schema |
| `travel_question` | `collect_requirements_node` | `{"question": str, "context": str\|None}` |
| `collected_info` | `collect_requirements_node` | Flags including `is_complete_for_planning` |
| `tavily_context` | `search_with_tavily_node` | List of result dicts, accumulates across calls |
| `tool_call_count` | `search_with_tavily_node` | Bounded by `MAX_TOOL_CALLS = 6` |
| `final_response` | `ask_clarification_node` / `generate_response_node` | `{"type": "itinerary"\|"answer"\|"clarification", "data": dict}` |
| `messages` | `app.py` + `respond_to_user_node` | Chat history as `[{"role": ..., "content": ...}]` |
| `needs_clarification` | `validate_inputs_node` | Drives the conditional edge |

## Testing Conventions

- Tests mock `tools.TAVILY_API_KEY` **and** `tools.TavilyClient` together — patching only `TavilyClient` is not enough because the API key guard runs first.
- Graph node tests import functions lazily inside each test to avoid module-level side effects.
- No integration tests exist; the test suite requires zero API keys.
