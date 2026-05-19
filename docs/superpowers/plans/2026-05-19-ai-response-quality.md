# AI Response Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix generic, impersonal, unreliable, and shallow AI responses by switching to Claude Sonnet 4.6, parallelising searches, adding a personalization-check pass, and rewriting prompts with an explicit personalization contract.

**Architecture:** A new `personalization_check_node` (Claude Haiku) is inserted between `generate_response` and `respond_to_user` for planning intents — it reads the generated itinerary against the user's stated preferences and patches violations. Tavily searches are parallelised with `ThreadPoolExecutor`. Hotel searches target booking sites via a new `expedia_hotel_search()` wrapper. `collect_requirements_node` and `ask_clarification_node` are migrated from fragile `json.loads()` to `with_structured_output()`.

**Tech Stack:** langchain-anthropic, Claude Sonnet 4.6 (generation), Claude Haiku 4.5 (personalization check), concurrent.futures.ThreadPoolExecutor, existing LangGraph + Streamlit stack.

---

## File Map

| File | Change |
|---|---|
| `requirements.txt` | Add `langchain-anthropic` |
| `schemas.py` | Add `IntentDetectionOutput` Pydantic model |
| `prompts.py` | Rewrite `ITINERARY_GENERATION_PROMPT`, tighten `QUESTION_ANSWER_PROMPT`, add `PERSONALIZATION_CHECK_PROMPT` |
| `tools.py` | Add `expedia_hotel_search()` |
| `graph.py` | Update `_llm()`, add `PACE_DESCRIPTIONS`/`BUDGET_GUIDANCE`, fix two nodes to use structured output, parallelise search node, update `_generate_itinerary`, add `_format_hotel_context`, add `personalization_check_node` + `should_check_personalization`, update `build_graph()`, update `build_initial_state()` |
| `app.py` | Replace static follow-up `st.markdown` with `st.button` in `_render_answer` and `_render_itinerary` |
| `tests/test_graph_nodes.py` | Tests for structured output nodes, parallel search, personalization check |
| `tests/test_tools.py` | Tests for `expedia_hotel_search` |

---

## Task 1: Add `IntentDetectionOutput` to schemas.py

**Files:**
- Modify: `schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas.py`:

```python
def test_intent_detection_output_planning():
    from schemas import IntentDetectionOutput
    obj = IntentDetectionOutput(
        intent="planning",
        trip_request={"destination": "Paris", "duration_days": 5},
        travel_question=None,
        clarification_needed=[],
    )
    assert obj.intent == "planning"
    assert obj.trip_request["destination"] == "Paris"
    assert obj.clarification_needed == []


def test_intent_detection_output_question():
    from schemas import IntentDetectionOutput
    obj = IntentDetectionOutput(
        intent="question",
        trip_request=None,
        travel_question="Best time to visit Tokyo?",
        clarification_needed=[],
    )
    assert obj.intent == "question"
    assert obj.travel_question == "Best time to visit Tokyo?"


def test_intent_detection_output_defaults():
    from schemas import IntentDetectionOutput
    obj = IntentDetectionOutput(intent="unknown")
    assert obj.trip_request is None
    assert obj.travel_question is None
    assert obj.clarification_needed == []
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_schemas.py::test_intent_detection_output_planning -v
```
Expected: `ImportError: cannot import name 'IntentDetectionOutput'`

- [ ] **Step 3: Add `IntentDetectionOutput` to `schemas.py`**

After the `TravelQuestion` class (around line 47), add:

```python
class IntentDetectionOutput(BaseModel):
    intent: Literal["planning", "question", "unknown"] = "unknown"
    trip_request: Optional[dict] = None
    travel_question: Optional[str] = None
    clarification_needed: List[str] = Field(default_factory=list)
```

Add `"unknown"` to the existing `Literal` import if needed — the schemas file already imports `Literal` from `typing`.

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_schemas.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```
git add schemas.py tests/test_schemas.py
git commit -m "feat: add IntentDetectionOutput schema for structured LLM output"
```

---

## Task 2: Add `langchain-anthropic` + update `_llm()` helper

**Files:**
- Modify: `requirements.txt`, `graph.py`

- [ ] **Step 1: Add `langchain-anthropic` to requirements.txt**

Add after `langchain-openai==1.2.1`:

```
langchain-anthropic==0.3.15
```

- [ ] **Step 2: Install it**

```
pip install langchain-anthropic==0.3.15
```

Expected: installs without error (uses existing `anthropic` SDK under the hood)

- [ ] **Step 3: Update `_llm()` in `graph.py`**

Replace the current `_llm` function (lines 48–49):

```python
def _llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)
    return ChatOpenAI(model=model, temperature=temperature)
```

- [ ] **Step 4: Smoke-test the import**

```
python -c "from graph import _llm; print('ok')"
```
Expected: `ok`

- [ ] **Step 5: Commit**

```
git add requirements.txt graph.py
git commit -m "feat: add langchain-anthropic and extend _llm() to support Claude models"
```

---

## Task 3: Fix `collect_requirements_node` — structured output

**Files:**
- Modify: `graph.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph_nodes.py`:

```python
def test_collect_requirements_planning_intent():
    from graph import collect_requirements_node
    from schemas import IntentDetectionOutput
    from unittest.mock import patch, MagicMock

    mock_output = IntentDetectionOutput(
        intent="planning",
        trip_request={"destination": "Paris", "duration_days": 5},
        travel_question=None,
        clarification_needed=[],
    )

    with patch("graph._llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_fn.return_value = mock_llm

        state = {"messages": [{"role": "user", "content": "Plan a 5-day trip to Paris"}]}
        result = collect_requirements_node(state)

    assert result["intent"] == "planning"
    assert result["trip_request"]["destination"] == "Paris"
    assert result["collected_info"]["has_destination"] is True
    assert result["collected_info"]["is_complete_for_planning"] is True


def test_collect_requirements_question_intent():
    from graph import collect_requirements_node
    from schemas import IntentDetectionOutput
    from unittest.mock import patch, MagicMock

    mock_output = IntentDetectionOutput(
        intent="question",
        trip_request=None,
        travel_question="Best time to visit Tokyo?",
        clarification_needed=[],
    )

    with patch("graph._llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_fn.return_value = mock_llm

        state = {"messages": [{"role": "user", "content": "Best time to visit Tokyo?"}]}
        result = collect_requirements_node(state)

    assert result["intent"] == "question"
    assert result["travel_question"]["question"] == "Best time to visit Tokyo?"


def test_collect_requirements_empty_messages_returns_state():
    from graph import collect_requirements_node
    state = {"messages": []}
    result = collect_requirements_node(state)
    assert result == state
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_graph_nodes.py::test_collect_requirements_planning_intent -v
```
Expected: FAIL — node still does `json.loads(response.content)`, mock returns an object not a string

- [ ] **Step 3: Rewrite `collect_requirements_node` in `graph.py`**

Replace lines 65–126 with:

```python
def collect_requirements_node(state: dict) -> dict:
    """Parse the latest user message to extract intent and request details."""
    from schemas import IntentDetectionOutput
    messages = state.get("messages", [])
    if not messages:
        return state

    latest = messages[-1].get("content", "")
    prompt = INTENT_DETECTION_PROMPT.format(user_message=latest)

    try:
        structured_llm = _llm().with_structured_output(IntentDetectionOutput)
        parsed: IntentDetectionOutput = structured_llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )

        intent = parsed.intent
        trip_req = parsed.trip_request or {}
        travel_q = parsed.travel_question
        clarification_needed = parsed.clarification_needed

        collected_info = {
            "has_destination": bool(trip_req.get("destination")),
            "has_duration": bool(trip_req.get("duration_days")),
            "has_dates": bool(trip_req.get("start_date")),
            "has_interests": bool(trip_req.get("interests")),
            "has_budget": bool(trip_req.get("budget_level")),
            "is_complete_for_planning": bool(
                trip_req.get("destination")
                and (trip_req.get("duration_days") or trip_req.get("start_date"))
            ),
        }

        is_planning_complete = collected_info.get("is_complete_for_planning", False)
        question_words = ("what", "when", "where", "how", "which", "why", "is ", "are ",
                          "do ", "can ", "should ", "would ", "?")
        message_is_question = any(latest.lower().startswith(w) or latest.endswith("?")
                                  for w in question_words)
        if intent == "planning" and not is_planning_complete and message_is_question:
            intent = "question"
            travel_q = travel_q or latest

        resolved_question = travel_q or (latest if intent == "question" else None)

        return {
            **state,
            "intent": intent,
            "trip_request": trip_req if intent == "planning" else state.get("trip_request"),
            "travel_question": (
                {"question": resolved_question, "context": None}
                if resolved_question
                else state.get("travel_question")
            ),
            "collected_info": collected_info,
            "clarification_questions": clarification_needed if intent == "planning" else [],
        }

    except Exception as exc:
        logger.error("collect_requirements_node failed: %s", exc)
        return {**state, "error": str(exc)}
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_graph_nodes.py::test_collect_requirements_planning_intent tests/test_graph_nodes.py::test_collect_requirements_question_intent tests/test_graph_nodes.py::test_collect_requirements_empty_messages_returns_state -v
```
Expected: all 3 PASS

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```
git add graph.py tests/test_graph_nodes.py
git commit -m "refactor: migrate collect_requirements_node to with_structured_output"
```

---

## Task 4: Fix `ask_clarification_node` — structured output

**Files:**
- Modify: `graph.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graph_nodes.py`:

```python
def test_ask_clarification_node_returns_clarification_response():
    from graph import ask_clarification_node
    from schemas import ClarificationResponse
    from unittest.mock import patch, MagicMock

    mock_output = ClarificationResponse(
        message="Where would you like to go, and for how long?",
        missing_fields=["destination", "duration_days"],
        open_questions=["What is your destination?", "How many days?"],
    )

    with patch("graph._llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = mock_output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_fn.return_value = mock_llm

        state = {
            "trip_request": {},
            "clarification_questions": ["destination", "duration_days"],
        }
        result = ask_clarification_node(state)

    assert result["final_response"]["type"] == "clarification"
    data = result["final_response"]["data"]
    assert "destination" in data["missing_fields"]
    assert len(data["open_questions"]) == 2
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_graph_nodes.py::test_ask_clarification_node_returns_clarification_response -v
```
Expected: FAIL

- [ ] **Step 3: Rewrite `ask_clarification_node` in `graph.py`**

Replace lines 150–181 with:

```python
def ask_clarification_node(state: dict) -> dict:
    """Generate a friendly clarification request for missing information."""
    trip_req = state.get("trip_request") or {}
    known_info = json.dumps(trip_req, indent=2)
    missing = state.get("clarification_questions") or ["destination", "duration_days"]

    prompt = CLARIFICATION_PROMPT.format(
        known_info=known_info,
        missing_fields=", ".join(missing),
    )

    try:
        structured_llm = _llm(temperature=0.3).with_structured_output(ClarificationResponse)
        data: ClarificationResponse = structured_llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
    except Exception as exc:
        logger.error("ask_clarification_node failed: %s", exc)
        data = ClarificationResponse(
            message="I'd love to help plan your trip! Could you tell me where you'd like to go and for how long?",
            missing_fields=missing,
            open_questions=[
                "What is your destination?",
                "How many days will you be traveling?",
            ],
        )

    return {
        **state,
        "final_response": {"type": "clarification", "data": data.model_dump()},
        "draft": data.message,
    }
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_graph_nodes.py::test_ask_clarification_node_returns_clarification_response -v
```
Expected: PASS

- [ ] **Step 5: Run full suite**

```
pytest tests/ -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```
git add graph.py tests/test_graph_nodes.py
git commit -m "refactor: migrate ask_clarification_node to with_structured_output"
```

---

## Task 5: Rewrite `ITINERARY_GENERATION_PROMPT` with personalization contract

**Files:**
- Modify: `prompts.py`, `graph.py`

- [ ] **Step 1: Add `PACE_DESCRIPTIONS` and `BUDGET_GUIDANCE` dicts to `graph.py`**

After the `MAX_TOOL_CALLS = 9` line, add:

```python
PACE_DESCRIPTIONS = {
    "relaxed": "max 2–3 activities per day, allow long lunches and afternoon rest",
    "moderate": "3–4 activities per day with breathing room between",
    "packed": "5–6 activities per day, efficient transitions, no dead time",
}

BUDGET_GUIDANCE = {
    "budget": "prioritise free attractions, street food and markets, hostels/guesthouses, public transit exclusively — avoid paid tours and taxis",
    "mid_range": "mix of paid attractions and free ones, sit-down restaurants ($$), 3-star hotels, occasional taxi",
    "luxury": "premium experiences: skip-the-line tickets, tasting menus ($$$), 4–5 star hotels, private transfers and guides",
}
```

- [ ] **Step 2: Add `_format_hotel_context` helper to `graph.py`**

After `_format_tavily_context`, add:

```python
def _format_hotel_context(tavily_context: list) -> str:
    items = [r for r in tavily_context if r.get("source_type") == "hotel_search"]
    if not items:
        return "No hotel search results available."
    lines: list[str] = []
    for i, item in enumerate(items[:5], start=1):
        lines.append(f"\n[H{i}] {item.get('title', 'Hotel')}")
        lines.append(f"URL: {item.get('url', '')}")
        lines.append(f"Details: {item.get('content_snippet', '')}")
    return "\n".join(lines)
```

Also update `_format_tavily_context` to exclude hotel results so they don't pollute the activity search context:

```python
def _format_tavily_context(tavily_context: list) -> str:
    if not tavily_context:
        return "No search results available. Use general knowledge and mark claims accordingly."
    items = [r for r in tavily_context if r.get("source_type") != "hotel_search"]
    if not items:
        return "No search results available. Use general knowledge and mark claims accordingly."
    lines: list[str] = []
    for i, item in enumerate(items[:10], start=1):
        lines.append(f"\n[{i}] {item.get('title', 'Unknown source')}")
        lines.append(f"URL: {item.get('url', '')}")
        lines.append(f"Snippet: {item.get('content_snippet', '')}")
    return "\n".join(lines)
```

- [ ] **Step 3: Replace `ITINERARY_GENERATION_PROMPT` in `prompts.py`**

```python
ITINERARY_GENERATION_PROMPT = """\
=== PERSONALIZATION CONTRACT ===
Before writing anything, you MUST honor ALL of the following for every day:
• Budget: {budget_level} — {budget_guidance}
• Pace: {pace} — {pace_description}
• Interests: {interests_list} — at least 60% of activities must directly reflect these interests
• Constraints: {constraints_list} — these are HARD rules, never violate them under any circumstance

=== TRIP REQUEST ===
{trip_request}

=== TRAVELER PROFILE ===
{user_profile}

=== WEB SEARCH RESULTS ===
{tavily_context}

=== HOTEL SEARCH RESULTS ===
{hotel_context}

=== ACTIVITY DEPTH RULES ===
For EVERY activity block:
1. Use the specific venue name — never "a local market", always e.g. "Mercado de San Miguel"
2. Write 2–3 sentences in notes: what it is, why it suits THIS traveler given their interests, one practical tip
3. Specify transport: exact metro line, bus number, or "10-min walk from [landmark]"
4. Set duration_hours to the recommended time at the venue
5. Include: best arrival time (e.g. "arrive before 9am to avoid queues"), book-ahead flag
6. For dining blocks: use specific restaurant names from search results. If none available, write "Find a [cuisine] restaurant near [neighborhood]" — never vague descriptions or "no specific details available"

=== LOGISTICS RULES ===
logistics_notes must cover:
• Exact transport between each day's areas (metro line numbers, bus routes, taxi estimate in local currency)
• Which activities need advance booking and how far ahead (e.g. "Book Alhambra tickets 2–3 weeks ahead")
• Rough daily spend estimate at {budget_level} level in local currency

Generate a complete itinerary. Return ONLY valid JSON matching this schema exactly:
{schema}

Additional instructions:
1. Build morning/afternoon/evening blocks for every day.
2. Include at least one rainy-day alternative under "alternatives".
3. Populate hotel_suggestions with 2–3 hotels per city at {budget_level} tier. Use hotel names from the Hotel Search Results section first; use training knowledge as fallback and mark notes as "[General knowledge — verify availability before booking]".
4. Populate restaurant_suggestions with 2–3 per city matching the traveler's interests and constraints. Use search results first; training knowledge marked "[General knowledge — verify current status before visiting]" as fallback.
5. List every assumption you made in "assumptions".
6. List unresolved questions in "open_questions".
7. Populate "sources" only from search results you actually referenced."""
```

- [ ] **Step 4: Update `_generate_itinerary` in `graph.py` to pass new format args**

Replace the `prompt = ITINERARY_GENERATION_PROMPT.format(...)` call (inside `_generate_itinerary`) with:

```python
trip_req = state.get("trip_request") or {}
pace = trip_req.get("pace", "moderate")
budget_level = trip_req.get("budget_level", "mid_range")
interests = trip_req.get("interests", [])
constraints = trip_req.get("constraints", [])

prompt = ITINERARY_GENERATION_PROMPT.format(
    trip_request=json.dumps(trip_req, indent=2),
    user_profile=json.dumps(state.get("user_profile") or {}, indent=2),
    tavily_context=_format_tavily_context(state.get("tavily_context", [])),
    hotel_context=_format_hotel_context(state.get("tavily_context", [])),
    pace_description=PACE_DESCRIPTIONS.get(pace, PACE_DESCRIPTIONS["moderate"]),
    budget_guidance=BUDGET_GUIDANCE.get(budget_level, BUDGET_GUIDANCE["mid_range"]),
    interests_list=", ".join(str(i) for i in interests) if interests else "general sightseeing",
    constraints_list=", ".join(str(c) for c in constraints) if constraints else "none",
    budget_level=budget_level,
    pace=pace,
    schema=schema_str,
)
```

- [ ] **Step 5: Smoke test — confirm app still imports**

```
python -c "from graph import build_graph; build_graph(); print('ok')"
```
Expected: `ok`

- [ ] **Step 6: Commit**

```
git add prompts.py graph.py
git commit -m "feat: rewrite ITINERARY_GENERATION_PROMPT with personalization contract and activity depth rules"
```

---

## Task 6: Add `PERSONALIZATION_CHECK_PROMPT` + tighten `QUESTION_ANSWER_PROMPT`

**Files:**
- Modify: `prompts.py`

- [ ] **Step 1: Add `PERSONALIZATION_CHECK_PROMPT` to `prompts.py`**

Append after `SCHEMA_REPAIR_PROMPT`:

```python
PERSONALIZATION_CHECK_PROMPT = """\
You are a quality reviewer for a travel itinerary.

The traveler requested:
• Interests: {interests_list}
• Budget: {budget_level} — {budget_guidance}
• Pace: {pace} — {pace_description}
• Constraints: {constraints_list}

Review the itinerary below and fix ONLY violations of the above. For each day, check:
1. Do activities reflect the stated interests? Replace mismatches with interest-aligned alternatives.
2. Are hotel/restaurant tiers consistent with {budget_level}? Correct any tier mismatches.
3. Does the number of activity blocks per day match the stated pace? Add or remove blocks as needed.
4. Are all constraints honored in every block? Fix any violations immediately.

Do NOT rewrite the whole plan — only change fields that violate the contract above.

Itinerary to review:
{itinerary}

Return the corrected itinerary as valid JSON matching this schema exactly:
{schema}"""
```

- [ ] **Step 2: Tighten `QUESTION_ANSWER_PROMPT` in `prompts.py`**

Replace instruction 7 in the existing prompt:

```
7. Always give specific named recommendations — never vague phrases like "several restaurants can be found" or "a number of options are available." If search results contain specific names, use them. If search results lack specific names, draw on your training knowledge and mark each recommendation with "[General knowledge — verify current status]". NEVER say you could not find names or redirect the user to search engines — always provide actionable named places.
```

- [ ] **Step 3: Smoke test**

```
python -c "from prompts import PERSONALIZATION_CHECK_PROMPT; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```
git add prompts.py
git commit -m "feat: add PERSONALIZATION_CHECK_PROMPT and tighten QUESTION_ANSWER_PROMPT specificity rule"
```

---

## Task 7: Add `expedia_hotel_search` to `tools.py`

**Files:**
- Modify: `tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tools.py`:

```python
def test_expedia_hotel_search_returns_hotel_results():
    from tools import expedia_hotel_search

    fake_result = {
        "title": "Generator Paris — Budget Hotel",
        "url": "https://booking.com/generator-paris",
        "content": "Trendy budget hotel near Canal Saint-Martin. Dorms and private rooms.",
        "type": "web",
    }
    mock_client = _make_mock_client([fake_result])
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        out = expedia_hotel_search(destination="Paris", check_in="2026-06-01", budget_level="budget")

    assert out.tool_status == "ok"
    assert len(out.results) == 1
    assert "Generator Paris" in out.results[0].title
    assert out.results[0].source_type == "hotel_search"


def test_expedia_hotel_search_returns_empty_on_api_error():
    from tools import expedia_hotel_search

    mock_client = _make_mock_client(raise_exc=RuntimeError("timeout"))
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        out = expedia_hotel_search(destination="Paris", check_in="2026-06-01")

    assert out.tool_status == "error"
    assert out.results == []


def test_expedia_hotel_search_uses_budget_label():
    from tools import expedia_hotel_search

    mock_client = _make_mock_client([])
    with patch("tools.TAVILY_API_KEY", FAKE_KEY), patch("tools.TavilyClient", return_value=mock_client):
        expedia_hotel_search(destination="Tokyo", check_in="2026-07-10", budget_level="luxury")

    query_used = mock_client.search.call_args.kwargs["query"]
    assert "luxury" in query_used.lower()
    assert "Tokyo" in query_used
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_tools.py::test_expedia_hotel_search_returns_hotel_results -v
```
Expected: `ImportError: cannot import name 'expedia_hotel_search'`

- [ ] **Step 3: Implement `expedia_hotel_search` in `tools.py`**

Append after the `tripadvisor_search` function:

```python
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
```

- [ ] **Step 4: Run to verify pass**

```
pytest tests/test_tools.py -v
```
Expected: all tests PASS (including 3 new ones)

- [ ] **Step 5: Commit**

```
git add tools.py tests/test_tools.py
git commit -m "feat: add expedia_hotel_search with hotel_search source tag"
```

---

## Task 8: Parallelise `search_with_tavily_node` + add hotel search

**Files:**
- Modify: `graph.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_graph_nodes.py`:

```python
def test_search_node_fires_all_queries_and_accumulates_results():
    from graph import search_with_tavily_node
    from schemas import TavilySearchOutput, TavilyResult

    fired = []

    def recording_fn(**kwargs):
        fired.append(kwargs.get("query", kwargs.get("destination", "?")))
        return TavilySearchOutput(
            results=[TavilyResult(title="R", url="u", content_snippet="s")],
            query=str(kwargs.get("query", "hotel")),
            tool_status="ok",
        )

    state = _planning_state()  # destination=Tokyo, no start_date
    state["tool_call_count"] = 0

    with patch("graph.tavily_search", side_effect=recording_fn), \
         patch("graph.tripadvisor_search", side_effect=recording_fn), \
         patch("graph.tavily_search_advanced", side_effect=recording_fn):
        result = search_with_tavily_node(state)

    assert len(fired) >= 3  # at minimum: attractions + travel tips + hotels
    assert len(result["tavily_context"]) == len(fired)
    assert result["tool_call_count"] == len(fired)


def test_search_node_calls_hotel_search_when_start_date_set():
    from graph import search_with_tavily_node
    from schemas import TavilySearchOutput

    state = _planning_state()
    state["trip_request"]["start_date"] = "2026-08-01"
    state["tool_call_count"] = 0

    mock_out = TavilySearchOutput(results=[], query="q", tool_status="ok")

    with patch("graph.tavily_search", return_value=mock_out), \
         patch("graph.tripadvisor_search", return_value=mock_out), \
         patch("graph.tavily_search_advanced", return_value=mock_out), \
         patch("graph.expedia_hotel_search", return_value=mock_out) as mock_hotel:
        search_with_tavily_node(state)

    mock_hotel.assert_called_once()
    call_kwargs = mock_hotel.call_args.kwargs
    assert call_kwargs["destination"] == "Tokyo"
    assert call_kwargs["check_in"] == "2026-08-01"
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_graph_nodes.py::test_search_node_fires_all_queries_and_accumulates_results -v
```
Expected: FAIL — `expedia_hotel_search` not imported in graph.py yet

- [ ] **Step 3: Update imports in `graph.py`**

Change the `from tools import ...` line to:

```python
from tools import tavily_search, tavily_search_advanced, tripadvisor_search, expedia_hotel_search
```

Also add `from concurrent.futures import ThreadPoolExecutor` at the top of `search_with_tavily_node` (alongside the existing `import re`).

- [ ] **Step 4: Replace `search_with_tavily_node` in `graph.py`**

Replace lines 184–302 with:

```python
def search_with_tavily_node(state: dict) -> dict:
    """Run searches in parallel to gather travel information (max MAX_TOOL_CALLS total)."""
    from concurrent.futures import ThreadPoolExecutor
    import re

    intent = state.get("intent", "unknown")
    call_count: int = state.get("tool_call_count", 0)

    if call_count >= MAX_TOOL_CALLS:
        logger.warning("MAX_TOOL_CALLS reached — skipping search")
        return state

    # Each entry is (fn, kwargs, is_hotel_search)
    search_tasks: list[tuple] = []

    if intent == "planning":
        trip_req = state.get("trip_request") or {}
        dest = trip_req.get("destination", "")
        interests = trip_req.get("interests", [])
        constraints = [str(c).lower() for c in trip_req.get("constraints", [])]
        constraint_text = " ".join(constraints)
        budget_level = trip_req.get("budget_level", "mid_range")
        start_date = trip_req.get("start_date", "")

        search_tasks += [
            (tripadvisor_search, {"query": f"{dest} top attractions things to do", "max_results": 5}, False),
            (tavily_search, {"query": f"{dest} travel tips transportation budget", "max_results": 5}, False),
        ]
        if interests:
            interests_str = " ".join(str(i) for i in interests[:2])
            search_tasks.append(
                (tripadvisor_search, {"query": f"{dest} {interests_str} recommendations", "max_results": 5}, False)
            )

        dietary_keywords = ["vegetarian", "vegan", "halal", "kosher", "gluten-free", "dairy-free"]
        cuisine_keywords = ["indian", "italian", "japanese", "chinese", "mexican", "thai",
                            "mediterranean", "french", "spanish", "greek", "middle eastern"]
        has_dietary = any(kw in constraint_text for kw in dietary_keywords)
        has_food_interest = any("food" in str(i).lower() or "dining" in str(i).lower() for i in interests)
        if has_dietary or has_food_interest:
            city_names = [c.strip() for c in re.split(r"[,+&]|\band\b", dest, flags=re.IGNORECASE) if c.strip()]
            search_cities = city_names if city_names else [dest]
            diet_type = next((kw for kw in dietary_keywords if kw in constraint_text), "") if has_dietary else ""
            cuisine_type = next((kw for kw in cuisine_keywords if kw in constraint_text), "")
            food_label = " ".join(filter(None, [cuisine_type, diet_type])) or "best"
            for city in search_cities:
                search_tasks.append((
                    tavily_search_advanced,
                    {"query": f"{food_label} restaurants {city} recommended named list 2024", "max_results": 5},
                    False,
                ))

        city_names = [c.strip() for c in re.split(r"[,+&]|\band\b", dest, flags=re.IGNORECASE) if c.strip()]
        hotel_cities = city_names if city_names else [dest]
        for city in hotel_cities[:3]:
            search_tasks.append((
                tripadvisor_search,
                {"query": f"best {budget_level.replace('_', ' ')} hotels {city} recommended 2024", "max_results": 5},
                False,
            ))

        if start_date:
            for city in hotel_cities[:2]:
                search_tasks.append((
                    expedia_hotel_search,
                    {"destination": city, "check_in": start_date, "budget_level": budget_level},
                    True,
                ))

    elif intent == "question":
        q = (state.get("travel_question") or {}).get("question", "")
        trip_req = state.get("trip_request") or {}
        dest = trip_req.get("destination", "")
        start_date = trip_req.get("start_date", "")

        restaurant_kw = ["restaurant", "eat", "food", "dining", "vegetarian", "vegan",
                         "halal", "kosher", "cafe", "cuisine", "where to eat"]
        cuisine_kw = ["indian", "italian", "japanese", "chinese", "mexican", "thai",
                      "mediterranean", "french", "spanish", "greek", "middle eastern",
                      "korean", "turkish", "lebanese", "moroccan"]
        is_restaurant_q = any(kw in q.lower() for kw in restaurant_kw)

        if is_restaurant_q:
            cuisine_type = next((kw for kw in cuisine_kw if kw in q.lower()), "")
            diet_type = next((kw for kw in ["vegetarian", "vegan", "halal", "kosher"] if kw in q.lower()), "")
            food_label = " ".join(filter(None, [cuisine_type, diet_type])) or "popular"
            dest_cities = [c.strip() for c in re.split(r"[,+&]|\band\b", dest, flags=re.IGNORECASE) if c.strip()] if dest else []
            q_only_cities = [c for c in dest_cities if c.lower() in q.lower()]
            search_cities = (q_only_cities or dest_cities)[:3]
            if search_cities:
                for city in search_cities:
                    search_tasks.append((
                        tavily_search_advanced,
                        {"query": f"best {food_label} restaurants {city} 2024 recommended", "max_results": 5},
                        False,
                    ))
            else:
                enriched_q = f"{food_label} restaurants {dest or q}".strip()
                search_tasks.append((tavily_search_advanced, {"query": enriched_q, "max_results": 5}, False))
        else:
            enriched_q = f"{q} {dest}".strip() if dest and dest.lower() not in q.lower() else q
            search_tasks.append((tavily_search, {"query": enriched_q, "max_results": 5}, False))
            if start_date and dest:
                search_tasks.append((tavily_search, {"query": f"{dest} weather {start_date}", "max_results": 5}, False))
    else:
        return state

    # Cap to remaining budget
    search_tasks = search_tasks[:MAX_TOOL_CALLS - call_count]

    def _run_task(task):
        fn, kwargs, _ = task
        return fn(**kwargs)

    accumulated: list = list(state.get("tavily_context", []))
    new_call_count = call_count

    with ThreadPoolExecutor(max_workers=min(len(search_tasks), 9)) as executor:
        results = list(executor.map(_run_task, search_tasks))

    for result in results:
        new_call_count += 1
        if result.tool_status == "ok":
            for r in result.results:
                accumulated.append({
                    "title": r.title,
                    "url": r.url,
                    "content_snippet": r.content_snippet,
                    "source_type": r.source_type,
                    "query": result.query,
                })

    return {**state, "tavily_context": accumulated, "tool_call_count": new_call_count}
```

- [ ] **Step 5: Run to verify pass**

```
pytest tests/test_graph_nodes.py -v
```
Expected: all tests PASS (including 2 new ones)

- [ ] **Step 6: Commit**

```
git add graph.py tests/test_graph_nodes.py
git commit -m "feat: parallelise Tavily searches with ThreadPoolExecutor and add hotel search"
```

---

## Task 9: Switch generation LLM to Claude Sonnet 4.6

**Files:**
- Modify: `graph.py`

- [ ] **Step 1: Update `_generate_itinerary` to use Claude Sonnet 4.6**

Change the first line of `_generate_itinerary`:

```python
# Before:
llm = _llm(model="gpt-4o", temperature=0.3)

# After:
llm = _llm(model="claude-sonnet-4-6", temperature=0.3)
```

- [ ] **Step 2: Update `_generate_answer` to use Claude Sonnet 4.6**

Change the first line of `_generate_answer`:

```python
# Before:
llm = _llm(model="gpt-4o", temperature=0.3)

# After:
llm = _llm(model="claude-sonnet-4-6", temperature=0.3)
```

- [ ] **Step 3: Verify no broken imports**

```
python -c "from graph import build_graph; build_graph(); print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Run full test suite**

```
pytest tests/ -v
```
Expected: all tests PASS (generation nodes are mocked in tests, so model string doesn't matter)

- [ ] **Step 5: Commit**

```
git add graph.py
git commit -m "feat: switch itinerary and answer generation to Claude Sonnet 4.6"
```

---

## Task 10: Add `personalization_check_node` + wire graph

**Files:**
- Modify: `graph.py`
- Test: `tests/test_graph_nodes.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_graph_nodes.py`:

```python
def test_personalization_check_patches_budget_violation():
    from graph import personalization_check_node
    from schemas import ItineraryResponse, DayPlan, TimeBlock, HotelSuggestion
    from unittest.mock import patch, MagicMock

    original = ItineraryResponse(
        destination="Paris",
        duration="5 days",
        itinerary=[
            DayPlan(
                day_number=1,
                theme="Arrival",
                blocks=[TimeBlock(time_of_day="afternoon", activity="Check in")],
            )
        ],
        hotel_suggestions=[
            HotelSuggestion(name="Four Seasons George V", city="Paris", budget_level="luxury")
        ],
        logistics_notes="Take metro.",
    )
    patched = ItineraryResponse(
        destination="Paris",
        duration="5 days",
        itinerary=original.itinerary,
        hotel_suggestions=[
            HotelSuggestion(name="Generator Paris", city="Paris", budget_level="budget")
        ],
        logistics_notes="Take metro.",
    )

    state = {
        "intent": "planning",
        "trip_request": {
            "destination": "Paris",
            "budget_level": "budget",
            "pace": "moderate",
            "interests": [],
            "constraints": [],
        },
        "itinerary_response": {"type": "itinerary", "data": original.model_dump()},
        "final_response": {"type": "itinerary", "data": original.model_dump()},
    }

    with patch("graph._llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = patched
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_fn.return_value = mock_llm

        result = personalization_check_node(state)

    hotels = result["itinerary_response"]["data"]["hotel_suggestions"]
    assert hotels[0]["name"] == "Generator Paris"


def test_personalization_check_skips_for_question_intent():
    from graph import personalization_check_node

    state = {
        "intent": "question",
        "final_response": {"type": "answer", "data": {"answer": "Spring is best."}},
        "itinerary_response": None,
    }
    result = personalization_check_node(state)
    assert result is state  # exact same object, no copy made


def test_personalization_check_falls_back_on_llm_error():
    from graph import personalization_check_node
    from schemas import ItineraryResponse, DayPlan, TimeBlock
    from unittest.mock import patch, MagicMock

    original_data = ItineraryResponse(
        destination="Paris",
        duration="5 days",
        itinerary=[DayPlan(day_number=1, theme="Arrival",
                           blocks=[TimeBlock(time_of_day="morning", activity="Arrive")])],
        logistics_notes="fly in",
    ).model_dump()

    state = {
        "intent": "planning",
        "trip_request": {"destination": "Paris", "budget_level": "mid_range",
                         "pace": "moderate", "interests": [], "constraints": []},
        "itinerary_response": {"type": "itinerary", "data": original_data},
        "final_response": {"type": "itinerary", "data": original_data},
    }

    with patch("graph._llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("Claude timeout")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_fn.return_value = mock_llm

        result = personalization_check_node(state)

    assert result["itinerary_response"]["data"]["destination"] == "Paris"


def test_should_check_personalization_planning():
    from graph import should_check_personalization
    assert should_check_personalization({"intent": "planning"}) == "personalization_check"


def test_should_check_personalization_question():
    from graph import should_check_personalization
    assert should_check_personalization({"intent": "question"}) == "respond_to_user"
```

- [ ] **Step 2: Run to verify failure**

```
pytest tests/test_graph_nodes.py::test_personalization_check_skips_for_question_intent -v
```
Expected: `ImportError: cannot import name 'personalization_check_node'`

- [ ] **Step 3: Add `personalization_check_node` and `should_check_personalization` to `graph.py`**

Insert after `generate_response_node` (before `respond_to_user_node`):

```python
def should_check_personalization(state: dict) -> str:
    """Conditional edge: route planning responses through quality check."""
    return "personalization_check" if state.get("intent") == "planning" else "respond_to_user"


def personalization_check_node(state: dict) -> dict:
    """Claude Haiku pass: verify itinerary honors user's personalization contract."""
    if state.get("intent") != "planning":
        return state

    itinerary_response = state.get("itinerary_response")
    if not itinerary_response:
        return state

    trip_req = state.get("trip_request") or {}
    pace = trip_req.get("pace", "moderate")
    budget_level = trip_req.get("budget_level", "mid_range")
    interests = trip_req.get("interests", [])
    constraints = trip_req.get("constraints", [])

    schema_str = json.dumps(ItineraryResponse.model_json_schema(), indent=2)
    prompt = PERSONALIZATION_CHECK_PROMPT.format(
        interests_list=", ".join(str(i) for i in interests) if interests else "general sightseeing",
        budget_level=budget_level,
        budget_guidance=BUDGET_GUIDANCE.get(budget_level, BUDGET_GUIDANCE["mid_range"]),
        pace=pace,
        pace_description=PACE_DESCRIPTIONS.get(pace, PACE_DESCRIPTIONS["moderate"]),
        constraints_list=", ".join(str(c) for c in constraints) if constraints else "none",
        itinerary=json.dumps(itinerary_response.get("data", {}), indent=2),
        schema=schema_str,
    )

    try:
        llm = _llm(model="claude-haiku-4-5-20251001", temperature=0.0)
        structured_llm = llm.with_structured_output(ItineraryResponse)
        patched: ItineraryResponse = structured_llm.invoke([HumanMessage(content=prompt)])
        patched_response = {"type": "itinerary", "data": patched.model_dump()}
        return {
            **state,
            "final_response": patched_response,
            "itinerary_response": patched_response,
        }
    except Exception as exc:
        logger.warning("personalization_check_node failed (%s), using original", exc)
        return state
```

Add `PERSONALIZATION_CHECK_PROMPT` to the imports from `prompts.py`:

```python
from prompts import (
    SYSTEM_PROMPT,
    INTENT_DETECTION_PROMPT,
    CLARIFICATION_PROMPT,
    ITINERARY_GENERATION_PROMPT,
    QUESTION_ANSWER_PROMPT,
    SCHEMA_REPAIR_PROMPT,
    PERSONALIZATION_CHECK_PROMPT,
)
```

- [ ] **Step 4: Wire the new node into `build_graph()`**

In `build_graph()`, replace:
```python
workflow.add_edge("generate_response", "respond_to_user")
```
with:
```python
workflow.add_node("personalization_check", personalization_check_node)
workflow.add_conditional_edges(
    "generate_response",
    should_check_personalization,
    {
        "personalization_check": "personalization_check",
        "respond_to_user": "respond_to_user",
    },
)
workflow.add_edge("personalization_check", "respond_to_user")
```

- [ ] **Step 5: Run to verify pass**

```
pytest tests/test_graph_nodes.py -v
```
Expected: all tests PASS

- [ ] **Step 6: Commit**

```
git add graph.py tests/test_graph_nodes.py
git commit -m "feat: add personalization_check_node (Claude Haiku) as post-generation quality gate"
```

---

## Task 11: Clickable follow-up questions in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace static follow-up rendering in `_render_answer`**

In `_render_answer` (around line 171–175), replace:
```python
follow_ups = data.get("follow_up_questions", [])
if follow_ups:
    st.subheader("🔮 Suggested Follow-ups")
    for q in follow_ups:
        st.markdown(f"- {q}")
```
with:
```python
follow_ups = data.get("follow_up_questions", [])
if follow_ups:
    st.subheader("🔮 Suggested Follow-ups")
    for i, q in enumerate(follow_ups):
        if st.button(f"💬 {q}", key=f"answer_followup_{i}"):
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

- [ ] **Step 2: Replace static open-question rendering in `_render_itinerary`**

In `_render_itinerary` (around line 139–141), replace:
```python
open_q = data.get("open_questions", [])
if open_q:
    st.subheader("❓ Open Questions")
    for q in open_q:
        st.markdown(f"- {q}")
```
with:
```python
open_q = data.get("open_questions", [])
if open_q:
    st.subheader("❓ Open Questions")
    for i, q in enumerate(open_q):
        if st.button(f"💬 {q}", key=f"itinerary_openq_{i}"):
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

- [ ] **Step 3: Run the app and manually verify**

```
streamlit run app.py
```

1. Enter a destination (e.g. "Lisbon"), set 4 days, click "Generate Itinerary"
2. Verify the itinerary renders with clickable "💬 ..." buttons under Open Questions
3. Click one button — verify it submits as a chat message and a Q&A answer appears
4. Ask a travel question via the chat input — verify follow-up question buttons appear in the answer
5. Click one — verify it chains into another answer

- [ ] **Step 4: Run full test suite one final time**

```
pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 5: Final commit**

```
git add app.py
git commit -m "feat: make follow-up and open questions clickable buttons in UI"
```
