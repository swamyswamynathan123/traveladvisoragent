import json
import pytest
from unittest.mock import patch, MagicMock
from schemas import TavilySearchOutput, TavilyResult


# ── helpers ──────────────────────────────────────────────────────────────────

def _planning_state(destination="Tokyo", duration=5, intent="planning"):
    from graph import build_initial_state
    state = build_initial_state()
    state["intent"] = intent
    state["trip_request"] = {
        "destination": destination,
        "duration_days": duration,
        "start_date": None,
        "origin": None,
        "interests": ["food"],
        "budget_level": "mid_range",
        "pace": "moderate",
        "constraints": [],
    }
    state["collected_info"] = {
        "has_destination": True,
        "has_duration": True,
        "has_dates": False,
        "has_interests": True,
        "has_budget": True,
        "is_complete_for_planning": True,
    }
    state["messages"] = [{"role": "user", "content": f"Plan a {duration}-day trip to {destination}"}]
    return state


def _question_state(question="Best time to visit Tokyo?"):
    from graph import build_initial_state
    state = build_initial_state()
    state["intent"] = "question"
    state["travel_question"] = {"question": question, "context": None}
    state["messages"] = [{"role": "user", "content": question}]
    return state


# ── build_initial_state ───────────────────────────────────────────────────────

def test_build_initial_state_defaults():
    from graph import build_initial_state
    state = build_initial_state()
    assert state["intent"] == "unknown"
    assert state["tool_call_count"] == 0
    assert state["needs_clarification"] is False
    assert state["tavily_context"] == []
    assert state["messages"] == []
    assert state["final_response"] is None


# ── validate_inputs_node ──────────────────────────────────────────────────────

def test_validate_inputs_complete_planning():
    from graph import validate_inputs_node
    state = _planning_state()
    result = validate_inputs_node(state)
    assert result["needs_clarification"] is False


def test_validate_inputs_incomplete_planning():
    from graph import build_initial_state, validate_inputs_node
    state = build_initial_state()
    state["intent"] = "planning"
    state["trip_request"] = {"destination": None, "duration_days": None}
    state["collected_info"] = {"is_complete_for_planning": False}
    result = validate_inputs_node(state)
    assert result["needs_clarification"] is True


def test_validate_inputs_question_with_question():
    from graph import validate_inputs_node
    state = _question_state()
    result = validate_inputs_node(state)
    assert result["needs_clarification"] is False


def test_validate_inputs_unknown_intent():
    from graph import build_initial_state, validate_inputs_node
    state = build_initial_state()
    state["intent"] = "unknown"
    result = validate_inputs_node(state)
    assert result["needs_clarification"] is True


# ── should_clarify ────────────────────────────────────────────────────────────

def test_should_clarify_true():
    from graph import should_clarify
    assert should_clarify({"needs_clarification": True}) == "ask_clarification"


def test_should_clarify_false():
    from graph import should_clarify
    assert should_clarify({"needs_clarification": False}) == "search_with_tavily"


# ── search_with_tavily_node ───────────────────────────────────────────────────

def test_search_node_skips_when_max_calls_reached():
    from graph import search_with_tavily_node, MAX_TOOL_CALLS
    state = _planning_state()
    state["tool_call_count"] = MAX_TOOL_CALLS
    result = search_with_tavily_node(state)
    assert result["tool_call_count"] == MAX_TOOL_CALLS  # unchanged — node skipped


def test_search_node_increments_call_count():
    from graph import search_with_tavily_node
    state = _planning_state()
    state["tool_call_count"] = 0

    mock_output = TavilySearchOutput(
        results=[TavilyResult(title="t", url="u", content_snippet="s")],
        query="Tokyo top attractions things to do",
        tool_status="ok",
    )
    with patch("graph.tavily_search", return_value=mock_output):
        result = search_with_tavily_node(state)

    assert result["tool_call_count"] > 0


def test_search_node_appends_results_to_context():
    from graph import search_with_tavily_node
    state = _planning_state()
    state["tool_call_count"] = 0

    mock_output = TavilySearchOutput(
        results=[TavilyResult(title="Tokyo Guide", url="https://x.com", content_snippet="great city")],
        query="Tokyo top attractions things to do",
        tool_status="ok",
    )
    with patch("graph.tavily_search", return_value=mock_output):
        result = search_with_tavily_node(state)

    assert len(result["tavily_context"]) >= 1
    assert any(r["title"] == "Tokyo Guide" for r in result["tavily_context"])


def test_search_node_handles_tavily_error_gracefully():
    from graph import search_with_tavily_node
    state = _planning_state()
    state["tool_call_count"] = 0

    mock_error = TavilySearchOutput(results=[], query="q", tool_status="error", error_message="API down")
    with patch("graph.tavily_search", return_value=mock_error):
        result = search_with_tavily_node(state)

    assert result["tool_call_count"] > 0  # still counted
    assert len(result["tavily_context"]) == 0


def test_search_node_question_intent():
    from graph import search_with_tavily_node
    state = _question_state()
    state["tool_call_count"] = 0

    mock_output = TavilySearchOutput(
        results=[TavilyResult(title="Q ans", url="https://y.com", content_snippet="spring is best")],
        query="Best time to visit Tokyo?",
        tool_status="ok",
    )
    with patch("graph.tavily_search", return_value=mock_output):
        result = search_with_tavily_node(state)

    assert result["tool_call_count"] >= 1


# ── respond_to_user_node ──────────────────────────────────────────────────────

def test_respond_to_user_adds_assistant_message_clarification():
    from graph import respond_to_user_node
    state = _planning_state()
    state["final_response"] = {
        "type": "clarification",
        "data": {"message": "Where would you like to go?", "missing_fields": [], "open_questions": []},
    }
    result = respond_to_user_node(state)
    msgs = result["messages"]
    assert any(m["role"] == "assistant" for m in msgs)


def test_respond_to_user_adds_assistant_message_itinerary():
    from graph import respond_to_user_node
    state = _planning_state()
    state["final_response"] = {
        "type": "itinerary",
        "data": {"destination": "Tokyo", "duration": "5 days", "itinerary": [], "logistics_notes": ""},
    }
    result = respond_to_user_node(state)
    msgs = result["messages"]
    assert any(m["role"] == "assistant" for m in msgs)


def test_respond_to_user_no_response_leaves_messages_unchanged():
    from graph import respond_to_user_node, build_initial_state
    state = build_initial_state()
    state["messages"] = [{"role": "user", "content": "hi"}]
    state["final_response"] = None
    result = respond_to_user_node(state)
    assert result["messages"] == [{"role": "user", "content": "hi"}]


# ── collect_requirements_node ─────────────────────────────────────────────────

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


# ── ask_clarification_node ───────────────────────────────────────────────────

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


# ── personalization_check_node ────────────────────────────────────────────────

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


def test_generate_itinerary_uses_streaming_llm():
    from graph import _generate_itinerary
    from schemas import ItineraryResponse, DayPlan, TimeBlock

    minimal_response = ItineraryResponse(
        destination="Paris",
        duration="3 days",
        itinerary=[DayPlan(day_number=1, theme="Arrival",
                           blocks=[TimeBlock(time_of_day="morning", activity="Arrive")])],
        logistics_notes="By train.",
    )

    with patch("graph._llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = minimal_response
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_fn.return_value = mock_llm

        result = _generate_itinerary(_planning_state())

    mock_llm_fn.assert_called_once_with(model="gpt-4o", temperature=0.3, streaming=True)
    assert result["final_response"]["type"] == "itinerary"
