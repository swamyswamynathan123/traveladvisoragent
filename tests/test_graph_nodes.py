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
    from graph import search_with_tavily_node
    state = _planning_state()
    state["tool_call_count"] = 6
    result = search_with_tavily_node(state)
    assert result["tool_call_count"] == 6  # unchanged


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
