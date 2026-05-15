import pytest
from pydantic import ValidationError
from schemas import (
    TripRequest, TravelQuestion, Source, TimeBlock, DayPlan, AlternativePlan,
    ItineraryResponse, QuestionResponse, ClarificationResponse,
    CollectedInfoFlags, TavilyResult, TavilySearchOutput,
    BudgetLevel, TravelerType, Pace, TravelAgentState,
)


def test_trip_request_defaults():
    req = TripRequest(destination="Paris")
    assert req.destination == "Paris"
    assert req.budget_level == BudgetLevel.mid_range
    assert req.pace == Pace.moderate
    assert req.interests == []
    assert req.constraints == []


def test_trip_request_full():
    req = TripRequest(
        destination="Tokyo",
        duration_days=7,
        start_date="2026-06-01",
        origin="New York",
        interests=["food", "culture"],
        budget_level=BudgetLevel.luxury,
        pace=Pace.packed,
        constraints=["wheelchair accessible"],
    )
    assert req.duration_days == 7
    assert "food" in req.interests


def test_time_block_valid():
    block = TimeBlock(time_of_day="morning", activity="Visit Eiffel Tower", location="Paris")
    assert block.time_of_day == "morning"


def test_time_block_invalid_time_of_day():
    with pytest.raises(ValidationError):
        TimeBlock(time_of_day="noon", activity="Lunch")


def test_day_plan():
    plan = DayPlan(
        day_number=1,
        theme="Iconic Paris",
        blocks=[
            TimeBlock(time_of_day="morning", activity="Eiffel Tower"),
            TimeBlock(time_of_day="afternoon", activity="Louvre"),
        ],
    )
    assert len(plan.blocks) == 2


def test_itinerary_response():
    resp = ItineraryResponse(
        destination="Paris",
        duration="3 days",
        itinerary=[
            DayPlan(day_number=1, blocks=[TimeBlock(time_of_day="morning", activity="Eiffel Tower")])
        ],
        logistics_notes="Take the metro everywhere.",
    )
    assert resp.alternatives == []
    assert resp.assumptions == []
    assert resp.sources == []


def test_question_response_defaults():
    resp = QuestionResponse(answer="Paris is best visited in spring.")
    assert resp.supporting_points == []
    assert resp.follow_up_questions == []
    assert resp.sources == []


def test_clarification_response():
    resp = ClarificationResponse(
        message="Where would you like to go?",
        missing_fields=["destination"],
        open_questions=["What is your destination?"],
    )
    assert len(resp.missing_fields) == 1


def test_tavily_search_output_ok():
    out = TavilySearchOutput(
        results=[TavilyResult(title="T", url="https://x.com", content_snippet="snippet")],
        query="Paris tips",
        tool_status="ok",
    )
    assert out.tool_status == "ok"
    assert len(out.results) == 1


def test_tavily_search_output_error():
    out = TavilySearchOutput(
        results=[],
        query="test",
        tool_status="error",
        error_message="Timeout",
    )
    assert out.error_message == "Timeout"


def test_collected_info_flags_defaults():
    flags = CollectedInfoFlags()
    assert flags.has_destination is False
    assert flags.is_complete_for_planning is False


def test_travel_agent_state_defaults():
    state = TravelAgentState()
    assert state.intent == "unknown"
    assert state.tool_call_count == 0
    assert state.needs_clarification is False
    assert state.tavily_context == []
    assert state.messages == []
