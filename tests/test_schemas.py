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
