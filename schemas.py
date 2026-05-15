from __future__ import annotations
from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class BudgetLevel(str, Enum):
    budget = "budget"
    mid_range = "mid_range"
    luxury = "luxury"


class TravelerType(str, Enum):
    solo = "solo"
    couple = "couple"
    family = "family"
    group = "group"


class Pace(str, Enum):
    relaxed = "relaxed"
    moderate = "moderate"
    packed = "packed"


class UserProfile(BaseModel):
    name: Optional[str] = None
    traveler_type: TravelerType = TravelerType.solo
    region: Optional[str] = None


class TripRequest(BaseModel):
    destination: str
    duration_days: Optional[int] = None
    start_date: Optional[str] = None
    origin: Optional[str] = None
    returning_to: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    budget_level: BudgetLevel = BudgetLevel.mid_range
    pace: Pace = Pace.moderate
    constraints: List[str] = Field(default_factory=list)


class TravelQuestion(BaseModel):
    question: str
    context: Optional[str] = None


class Source(BaseModel):
    title: str
    url: str
    snippet: str


class TimeBlock(BaseModel):
    time_of_day: Literal["morning", "afternoon", "evening"]
    activity: str
    location: Optional[str] = None
    notes: Optional[str] = None
    duration_hours: Optional[float] = None


class DayPlan(BaseModel):
    day_number: int
    date: Optional[str] = None
    theme: Optional[str] = None
    blocks: List[TimeBlock]
    tips: Optional[str] = None


class AlternativePlan(BaseModel):
    name: str
    description: str
    activities: List[str] = Field(default_factory=list)


class HotelSuggestion(BaseModel):
    name: str
    city: str
    neighborhood: Optional[str] = None
    budget_level: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None


class RestaurantSuggestion(BaseModel):
    name: str
    city: str
    neighborhood: Optional[str] = None
    cuisine: Optional[str] = None
    price_range: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None


class ItineraryResponse(BaseModel):
    destination: str
    duration: str
    traveler_type: Optional[str] = None
    itinerary: List[DayPlan]
    hotel_suggestions: List[HotelSuggestion] = Field(default_factory=list)
    restaurant_suggestions: List[RestaurantSuggestion] = Field(default_factory=list)
    alternatives: List[AlternativePlan] = Field(default_factory=list)
    logistics_notes: str
    assumptions: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)


class QuestionResponse(BaseModel):
    answer: str
    supporting_points: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)


class ClarificationResponse(BaseModel):
    message: str
    missing_fields: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)


class CollectedInfoFlags(BaseModel):
    has_destination: bool = False
    has_duration: bool = False
    has_dates: bool = False
    has_interests: bool = False
    has_budget: bool = False
    is_complete_for_planning: bool = False


class TavilyResult(BaseModel):
    title: str
    url: str
    content_snippet: str
    source_type: str = "web"


class TavilySearchOutput(BaseModel):
    results: List[TavilyResult] = Field(default_factory=list)
    query: str
    tool_status: Literal["ok", "error"]
    error_message: Optional[str] = None


class TravelAgentState(BaseModel):
    """Full LangGraph agent state. Serialized as dict when passed to graph."""
    user_profile: Optional[dict] = None
    trip_request: Optional[dict] = None
    travel_question: Optional[dict] = None
    messages: List[dict] = Field(default_factory=list)
    intent: str = "unknown"
    collected_info: dict = Field(default_factory=dict)
    tavily_context: List[dict] = Field(default_factory=list)
    draft: Optional[str] = None
    final_response: Optional[dict] = None
    tool_call_count: int = 0
    needs_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    error: Optional[str] = None
