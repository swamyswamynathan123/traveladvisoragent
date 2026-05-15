from __future__ import annotations
import json
import logging
from typing import Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from schemas import ItineraryResponse, QuestionResponse, ClarificationResponse
from tools import tavily_search, tripadvisor_search
from prompts import (
    SYSTEM_PROMPT,
    INTENT_DETECTION_PROMPT,
    CLARIFICATION_PROMPT,
    ITINERARY_GENERATION_PROMPT,
    QUESTION_ANSWER_PROMPT,
    SCHEMA_REPAIR_PROMPT,
)

logger = logging.getLogger(__name__)
MAX_TOOL_CALLS = 6


# ── state factory ─────────────────────────────────────────────────────────────

def build_initial_state() -> dict:
    return {
        "user_profile": None,
        "trip_request": None,
        "travel_question": None,
        "messages": [],
        "intent": "unknown",
        "collected_info": {},
        "tavily_context": [],
        "draft": None,
        "final_response": None,
        "tool_call_count": 0,
        "needs_clarification": False,
        "clarification_questions": [],
        "error": None,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def _format_tavily_context(tavily_context: list) -> str:
    if not tavily_context:
        return "No search results available. Use general knowledge and mark claims accordingly."
    lines: list[str] = []
    for i, item in enumerate(tavily_context[:10], start=1):
        lines.append(f"\n[{i}] {item.get('title', 'Unknown source')}")
        lines.append(f"URL: {item.get('url', '')}")
        lines.append(f"Snippet: {item.get('content_snippet', '')}")
    return "\n".join(lines)


# ── nodes ─────────────────────────────────────────────────────────────────────

def collect_requirements_node(state: dict) -> dict:
    """Parse the latest user message to extract intent and request details."""
    messages = state.get("messages", [])
    if not messages:
        return state

    latest = messages[-1].get("content", "")
    prompt = INTENT_DETECTION_PROMPT.format(user_message=latest)

    try:
        response = _llm().invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        parsed: dict = json.loads(response.content)

        intent = parsed.get("intent", "unknown")
        trip_req = parsed.get("trip_request") or {}
        travel_q = parsed.get("travel_question")
        clarification_needed: list = parsed.get("clarification_needed", [])

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

        # If planning intent is incomplete and the message looks like a question,
        # treat it as question intent to avoid spurious clarification requests.
        is_planning_complete = collected_info.get("is_complete_for_planning", False)
        question_words = ("what", "when", "where", "how", "which", "why", "is ", "are ",
                          "do ", "can ", "should ", "would ", "?")
        message_is_question = any(latest.lower().startswith(w) or latest.endswith("?")
                                  for w in question_words)
        if intent == "planning" and not is_planning_complete and message_is_question:
            intent = "question"
            travel_q = travel_q or latest

        # Ensure travel_question is never null when intent is "question"
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

    except (json.JSONDecodeError, Exception) as exc:
        logger.error("collect_requirements_node failed: %s", exc)
        return {**state, "error": str(exc)}


def validate_inputs_node(state: dict) -> dict:
    """Set needs_clarification based on intent and collected info completeness."""
    intent = state.get("intent", "unknown")

    if intent == "planning":
        is_complete = state.get("collected_info", {}).get("is_complete_for_planning", False)
        needs = not is_complete
    elif intent == "question":
        q_data = state.get("travel_question") or {}
        needs = not bool(q_data.get("question"))
    else:
        needs = True

    return {**state, "needs_clarification": needs}


def should_clarify(state: dict) -> str:
    """Conditional edge: route to clarification or search."""
    return "ask_clarification" if state.get("needs_clarification") else "search_with_tavily"


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
        response = _llm(temperature=0.3).invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        data: dict = json.loads(response.content)
    except Exception as exc:
        logger.error("ask_clarification_node failed: %s", exc)
        data = {
            "message": "I'd love to help plan your trip! Could you tell me where you'd like to go and for how long?",
            "missing_fields": missing,
            "open_questions": [
                "What is your destination?",
                "How many days will you be traveling?",
            ],
        }

    return {
        **state,
        "final_response": {"type": "clarification", "data": data},
        "draft": data.get("message", ""),
    }


def search_with_tavily_node(state: dict) -> dict:
    """Run Tavily searches to gather travel information (max MAX_TOOL_CALLS total)."""
    intent = state.get("intent", "unknown")
    call_count: int = state.get("tool_call_count", 0)

    if call_count >= MAX_TOOL_CALLS:
        logger.warning("MAX_TOOL_CALLS reached — skipping Tavily search")
        return state

    import re

    # Each entry is (query, search_fn) — TripAdvisor for restaurants/attractions, Tavily for everything else
    search_tasks: list[tuple[str, object]] = []

    if intent == "planning":
        trip_req = state.get("trip_request") or {}
        dest = trip_req.get("destination", "")
        interests = trip_req.get("interests", [])
        constraints = [str(c).lower() for c in trip_req.get("constraints", [])]
        constraint_text = " ".join(constraints)

        search_tasks += [
            (f"{dest} top attractions things to do", tripadvisor_search),
            (f"{dest} travel tips transportation budget", tavily_search),
        ]
        if interests:
            interests_str = " ".join(str(i) for i in interests[:2])
            search_tasks.append((f"{dest} {interests_str} recommendations", tripadvisor_search))

        dietary_keywords = ["vegetarian", "vegan", "halal", "kosher", "gluten", "dairy-free"]
        has_dietary = any(kw in constraint_text for kw in dietary_keywords)
        has_food_interest = any("food" in str(i).lower() or "dining" in str(i).lower() for i in interests)
        if has_dietary or has_food_interest:
            city_names = [c.strip() for c in re.split(r"[,+&]|\band\b", dest, flags=re.IGNORECASE) if c.strip()]
            search_cities = city_names[:2] if len(city_names) > 1 else [dest]
            if has_dietary:
                diet_type = next((kw for kw in dietary_keywords if kw in constraint_text), "special diet")
                for city in search_cities:
                    search_tasks.append((f"{city} best {diet_type} restaurants", tripadvisor_search))
            else:
                for city in search_cities:
                    search_tasks.append((f"{city} best restaurants must-try food", tripadvisor_search))

    elif intent == "question":
        q = (state.get("travel_question") or {}).get("question", "")
        trip_req = state.get("trip_request") or {}
        dest = trip_req.get("destination", "")
        start_date = trip_req.get("start_date", "")
        enriched_q = f"{q} {dest}".strip() if dest and dest.lower() not in q.lower() else q

        restaurant_kw = ["restaurant", "eat", "food", "dining", "vegetarian", "vegan",
                         "halal", "kosher", "cafe", "cuisine", "where to eat"]
        is_restaurant_q = any(kw in q.lower() for kw in restaurant_kw)
        search_fn = tripadvisor_search if is_restaurant_q else tavily_search
        search_tasks.append((enriched_q, search_fn))

        if dest and is_restaurant_q:
            city_names = [c.strip() for c in re.split(r"[,+&]|\band\b", dest, flags=re.IGNORECASE) if c.strip()]
            for city in city_names[:2]:
                if city.lower() not in q.lower():
                    search_tasks.append((f"{q} {city}", tripadvisor_search))
        elif start_date and dest:
            search_tasks.append((f"{dest} weather {start_date}", tavily_search))

    else:
        return state

    budget = MAX_TOOL_CALLS - call_count
    search_tasks = search_tasks[:budget]

    accumulated: list = list(state.get("tavily_context", []))
    new_call_count = call_count

    for query, search_fn in search_tasks:
        result = search_fn(query=query, max_results=5)
        new_call_count += 1
        if result.tool_status == "ok":
            for r in result.results:
                accumulated.append({
                    "title": r.title,
                    "url": r.url,
                    "content_snippet": r.content_snippet,
                    "source_type": r.source_type,
                    "query": query,
                })

    return {**state, "tavily_context": accumulated, "tool_call_count": new_call_count}


def generate_response_node(state: dict) -> dict:
    """Generate a structured LLM response with one retry on schema failure."""
    intent = state.get("intent", "unknown")
    if intent == "planning":
        return _generate_itinerary(state)
    elif intent == "question":
        return _generate_answer(state)
    else:
        fallback = ClarificationResponse(
            message="I'm not sure what you're asking. Could you rephrase your request?",
            missing_fields=[],
            open_questions=["What would you like help with?"],
        )
        return {**state, "final_response": {"type": "clarification", "data": fallback.model_dump()}}


def _generate_itinerary(state: dict) -> dict:
    llm = _llm(model="gpt-4o", temperature=0.3)
    schema_str = json.dumps(ItineraryResponse.model_json_schema(), indent=2)
    prompt = ITINERARY_GENERATION_PROMPT.format(
        trip_request=json.dumps(state.get("trip_request") or {}, indent=2),
        user_profile=json.dumps(state.get("user_profile") or {}, indent=2),
        tavily_context=_format_tavily_context(state.get("tavily_context", [])),
        schema=schema_str,
    )
    try:
        structured_llm = llm.with_structured_output(ItineraryResponse)
        response: ItineraryResponse = structured_llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return {
            **state,
            "draft": response.model_dump_json(),
            "final_response": {"type": "itinerary", "data": response.model_dump()},
        }
    except Exception as exc:
        logger.warning("Itinerary generation failed (%s), retrying with repair", exc)
        return _retry_with_repair(llm, state, "itinerary", ItineraryResponse, schema_str)


def _generate_answer(state: dict) -> dict:
    llm = _llm(model="gpt-4o", temperature=0.3)
    schema_str = json.dumps(QuestionResponse.model_json_schema(), indent=2)
    q_data = state.get("travel_question") or {}
    trip_req = state.get("trip_request") or {}
    trip_context_parts = []
    if trip_req.get("destination"):
        trip_context_parts.append(f"Destination: {trip_req['destination']}")
    if trip_req.get("start_date"):
        trip_context_parts.append(f"Travel date: {trip_req['start_date']}")
    elif trip_req.get("duration_days"):
        trip_context_parts.append(f"Duration: {trip_req['duration_days']} days")
    if trip_req.get("constraints"):
        trip_context_parts.append(f"Constraints: {', '.join(str(c) for c in trip_req['constraints'])}")
    trip_context = "; ".join(trip_context_parts) if trip_context_parts else q_data.get("context") or "No additional context."
    prompt = QUESTION_ANSWER_PROMPT.format(
        question=q_data.get("question", ""),
        context=trip_context,
        tavily_context=_format_tavily_context(state.get("tavily_context", [])),
        schema=schema_str,
    )
    try:
        structured_llm = llm.with_structured_output(QuestionResponse)
        response: QuestionResponse = structured_llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return {
            **state,
            "draft": response.model_dump_json(),
            "final_response": {"type": "answer", "data": response.model_dump()},
        }
    except Exception as exc:
        logger.warning("Answer generation failed (%s), retrying with repair", exc)
        return _retry_with_repair(llm, state, "answer", QuestionResponse, schema_str)


def _retry_with_repair(llm, state: dict, resp_type: str, schema_cls, schema_str: str) -> dict:
    """Retry structured output generation with a repair prompt."""
    repair_prompt = SCHEMA_REPAIR_PROMPT.format(
        original_response=state.get("draft", ""),
        schema=schema_str,
    )
    try:
        structured_llm = llm.with_structured_output(schema_cls)
        response = structured_llm.invoke([HumanMessage(content=repair_prompt)])
        return {
            **state,
            "final_response": {"type": resp_type, "data": response.model_dump()},
        }
    except Exception as exc2:
        logger.error("Repair also failed: %s", exc2)
        fallback = ClarificationResponse(
            message="I had trouble generating a complete response. Please rephrase your request.",
            missing_fields=[],
            open_questions=["Could you provide more detail about your travel plans?"],
        )
        return {
            **state,
            "final_response": {"type": "clarification", "data": fallback.model_dump()},
            "error": str(exc2),
        }


def respond_to_user_node(state: dict) -> dict:
    """Append the assistant message to chat history."""
    final_response = state.get("final_response")
    if not final_response:
        return state

    resp_type = final_response.get("type", "")
    data = final_response.get("data", {})

    if resp_type == "clarification":
        text = data.get("message", "I need more information.")
    elif resp_type == "itinerary":
        dest = data.get("destination", "")
        dur = data.get("duration", "")
        text = f"Here is your {dur} itinerary for {dest}!"
    elif resp_type == "answer":
        text = data.get("answer", "Here is what I found.")
    else:
        text = "Here is your travel information."

    messages = list(state.get("messages", []))
    messages.append({"role": "assistant", "content": text})
    return {**state, "messages": messages}


# ── graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    """Compile and return the LangGraph travel advisor graph."""
    workflow = StateGraph(dict)

    workflow.add_node("collect_requirements", collect_requirements_node)
    workflow.add_node("validate_inputs", validate_inputs_node)
    workflow.add_node("ask_clarification", ask_clarification_node)
    workflow.add_node("search_with_tavily", search_with_tavily_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("respond_to_user", respond_to_user_node)

    workflow.set_entry_point("collect_requirements")
    workflow.add_edge("collect_requirements", "validate_inputs")
    workflow.add_conditional_edges(
        "validate_inputs",
        should_clarify,
        {
            "ask_clarification": "ask_clarification",
            "search_with_tavily": "search_with_tavily",
        },
    )
    workflow.add_edge("ask_clarification", END)
    workflow.add_edge("search_with_tavily", "generate_response")
    workflow.add_edge("generate_response", "respond_to_user")
    workflow.add_edge("respond_to_user", END)

    return workflow.compile()
