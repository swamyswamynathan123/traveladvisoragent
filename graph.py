from __future__ import annotations
import json
import logging
from typing import Optional

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from schemas import ItineraryResponse, QuestionResponse, ClarificationResponse
from tools import tavily_search, tavily_search_advanced, tripadvisor_search, expedia_hotel_search
from prompts import (
    SYSTEM_PROMPT,
    INTENT_DETECTION_PROMPT,
    CLARIFICATION_PROMPT,
    ITINERARY_GENERATION_PROMPT,
    QUESTION_ANSWER_PROMPT,
    SCHEMA_REPAIR_PROMPT,
    PERSONALIZATION_CHECK_PROMPT,
)

logger = logging.getLogger(__name__)
MAX_TOOL_CALLS = 9

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


def _safe_format_str(value: str) -> str:
    """Escape curly braces in user-supplied strings before prompt .format() interpolation."""
    return str(value).replace("{", "{{").replace("}", "}}")


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
        "itinerary_response": None,   # persists across Q&A queries
        "tool_call_count": 0,
        "needs_clarification": False,
        "clarification_questions": [],
        "error": None,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _llm(model: str = "gpt-4o-mini", temperature: float = 0.0, streaming: bool = False):
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)
    return ChatOpenAI(model=model, temperature=temperature, streaming=streaming)


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


# ── nodes ─────────────────────────────────────────────────────────────────────

def collect_requirements_node(state: dict) -> dict:
    """Parse the latest user message to extract intent and request details."""
    from schemas import IntentDetectionOutput
    messages = state.get("messages", [])
    if not messages:
        return state

    latest = messages[-1].get("content", "")
    prompt = INTENT_DETECTION_PROMPT.format(user_message=_safe_format_str(latest))

    try:
        structured_llm = _llm().with_structured_output(IntentDetectionOutput, method="function_calling")
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
        structured_llm = _llm(temperature=0.3).with_structured_output(ClarificationResponse, method="function_calling")
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


def search_with_tavily_node(state: dict) -> dict:
    """Run searches in parallel to gather travel information (max MAX_TOOL_CALLS total)."""
    from concurrent.futures import ThreadPoolExecutor
    import re

    intent = state.get("intent", "unknown")
    call_count: int = state.get("tool_call_count", 0)

    if call_count >= MAX_TOOL_CALLS:
        logger.warning("MAX_TOOL_CALLS reached — skipping search")
        return state

    # Each entry is (fn, kwargs)
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
            (tripadvisor_search, {"query": f"{dest} top attractions things to do", "max_results": 5}),
            (tavily_search, {"query": f"{dest} travel tips transportation budget", "max_results": 5}),
        ]
        if interests:
            interests_str = " ".join(str(i) for i in interests[:2])
            search_tasks.append(
                (tripadvisor_search, {"query": f"{dest} {interests_str} recommendations", "max_results": 5})
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
                ))

        city_names = [c.strip() for c in re.split(r"[,+&]|\band\b", dest, flags=re.IGNORECASE) if c.strip()]
        hotel_cities = city_names if city_names else [dest]
        for city in hotel_cities[:3]:
            search_tasks.append((
                tripadvisor_search,
                {"query": f"best {budget_level.replace('_', ' ')} hotels {city} recommended 2024", "max_results": 5},
            ))

        if start_date:
            for city in hotel_cities[:2]:
                search_tasks.append((
                    expedia_hotel_search,
                    {"destination": city, "check_in": start_date, "budget_level": budget_level},
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
                    ))
            else:
                enriched_q = f"{food_label} restaurants {dest or q}".strip()
                search_tasks.append((tavily_search_advanced, {"query": enriched_q, "max_results": 5}))
        else:
            enriched_q = f"{q} {dest}".strip() if dest and dest.lower() not in q.lower() else q
            search_tasks.append((tavily_search, {"query": enriched_q, "max_results": 5}))
            if start_date and dest:
                search_tasks.append((tavily_search, {"query": f"{dest} weather {start_date}", "max_results": 5}))
    else:
        return state

    # Cap to remaining budget
    search_tasks = search_tasks[:MAX_TOOL_CALLS - call_count]

    if not search_tasks:
        return state

    def _run_task(task):
        fn, kwargs = task[:2]
        try:
            return fn(**kwargs)
        except Exception as exc:
            logger.warning("Search task %s failed: %s", fn.__name__, exc)
            from schemas import TavilySearchOutput
            return TavilySearchOutput(results=[], query=str(kwargs), tool_status="error")

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
    llm = _llm(model="gpt-4o", temperature=0.3, streaming=True)
    schema_str = json.dumps(ItineraryResponse.model_json_schema(), indent=2)
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
        interests_list=_safe_format_str(", ".join(str(i) for i in interests) if interests else "general sightseeing"),
        constraints_list=_safe_format_str(", ".join(str(c) for c in constraints) if constraints else "none"),
        budget_level=budget_level,
        pace=pace,
        schema=schema_str,
    )
    try:
        structured_llm = llm.with_structured_output(ItineraryResponse, method="function_calling")
        response: ItineraryResponse = structured_llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        itinerary = {"type": "itinerary", "data": response.model_dump()}
        return {
            **state,
            "draft": response.model_dump_json(),
            "final_response": itinerary,
            "itinerary_response": itinerary,
        }
    except Exception as exc:
        logger.warning("Itinerary generation failed (%s), retrying with repair", exc)
        return _retry_with_repair(llm, state, "itinerary", ItineraryResponse, schema_str)


def _generate_answer(state: dict) -> dict:
    llm = _llm(model="gpt-4o", temperature=0.3, streaming=True)
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
        structured_llm = llm.with_structured_output(QuestionResponse, method="function_calling")
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
        structured_llm = llm.with_structured_output(schema_cls, method="function_calling")
        response = structured_llm.invoke([HumanMessage(content=repair_prompt)])
        response_payload = {"type": resp_type, "data": response.model_dump()}
        extra = {"itinerary_response": response_payload} if resp_type == "itinerary" else {}
        return {**state, "final_response": response_payload, **extra}
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
        interests_list=_safe_format_str(", ".join(str(i) for i in interests) if interests else "general sightseeing"),
        budget_level=budget_level,
        budget_guidance=BUDGET_GUIDANCE.get(budget_level, BUDGET_GUIDANCE["mid_range"]),
        pace=pace,
        pace_description=PACE_DESCRIPTIONS.get(pace, PACE_DESCRIPTIONS["moderate"]),
        constraints_list=_safe_format_str(", ".join(str(c) for c in constraints) if constraints else "none"),
        itinerary=json.dumps(itinerary_response.get("data", {}), indent=2),
        schema=schema_str,
    )

    try:
        llm = _llm(model="gpt-4o-mini", temperature=0.0)
        structured_llm = llm.with_structured_output(ItineraryResponse, method="function_calling")
        patched: ItineraryResponse = structured_llm.invoke([HumanMessage(content=prompt)])
        patched_data = patched.model_dump()
        if patched_data.get("budget_estimate") is None:
            patched_data["budget_estimate"] = itinerary_response.get("data", {}).get("budget_estimate")
        patched_response = {"type": "itinerary", "data": patched_data}
        return {
            **state,
            "final_response": patched_response,
            "itinerary_response": patched_response,
        }
    except Exception as exc:
        logger.warning("personalization_check_node failed (%s), using original", exc)
        return state


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
    workflow.add_node("personalization_check", personalization_check_node)
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
    workflow.add_conditional_edges(
        "generate_response",
        should_check_personalization,
        {
            "personalization_check": "personalization_check",
            "respond_to_user": "respond_to_user",
        },
    )
    workflow.add_edge("personalization_check", "respond_to_user")
    workflow.add_edge("respond_to_user", END)

    return workflow.compile()
