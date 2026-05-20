from __future__ import annotations
import json
import logging

import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

from graph import build_graph, build_initial_state

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Travel Advisor Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state bootstrap ───────────────────────────────────────────────────

if "agent_state" not in st.session_state:
    st.session_state.agent_state = build_initial_state()
if "compiled_graph" not in st.session_state:
    st.session_state.compiled_graph = build_graph()
if "show_refine_input" not in st.session_state:
    st.session_state.show_refine_input = False


# ── rendering helpers ─────────────────────────────────────────────────────────

def _render_itinerary(data: dict) -> None:
    dest = data.get("destination", "")
    dur = data.get("duration", "")
    ttype = data.get("traveler_type", "")
    header = f"🗺️ {dur} Itinerary for {dest}"
    if ttype:
        header += f" ({ttype})"
    st.subheader(header)

    for day in data.get("itinerary", []):
        day_num = day.get("day_number", "?")
        theme = day.get("theme", "")
        label = f"Day {day_num}" + (f" — {theme}" if theme else "")
        with st.expander(label, expanded=(day_num == 1)):
            time_icons = {"morning": "🌅", "afternoon": "☀️", "evening": "🌙"}
            for block in day.get("blocks", []):
                tod = block.get("time_of_day", "")
                icon = time_icons.get(tod, "⏰")
                st.markdown(f"**{icon} {tod.capitalize()}**")
                st.markdown(f"📍 {block.get('activity', '')}")
                if block.get("location"):
                    st.caption(f"Location: {block['location']}")
                if block.get("notes"):
                    cleaned_notes = (
                        block["notes"]
                        .replace("[General knowledge — verify before traveling]", "")
                        .replace("[General knowledge]", "")
                        .strip()
                    )
                    if cleaned_notes:
                        st.info(cleaned_notes)
                if block.get("duration_hours"):
                    st.caption(f"~{block['duration_hours']} hr")
                st.divider()
            if day.get("tips"):
                st.success(f"💡 Tip: {day['tips']}")

    hotels = data.get("hotel_suggestions", [])
    if hotels:
        st.subheader("🏨 Hotel Suggestions")
        by_city: dict = {}
        for h in hotels:
            by_city.setdefault(h.get("city", "General"), []).append(h)
        for city, city_hotels in by_city.items():
            st.markdown(f"**{city}**")
            cols = st.columns(min(len(city_hotels), 3))
            for col, hotel in zip(cols, city_hotels):
                with col:
                    budget_icons = {"budget": "💰", "mid_range": "💰💰", "luxury": "💰💰💰"}
                    budget = hotel.get("budget_level", "")
                    icon = budget_icons.get(budget, "")
                    st.markdown(f"**{hotel.get('name', '')}** {icon}")
                    if hotel.get("neighborhood"):
                        st.caption(f"📍 {hotel['neighborhood']}")
                    if hotel.get("description"):
                        st.write(hotel["description"])
                    if hotel.get("notes"):
                        st.caption(hotel["notes"])

    restaurants = data.get("restaurant_suggestions", [])
    if restaurants:
        st.subheader("🍽️ Restaurant Suggestions")
        by_city: dict = {}
        for r in restaurants:
            by_city.setdefault(r.get("city", "General"), []).append(r)
        for city, city_restaurants in by_city.items():
            st.markdown(f"**{city}**")
            cols = st.columns(min(len(city_restaurants), 3))
            for col, restaurant in zip(cols, city_restaurants):
                with col:
                    price = restaurant.get("price_range", "")
                    cuisine = restaurant.get("cuisine", "")
                    header = restaurant.get("name", "")
                    if price:
                        header += f" {price}"
                    st.markdown(f"**{header}**")
                    if cuisine:
                        st.caption(f"🍴 {cuisine}")
                    if restaurant.get("neighborhood"):
                        st.caption(f"📍 {restaurant['neighborhood']}")
                    if restaurant.get("description"):
                        st.write(restaurant["description"])
                    if restaurant.get("notes"):
                        st.caption(restaurant["notes"])

    alts = data.get("alternatives", [])
    if alts:
        st.subheader("🔄 Alternatives")
        for alt in alts:
            with st.expander(f"Alternative: {alt.get('name', '')}"):
                st.write(alt.get("description", ""))
                for act in alt.get("activities", []):
                    st.markdown(f"- {act}")

    if data.get("logistics_notes"):
        st.subheader("🚗 Logistics Notes")
        st.markdown(data["logistics_notes"])

    col1, col2 = st.columns(2)
    with col1:
        assumptions = data.get("assumptions", [])
        if assumptions:
            st.subheader("📝 Assumptions")
            for a in assumptions:
                st.markdown(f"- {a}")
    with col2:
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

    sources = data.get("sources", [])
    if sources:
        st.subheader("📚 Sources")
        for s in sources:
            title = s.get("title", "Source")
            url = s.get("url", "#")
            snippet = s.get("snippet", "")
            st.markdown(f"[{title}]({url})")
            if snippet:
                st.caption(snippet)

    _render_map(data)


def _render_map(data: dict) -> None:
    _DAY_COLORS = [
        [255, 87, 51], [52, 152, 219], [39, 174, 96], [155, 89, 182],
        [243, 156, 18], [26, 188, 156], [231, 76, 60], [52, 73, 94],
    ]
    points = []
    for day in data.get("itinerary", []):
        day_num = day.get("day_number", 1)
        color = _DAY_COLORS[(day_num - 1) % len(_DAY_COLORS)]
        for block in day.get("blocks", []):
            lat = block.get("latitude")
            lon = block.get("longitude")
            if lat and lon:
                points.append({
                    "lat": lat, "lon": lon,
                    "label": f"Day {day_num}: {block.get('activity', '')}",
                    "color": color,
                })
    if not points:
        return
    df = pd.DataFrame(points)
    lat_range = df["lat"].max() - df["lat"].min()
    lon_range = df["lon"].max() - df["lon"].min()
    spread = max(lat_range, lon_range)
    zoom = 5 if spread > 5 else 7 if spread > 2 else 11 if spread > 0.3 else 13
    st.subheader("🗺️ Map")
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=300,
            pickable=True,
        )],
        initial_view_state=pdk.ViewState(
            latitude=df["lat"].mean(),
            longitude=df["lon"].mean(),
            zoom=zoom,
        ),
        tooltip={"text": "{label}"},
    ))


def _render_answer(data: dict) -> None:
    st.subheader("💬 Travel Answer")
    st.write(data.get("answer", ""))

    pts = data.get("supporting_points", [])
    if pts:
        st.subheader("📋 Key Points")
        for p in pts:
            st.markdown(f"- {p}")

    assumptions = data.get("assumptions", [])
    if assumptions:
        with st.expander("📝 Assumptions & Caveats"):
            for a in assumptions:
                st.markdown(f"- {a}")

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

    sources = data.get("sources", [])
    if sources:
        st.subheader("📚 Sources")
        for s in sources:
            st.markdown(f"[{s.get('title', 'Source')}]({s.get('url', '#')})")
            if s.get("snippet"):
                st.caption(s["snippet"])


def _render_clarification(data: dict) -> None:
    st.info(data.get("message", ""))
    missing = data.get("missing_fields", [])
    if missing:
        st.write("**Missing information:**", ", ".join(missing))
    for q in data.get("open_questions", []):
        st.markdown(f"- {q}")


def _run_graph(state: dict) -> dict:
    _NODE_LABELS = {
        "collect_requirements": "🧠 Understanding your request...",
        "validate_inputs": "✅ Validating inputs...",
        "ask_clarification": "💬 Preparing clarification...",
        "search_with_tavily": "🔍 Searching travel information...",
        "generate_response": "✍️ Generating response...",
        "respond_to_user": "📝 Wrapping up...",
    }
    result = state
    with st.status("Working on it...", expanded=True) as status:
        for chunk in st.session_state.compiled_graph.stream(state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                st.write(_NODE_LABELS.get(node_name, f"Running {node_name}..."))
                result = node_output
        status.update(label="Done!", state="complete", expanded=False)
    return result


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("✈️ Travel Advisor")
    st.markdown("Plan trips and answer travel questions using AI + real-time web search.")
    st.divider()

    st.subheader("Trip Details")

    destination = st.text_input(
        "Destination(s) *",
        placeholder="e.g., Paris  |  Tokyo + Kyoto",
        help="Required for itinerary generation.",
    )
    col_d, col_s = st.columns(2)
    with col_d:
        duration_days = st.number_input("Duration (days)", min_value=1, max_value=60, value=5, step=1)
    with col_s:
        start_date = st.date_input("Start Date (opt.)")

    origin = st.text_input("Departing From (opt.)", placeholder="e.g., London")
    returning_to = st.text_input("Returning To (opt.)", placeholder="e.g., London")

    interests = st.multiselect(
        "Interests",
        [
            "Culture & History",
            "Food & Dining",
            "Nature & Outdoors",
            "Adventure Sports",
            "Art & Museums",
            "Shopping",
            "Nightlife",
            "Family-Friendly",
            "Wellness & Spa",
        ],
    )

    budget_level = st.select_slider(
        "Budget Level",
        options=["budget", "mid_range", "luxury"],
        value="mid_range",
    )
    pace = st.select_slider(
        "Travel Pace",
        options=["relaxed", "moderate", "packed"],
        value="moderate",
    )
    traveler_type = st.selectbox(
        "Traveler Type",
        ["solo", "couple", "family", "group"],
    )
    constraints_raw = st.text_area(
        "Special Constraints (opt.)",
        placeholder="e.g., wheelchair accessible, vegetarian only, no flying",
    )

    st.divider()

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        generate_clicked = st.button("🗓️ Generate Itinerary", type="primary", use_container_width=True)
    with btn_col2:
        refine_clicked = st.button("✏️ Refine Plan", use_container_width=True)

    ask_q_clicked = st.button("❓ Ask a Travel Question", use_container_width=True)

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.agent_state = build_initial_state()
        st.session_state.show_refine_input = False
        st.rerun()

    # ── cost counter ──────────────────────────────────────────────────────────
    st.divider()
    call_count = st.session_state.agent_state.get("tool_call_count", 0)
    max_calls = 9
    st.caption(f"Searches used: {call_count} / {max_calls}")
    st.progress(min(call_count / max_calls, 1.0))
    if call_count > 0:
        est = call_count * 0.001 + 0.06  # ~$0.001/Tavily + ~$0.06 GPT-4o
        st.caption(f"Est. cost this session: ~${est:.2f}")

    # ── save / load ───────────────────────────────────────────────────────────
    st.divider()
    _itinerary = st.session_state.agent_state.get("itinerary_response")
    if _itinerary:
        _dest = _itinerary.get("data", {}).get("destination", "itinerary")
        _fname = f"{_dest.replace(' ', '_').lower()}_itinerary.json"
        st.download_button(
            "💾 Save Itinerary",
            data=json.dumps(_itinerary, indent=2),
            file_name=_fname,
            mime="application/json",
            use_container_width=True,
        )
    _uploaded = st.file_uploader("📂 Load saved itinerary", type="json", label_visibility="collapsed")
    if _uploaded:
        _loaded = json.loads(_uploaded.read())
        st.session_state.agent_state["final_response"] = _loaded
        st.session_state.agent_state["itinerary_response"] = _loaded
        st.rerun()

# ── generate itinerary ────────────────────────────────────────────────────────

if generate_clicked:
    if not destination.strip():
        st.sidebar.error("Please enter a destination.")
    else:
        constraints = [c.strip() for c in constraints_raw.split(",") if c.strip()]
        normalized_interests = [
            i.lower().replace(" & ", "_").replace(" ", "_") for i in interests
        ]

        new_state = build_initial_state()
        new_state["user_profile"] = {"traveler_type": traveler_type}
        new_state["intent"] = "planning"
        new_state["trip_request"] = {
            "destination": destination.strip(),
            "duration_days": int(duration_days),
            "start_date": str(start_date) if start_date else None,
            "origin": origin.strip() or None,
            "returning_to": returning_to.strip() or None,
            "interests": normalized_interests,
            "budget_level": budget_level,
            "pace": pace,
            "constraints": constraints,
        }
        new_state["collected_info"] = {
            "has_destination": True,
            "has_duration": True,
            "has_dates": bool(start_date),
            "has_interests": bool(interests),
            "has_budget": True,
            "is_complete_for_planning": True,
        }
        user_msg = (
            f"Plan a {duration_days}-day trip to {destination.strip()}. "
            f"Traveler type: {traveler_type}. Budget: {budget_level}. Pace: {pace}. "
            f"Interests: {', '.join(interests) or 'general sightseeing'}. "
            f"Constraints: {', '.join(constraints) or 'none'}."
        )
        if origin:
            user_msg += f" Departing from {origin}."
        if returning_to:
            user_msg += f" Returning to {returning_to}."
        if start_date:
            user_msg += f" Start date: {start_date}."
        new_state["messages"] = [{"role": "user", "content": user_msg}]

        result_state = _run_graph(new_state)
        st.session_state.agent_state = result_state
        st.rerun()

# ── refine plan ───────────────────────────────────────────────────────────────

if refine_clicked:
    st.session_state.show_refine_input = True

if st.session_state.show_refine_input:
    refine_text = st.sidebar.text_area(
        "What would you like to change?",
        placeholder="e.g., Add more food experiences on day 2",
    )
    if st.sidebar.button("Submit Refinement", type="primary"):
        if refine_text.strip():
            state = dict(st.session_state.agent_state)
            msgs = list(state.get("messages", []))
            msgs.append({"role": "user", "content": refine_text.strip()})
            state["messages"] = msgs
            state["tavily_context"] = []
            state["tool_call_count"] = 0
            state["final_response"] = None
            state["needs_clarification"] = False
            result_state = _run_graph(state)
            st.session_state.agent_state = result_state
            st.session_state.show_refine_input = False
            st.rerun()

# ── main area ─────────────────────────────────────────────────────────────────

st.title("Travel Advisor Agent ✈️")
st.caption("Powered by OpenAI + Tavily real-time web search + LangGraph")

agent_state = st.session_state.agent_state
messages = agent_state.get("messages", [])
final_response = agent_state.get("final_response")
itinerary_response = agent_state.get("itinerary_response")

# Chat history
for msg in messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.write(content)

# Always show the itinerary if one has been generated
if itinerary_response:
    st.divider()
    _render_itinerary(itinerary_response.get("data", {}))

# Show the latest Q&A answer or clarification below the itinerary
if final_response and final_response.get("type") != "itinerary":
    resp_type = final_response.get("type")
    data = final_response.get("data", {})
    st.divider()
    if resp_type == "answer":
        _render_answer(data)
    elif resp_type == "clarification":
        _render_clarification(data)

# Chat input for follow-up Q&A
if prompt := st.chat_input("Ask a follow-up or travel question..."):
    state = dict(st.session_state.agent_state)
    msgs = list(state.get("messages", []))
    msgs.append({"role": "user", "content": prompt})
    state["messages"] = msgs
    state["intent"] = "unknown"  # re-classify each new chat message from scratch
    state["tavily_context"] = []
    state["tool_call_count"] = 0
    state["final_response"] = None
    state["needs_clarification"] = False
    result_state = _run_graph(state)
    st.session_state.agent_state = result_state
    st.rerun()
