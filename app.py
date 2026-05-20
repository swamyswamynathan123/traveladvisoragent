from __future__ import annotations
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graph import build_graph, build_initial_state
from prompts import PACKING_LIST_PROMPT
from schemas import PackingListResponse

_HISTORY_FILE = Path(__file__).parent / "itineraries" / "history.json"

_ALL_INTERESTS = [
    "Culture & History", "Food & Dining", "Nature & Outdoors",
    "Adventure Sports", "Art & Museums", "Shopping",
    "Nightlife", "Family-Friendly", "Wellness & Spa",
]
# normalized key → display label (used when restoring saved trip data)
_NORM_TO_DISPLAY = {
    i.lower().replace(" & ", "_").replace(" ", "_"): i for i in _ALL_INTERESTS
}


def _load_history() -> list:
    if not _HISTORY_FILE.exists():
        return []
    try:
        return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_to_history(itinerary_response: dict, trip_request: dict | None = None) -> None:
    _HISTORY_FILE.parent.mkdir(exist_ok=True)
    dest = itinerary_response.get("data", {}).get("destination", "Unknown")
    entry = {
        "destination": dest,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "itinerary": itinerary_response,
        "trip_request": trip_request or {},
    }
    history = _load_history()
    history = [entry] + [h for h in history if h.get("destination") != dest]
    _HISTORY_FILE.write_text(json.dumps(history[:10], indent=2), encoding="utf-8")


def _prefill_sidebar(trip_req: dict) -> None:
    """Write trip_request values into sidebar widget session-state keys so they render pre-filled."""
    from datetime import date as _date
    if trip_req.get("destination"):
        st.session_state["sb_destination"] = trip_req["destination"]
    if trip_req.get("duration_days"):
        st.session_state["sb_duration_days"] = int(trip_req["duration_days"])
    sd = trip_req.get("start_date")
    if sd:
        try:
            st.session_state["sb_start_date"] = _date.fromisoformat(sd)
        except (ValueError, TypeError):
            pass
    st.session_state["sb_origin"] = trip_req.get("origin") or ""
    st.session_state["sb_returning_to"] = trip_req.get("returning_to") or ""
    if trip_req.get("interests"):
        display = [_NORM_TO_DISPLAY.get(n, n) for n in trip_req["interests"]]
        st.session_state["sb_interests"] = [d for d in display if d in _ALL_INTERESTS]
    if trip_req.get("budget_level"):
        st.session_state["sb_budget_level"] = trip_req["budget_level"]
    if trip_req.get("pace"):
        st.session_state["sb_pace"] = trip_req["pace"]
    if trip_req.get("traveler_type"):
        st.session_state["sb_traveler_type"] = trip_req["traveler_type"]
    if trip_req.get("constraints"):
        st.session_state["sb_constraints"] = ", ".join(trip_req["constraints"])

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail",
}
_WMO_ICONS = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️", 51: "🌦️", 53: "🌦️", 55: "🌧️",
    61: "🌧️", 63: "🌧️", 65: "🌧️", 71: "❄️", 73: "❄️", 75: "❄️", 77: "🌨️",
    80: "🌦️", 81: "🌦️", 82: "⛈️", 85: "🌨️", 86: "🌨️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}


def _geocode(place: str) -> Optional[tuple[float, float]]:
    """
    Return (lat, lon) for a place name using Nominatim.
    For multi-city destinations (e.g. 'Madrid, Barcelona, Seville'), tries
    the full string first then each city individually, returning the first match.
    """
    import re
    # Build candidate list: full string, then each split part (for multi-city)
    parts = [p.strip() for p in re.split(r'[,+&/]|\band\b', place, flags=re.IGNORECASE) if p.strip()]
    candidates = [place] if len(parts) > 1 else []
    candidates += parts

    for candidate in candidates:
        try:
            encoded = urllib.parse.quote_plus(candidate)
            url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "TravelAdvisorAgent/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                results = json.loads(resp.read())
            if results:
                return float(results[0]["lat"]), float(results[0]["lon"])
        except Exception:
            pass
    return None


def _fetch_weather(lat: float, lon: float, start_date: str, num_days: int) -> list[dict]:
    """
    Call Open-Meteo for daily forecasts. Returns list of dicts keyed by date.
    Only works for dates within the 16-day forecast window.
    """
    try:
        from datetime import date, timedelta
        start = date.fromisoformat(start_date)
        end = start + timedelta(days=num_days - 1)
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&start_date={start}&end_date={end}"
            f"&timezone=auto"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        codes = daily.get("weathercode", [])
        t_max = daily.get("temperature_2m_max", [])
        t_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        return [
            {
                "date": dates[i],
                "code": codes[i] if i < len(codes) else 0,
                "t_max": t_max[i] if i < len(t_max) else None,
                "t_min": t_min[i] if i < len(t_min) else None,
                "precip": precip[i] if i < len(precip) else None,
            }
            for i in range(len(dates))
        ]
    except Exception:
        return []


def _generate_packing_list(itinerary_data: dict) -> Optional[PackingListResponse]:
    """Call GPT-4o to generate a packing list for the current itinerary."""
    try:
        activities = []
        for day in itinerary_data.get("itinerary", [])[:3]:
            for block in day.get("blocks", []):
                act = block.get("activity", "")
                if act:
                    activities.append(act)
        activities_summary = "; ".join(activities[:9]) or "general sightseeing"

        trip_req = st.session_state.agent_state.get("trip_request") or {}
        schema_json = json.dumps(PackingListResponse.model_json_schema(), indent=2)
        prompt_text = PACKING_LIST_PROMPT.format(
            destination=itinerary_data.get("destination", "Unknown"),
            duration=itinerary_data.get("duration", ""),
            start_date=trip_req.get("start_date") or "Not specified",
            traveler_type=itinerary_data.get("traveler_type") or trip_req.get("traveler_type", "solo"),
            interests=", ".join(trip_req.get("interests", [])) or "general sightseeing",
            budget_level=trip_req.get("budget_level", "mid_range"),
            constraints=", ".join(trip_req.get("constraints", [])) or "none",
            activities_summary=activities_summary,
            schema=schema_json,
        )
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        structured = llm.with_structured_output(PackingListResponse, method="function_calling")
        return structured.invoke(prompt_text)
    except Exception as exc:
        logging.warning("Packing list generation failed: %s", exc)
        return None


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
if "packing_list" not in st.session_state:
    st.session_state.packing_list = None
if "weather_data" not in st.session_state:
    st.session_state.weather_data = {}  # day_number -> weather dict
if "pending_prefill" not in st.session_state:
    st.session_state.pending_prefill = None

# Apply any pending sidebar prefill BEFORE widgets are instantiated
if st.session_state.pending_prefill:
    _prefill_sidebar(st.session_state.pending_prefill)
    st.session_state.pending_prefill = None


# ── rendering helpers ─────────────────────────────────────────────────────────

def _render_itinerary(data: dict, weather: dict | None = None) -> None:
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
        # append weather summary to expander label if available
        w = (weather or {}).get(day_num)
        if w:
            code = w.get("code", 0)
            icon = _WMO_ICONS.get(code, "🌡️")
            t_max = w.get("t_max")
            t_min = w.get("t_min")
            if t_max is not None and t_min is not None:
                label += f"  {icon} {t_min:.0f}–{t_max:.0f}°C"
            else:
                label += f"  {icon}"
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
            if w:
                code = w.get("code", 0)
                icon = _WMO_ICONS.get(code, "🌡️")
                desc = _WMO_CODES.get(code, "Unknown")
                t_max = w.get("t_max")
                t_min = w.get("t_min")
                precip = w.get("precip")
                temp_str = f"{t_min:.0f}–{t_max:.0f}°C" if (t_max is not None and t_min is not None) else ""
                precip_str = f" · {precip:.1f}mm rain" if precip else ""
                st.caption(f"{icon} **Weather:** {desc}{('  ' + temp_str) if temp_str else ''}{precip_str}")

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


def _render_packing_list(pl: PackingListResponse) -> None:
    st.subheader("🎒 Packing List")
    st.caption(pl.trip_summary)
    cols = st.columns(min(len(pl.categories), 3))
    for i, cat in enumerate(pl.categories):
        with cols[i % len(cols)]:
            st.markdown(f"**{cat.category}**")
            for item in cat.items:
                st.markdown(f"- {item}")
    if pl.notes:
        with st.expander("📌 Packing Tips"):
            for note in pl.notes:
                st.markdown(f"- {note}")


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
        key="sb_destination",
    )
    col_d, col_s = st.columns(2)
    with col_d:
        duration_days = st.number_input("Duration (days)", min_value=1, max_value=60, value=5, step=1, key="sb_duration_days")
    with col_s:
        start_date = st.date_input("Start Date (opt.)", key="sb_start_date")

    origin = st.text_input("Departing From (opt.)", placeholder="e.g., London", key="sb_origin")
    returning_to = st.text_input("Returning To (opt.)", placeholder="e.g., London", key="sb_returning_to")

    interests = st.multiselect(
        "Interests",
        _ALL_INTERESTS,
        key="sb_interests",
    )

    budget_level = st.select_slider(
        "Budget Level",
        options=["budget", "mid_range", "luxury"],
        value="mid_range",
        key="sb_budget_level",
    )
    pace = st.select_slider(
        "Travel Pace",
        options=["relaxed", "moderate", "packed"],
        value="moderate",
        key="sb_pace",
    )
    traveler_type = st.selectbox(
        "Traveler Type",
        ["solo", "couple", "family", "group"],
        key="sb_traveler_type",
    )
    constraints_raw = st.text_area(
        "Special Constraints (opt.)",
        placeholder="e.g., wheelchair accessible, vegetarian only, no flying",
        key="sb_constraints",
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
        st.session_state.packing_list = None
        st.session_state.weather_data = {}
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
        # Support both raw itinerary JSON and history-entry JSON (which includes trip_request)
        if "itinerary" in _loaded and "trip_request" in _loaded:
            _it_data = _loaded["itinerary"]
            _req_data = _loaded.get("trip_request") or {}
        else:
            _it_data = _loaded
            _req_data = {}
        st.session_state.agent_state["final_response"] = _it_data
        st.session_state.agent_state["itinerary_response"] = _it_data
        st.session_state.agent_state["trip_request"] = _req_data
        st.session_state.packing_list = None
        st.session_state.weather_data = {}
        if _req_data:
            st.session_state.pending_prefill = _req_data
        st.rerun()

    # ── history ───────────────────────────────────────────────────────────────
    _history = _load_history()
    if _history:
        st.divider()
        st.caption("Recent trips")
        for _entry in _history[:5]:
            _label = f"🗺️ {_entry['destination']}"
            _help = _entry.get("saved_at", "")
            if st.button(_label, key=f"hist_{_entry['destination']}_{_help}",
                         help=_help, use_container_width=True):
                _it = _entry["itinerary"]
                _req = _entry.get("trip_request") or {}
                st.session_state.agent_state["final_response"] = _it
                st.session_state.agent_state["itinerary_response"] = _it
                st.session_state.agent_state["trip_request"] = _req
                st.session_state.packing_list = None
                st.session_state.weather_data = {}
                if _req:
                    st.session_state.pending_prefill = _req
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
        if result_state.get("itinerary_response"):
            _save_to_history(result_state["itinerary_response"], result_state.get("trip_request"))
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
    _render_itinerary(itinerary_response.get("data", {}), weather=st.session_state.weather_data or None)

    # ── action buttons below itinerary ────────────────────────────────────────
    btn_a, btn_b, _spacer = st.columns([1, 1, 3])
    with btn_a:
        if st.button("🎒 Generate Packing List", use_container_width=True):
            with st.spinner("Generating packing list..."):
                result = _generate_packing_list(itinerary_response.get("data", {}))
            if result:
                st.session_state.packing_list = result
                st.rerun()
            else:
                st.error("Could not generate packing list. Please try again.")

    _trip_req = agent_state.get("trip_request") or {}
    _start_date = _trip_req.get("start_date")
    # Fall back to the sidebar date widget for trips loaded from history without a stored start_date
    if not _start_date and "sb_start_date" in st.session_state:
        _sb_date = st.session_state["sb_start_date"]
        if _sb_date:
            _start_date = str(_sb_date)
    with btn_b:
        _weather_label = "🌤️ Refresh Weather" if st.session_state.weather_data else "🌤️ Add Weather Forecast"
        _weather_disabled = not bool(_start_date)
        _weather_help = "Set a Start Date in the sidebar to enable weather forecasts." if _weather_disabled else None
        if st.button(_weather_label, use_container_width=True,
                     disabled=_weather_disabled, help=_weather_help):
            _itin_data = itinerary_response.get("data", {})
            _dest_for_geo = _itin_data.get("destination", "")
            _duration = _trip_req.get("duration_days", 7)
            with st.spinner("Fetching weather forecast..."):
                _coords = _geocode(_dest_for_geo)
            if _coords:
                _lat, _lon = _coords
                with st.spinner("Loading forecast data..."):
                    _forecast = _fetch_weather(_lat, _lon, _start_date, int(_duration))
                if _forecast:
                    # map day_number (1-based) → weather entry
                    _weather_map = {}
                    for _i, _w in enumerate(_forecast):
                        _weather_map[_i + 1] = _w
                    st.session_state.weather_data = _weather_map
                    st.rerun()
                else:
                    st.warning("Weather data is only available within the next 16 days.")
            else:
                st.error(f"Could not geocode '{_dest_for_geo}'. Try a more specific destination name.")

    if st.session_state.packing_list:
        st.divider()
        _render_packing_list(st.session_state.packing_list)

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
