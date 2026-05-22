from __future__ import annotations
import html as _html
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

load_dotenv()  # must precede local imports so os.getenv() reads .env values

from graph import build_graph, build_initial_state
from prompts import PACKING_LIST_PROMPT
from schemas import PackingListResponse
from search import search_flights, search_hotels

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


def _fetch_weather(lat: float, lon: float, start_date: str, num_days: int) -> tuple[list[dict], str]:
    """
    Return (daily_rows, label) using the appropriate Open-Meteo endpoint:
      - "forecast"   → within the next 16 days (forecast API)
      - "historical" → past trip up to 92 days ago (forecast API with past window)
      - "archive"    → older past trips (archive API, data from 1940)
      - "typical"    → future trip beyond 16 days (archive API, same dates last year)
    Returns ([], "error") on failure.
    """
    from datetime import date, timedelta

    FORECAST_PAST_DAYS = 92   # forecast API's maximum past window
    ARCHIVE_DELAY_DAYS  = 5   # archive API lags ~5 days behind today

    def _shift_year(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            return d.replace(year=d.year - 1, day=28)

    def _call(api_url: str, q_start: date, q_end: date) -> list[dict]:
        if q_start > q_end:
            return []
        url = (
            f"{api_url}?latitude={lat}&longitude={lon}"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&start_date={q_start}&end_date={q_end}&timezone=auto"
        )
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                raw = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logging.warning("Weather API error %s for URL %s — %s", exc.code, url, body)
            raise
        daily = raw.get("daily", {})
        dates = daily.get("time", [])
        codes = daily.get("weather_code", daily.get("weathercode", []))
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

    try:
        start = date.fromisoformat(start_date)
        end   = start + timedelta(days=num_days - 1)
        today = date.today()
        forecast_limit  = today + timedelta(days=15)  # API counts today as day 1, so max is today+15
        archive_horizon = today - timedelta(days=ARCHIVE_DELAY_DAYS)

        if start > forecast_limit:
            # Far future — use same calendar dates from last year as indicative weather.
            # Keep shifting back until the dates fall within the archive's coverage.
            qs, qe = _shift_year(start), _shift_year(end)
            while qs > archive_horizon:
                qs, qe = _shift_year(qs), _shift_year(qe)
            rows = _call("https://archive-api.open-meteo.com/v1/archive", qs, qe)
            label = "typical"

        elif start < today - timedelta(days=FORECAST_PAST_DAYS):
            # Old past trip — use archive.  BUG FIX: was checking `end` before; must check `start`.
            qs = start
            qe = min(end, archive_horizon)  # archive may not have the last ~5 days
            rows = _call("https://archive-api.open-meteo.com/v1/archive", qs, qe)
            label = "archive"

        else:
            # Recent past (≤92 days ago) or near future (≤16 days): forecast API.
            qs = start
            qe = min(end, forecast_limit)
            rows = _call("https://api.open-meteo.com/v1/forecast", qs, qe)
            label = "forecast" if start >= today else "historical"

        return rows, label

    except Exception as exc:
        logging.warning("Weather fetch failed: %s", exc)
        return [], "error"


def _build_weather_map(destination: str, start_date: str, total_days: int, itin_data: dict) -> tuple[dict, str]:
    """
    Build a day_number → weather_entry dict.
    For multi-city trips, geocodes each city separately and maps days to the
    nearest city using block coordinates, then fetches per-city forecasts.
    Falls back to single-city logic for single-destination trips.
    """
    import re
    from datetime import date, timedelta

    cities = [c.strip() for c in re.split(r'[,+&/]|\band\b', destination, flags=re.IGNORECASE) if c.strip()]

    if len(cities) <= 1:
        coords = _geocode(destination)
        if not coords:
            return {}, "error"
        actual_days = len(itin_data.get("itinerary", [])) or total_days
        rows, label = _fetch_weather(coords[0], coords[1], start_date, actual_days)
        return {i + 1: r for i, r in enumerate(rows)}, label

    # Geocode each city
    city_coords: dict[str, tuple] = {}
    for city in cities:
        c = _geocode(city)
        if c:
            city_coords[city] = c

    if not city_coords:
        return {}, "error"

    def _nearest_city(lat: float, lon: float) -> str:
        return min(city_coords, key=lambda c: (lat - city_coords[c][0]) ** 2 + (lon - city_coords[c][1]) ** 2)

    # Assign each day to the nearest geocoded city using its first block with coordinates
    day_to_city: dict[int, str] = {}
    last_city = cities[0]
    for day in sorted(itin_data.get("itinerary", []), key=lambda d: d.get("day_number", 0)):
        dn = day.get("day_number", 1)
        assigned = last_city
        for block in day.get("blocks", []):
            blat, blon = block.get("latitude"), block.get("longitude")
            if blat and blon:
                assigned = _nearest_city(blat, blon)
                break
        day_to_city[dn] = assigned
        last_city = assigned

    # Fetch one forecast per city covering only the days spent there
    trip_start = date.fromisoformat(start_date)
    city_day_lists: dict[str, list[int]] = {}
    for dn, city in day_to_city.items():
        city_day_lists.setdefault(city, []).append(dn)

    weather_map: dict[int, dict] = {}
    overall_label = "forecast"
    for city, days in city_day_lists.items():
        if city not in city_coords:
            continue
        clat, clon = city_coords[city]
        first_day = min(days)
        num_days = max(days) - first_day + 1
        city_start = trip_start + timedelta(days=first_day - 1)
        rows, label = _fetch_weather(clat, clon, str(city_start), num_days)
        if label in ("typical", "archive", "historical"):
            overall_label = label
        for i, dn in enumerate(range(first_day, first_day + len(rows))):
            if dn in day_to_city and day_to_city[dn] == city and i < len(rows):
                weather_map[dn] = rows[i]

    return weather_map, overall_label


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


def _render_flight_card(f: dict, highlight: bool) -> str:
    border = "#4c5bd4" if highlight else "#374151"
    price_color = "#4ade80" if highlight else "#f59e0b"
    airline = _html.escape(str(f.get("airline") or ""))
    flight_number = _html.escape(str(f.get("flight_number") or ""))
    duration = _html.escape(str(f.get("duration") or ""))
    stops = _html.escape(str(f.get("stops") or ""))
    departure_time = _html.escape(str(f.get("departure_time") or ""))
    price = _html.escape(str(f.get("price") or "N/A"))
    origin = _html.escape(str(f.get("origin") or ""))
    destination = _html.escape(str(f.get("destination") or ""))
    raw_url = f.get("url") or "#"
    url = _html.escape(raw_url if raw_url.startswith(("http://", "https://")) else "#", quote=True)
    meta_parts = [p for p in [duration, stops, f"Departs {departure_time}" if departure_time else None] if p]
    meta = " · ".join(meta_parts) if meta_parts else ""
    airline_line = airline + (f" · {flight_number}" if flight_number else "")
    route = f"{origin} → {destination}"
    return (
        f'<div style="background:#1e2130;border-radius:6px;padding:10px;display:flex;'
        f'align-items:center;gap:12px;border-left:3px solid {border};margin-bottom:8px">'
        f'<div style="font-size:20px">🛫</div>'
        f'<div style="flex:1">'
        f'<div style="font-weight:bold;color:#e2e8f0;font-size:13px">{airline_line}</div>'
        f'<div style="color:#94a3b8;font-size:11px">{route}{(" · " + meta) if meta else ""}</div>'
        f'</div>'
        f'<div style="text-align:right;margin-right:8px">'
        f'<div style="color:{price_color};font-size:15px;font-weight:bold">{price}</div>'
        f'<div style="color:#64748b;font-size:10px">per person</div>'
        f'</div>'
        f'<a href="{url}" target="_blank" style="background:{border};color:white;'
        f'border-radius:4px;padding:5px 12px;font-size:11px;text-decoration:none;white-space:nowrap">Book →</a>'
        f'</div>'
    )


def _render_hotel_card(h: dict, highlight: bool) -> str:
    border = "#4c5bd4" if highlight else "#374151"
    price_color = "#4ade80" if highlight else "#f59e0b"
    name = _html.escape(str(h.get("name") or ""))
    neighborhood = _html.escape(str(h.get("neighborhood") or ""))
    amenities = _html.escape(str(h.get("amenities") or ""))
    rating = _html.escape(str(h.get("rating") or ""))
    rating_label = _html.escape(str(h.get("rating_label") or ""))
    price_per_night = _html.escape(str(h.get("price_per_night") or "N/A"))
    price_total = _html.escape(str(h.get("price_total") or ""))
    raw_url = h.get("url") or "#"
    url = _html.escape(raw_url if raw_url.startswith(("http://", "https://")) else "#", quote=True)
    stars_val = h.get("stars")
    try:
        stars_int = int(stars_val) if stars_val is not None else 0
    except (TypeError, ValueError):
        stars_int = 0
    stars_int = max(0, min(5, stars_int))
    stars_str = "★" * stars_int if stars_int > 0 else ""
    name_line = name + (f" {stars_str}" if stars_str else "")
    sub_parts = [p for p in [neighborhood, amenities] if p]
    sub = " · ".join(sub_parts) if sub_parts else ""
    rating_str = ""
    if rating:
        rating_str = f"★ {rating}"
        if rating_label:
            rating_str += f" · {rating_label}"
    total_str = f"~{price_total} total" if price_total else ""
    rating_div = (
        f'<div style="color:#fbbf24;font-size:11px;margin-top:2px">{rating_str}</div>'
        if rating_str else ""
    )
    return (
        f'<div style="background:#1e2130;border-radius:6px;padding:10px;display:flex;'
        f'gap:12px;border-left:3px solid {border};margin-bottom:8px">'
        f'<div style="font-size:26px;align-self:center">🏨</div>'
        f'<div style="flex:1">'
        f'<div style="font-weight:bold;color:#e2e8f0;font-size:13px">{name_line}</div>'
        f'<div style="color:#94a3b8;font-size:11px">{sub}</div>'
        f'{rating_div}'
        f'</div>'
        f'<div style="text-align:right;align-self:center;margin-right:8px">'
        f'<div style="color:{price_color};font-size:14px;font-weight:bold">'
        f'{price_per_night}'
        f'<span style="font-size:10px;color:#94a3b8">/night</span></div>'
        f'<div style="color:#64748b;font-size:10px">{total_str}</div>'
        f'<a href="{url}" target="_blank" style="background:{border};color:white;'
        f'border-radius:4px;padding:4px 10px;font-size:10px;text-decoration:none;'
        f'display:inline-block;margin-top:4px">Book →</a>'
        f'</div>'
        f'</div>'
    )


logging.basicConfig(level=logging.INFO)

_in_streamlit = getattr(getattr(st, "runtime", None), "exists", lambda: False)()
if _in_streamlit:
    st.set_page_config(
        page_title="Travel Advisor Agent",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ── session state bootstrap ───────────────────────────────────────────────────

if _in_streamlit:
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
    if "weather_label" not in st.session_state:
        st.session_state.weather_label = "forecast"
    if "pending_prefill" not in st.session_state:
        st.session_state.pending_prefill = None
    if "geocode_cache" not in st.session_state:
        st.session_state.geocode_cache = {}  # location string → (lat, lon) or None
    if "flight_results" not in st.session_state:
        st.session_state.flight_results = []
    if "hotel_results" not in st.session_state:
        st.session_state.hotel_results = []
    if "flight_search_done" not in st.session_state:
        st.session_state.flight_search_done = False
    if "hotel_search_done" not in st.session_state:
        st.session_state.hotel_search_done = False
    # Sidebar widget defaults — only set once; _prefill_sidebar overwrites these on history load
    if "sb_duration_days" not in st.session_state:
        st.session_state.sb_duration_days = 5
    if "sb_budget_level" not in st.session_state:
        st.session_state.sb_budget_level = "mid_range"
    if "sb_pace" not in st.session_state:
        st.session_state.sb_pace = "moderate"
    if "sb_traveler_type" not in st.session_state:
        st.session_state.sb_traveler_type = "solo"

    # Apply any pending sidebar prefill BEFORE widgets are instantiated
    if st.session_state.pending_prefill:
        _prefill_sidebar(st.session_state.pending_prefill)
        st.session_state.pending_prefill = None


# ── rendering helpers ─────────────────────────────────────────────────────────

def _render_itinerary(data: dict, weather: dict | None = None, weather_label: str = "forecast") -> None:
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
                _wtype = {"typical": " (typical, last year)", "archive": " (historical)", "historical": " (actual)"}.get(weather_label, "")
                st.caption(f"{icon} **Weather{_wtype}:** {desc}{('  ' + temp_str) if temp_str else ''}{precip_str}")

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
    cache: dict = st.session_state.geocode_cache
    points = []
    to_geocode: list[tuple] = []  # (day_num, color, activity, location)

    for day in data.get("itinerary", []):
        day_num = day.get("day_number", 1)
        color = _DAY_COLORS[(day_num - 1) % len(_DAY_COLORS)]
        for block in day.get("blocks", []):
            lat = block.get("latitude")
            lon = block.get("longitude")
            label = f"Day {day_num}: {block.get('activity', '')}"
            if lat and lon:
                points.append({"lat": lat, "lon": lon, "label": label, "color": color})
            else:
                loc = block.get("location") or block.get("activity", "")[:80]
                if loc:
                    if loc in cache:
                        coords = cache[loc]
                        if coords:
                            points.append({"lat": coords[0], "lon": coords[1], "label": label, "color": color})
                    elif len(to_geocode) < 12:  # cap to avoid excessive API calls
                        to_geocode.append((day_num, color, label, loc))

    # Geocode missing locations (first render only; cached on subsequent renders)
    if to_geocode:
        for _dn, _color, _label, _loc in to_geocode:
            coords = _geocode(_loc)
            cache[_loc] = coords
            if coords:
                points.append({"lat": coords[0], "lon": coords[1], "label": _label, "color": _color})

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


class StreamlitTokenCallback(BaseCallbackHandler):
    def __init__(self, container) -> None:
        self._container = container
        self._buffer = ""

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs) -> None:
        self._buffer = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._buffer += token
        self._container.code(self._buffer + "▌", language=None)

    def on_llm_end(self, response, **kwargs) -> None:
        self._container.empty()

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        self._container.empty()


def _run_graph(state: dict) -> dict:
    _NODE_LABELS = {
        "collect_requirements": "🧠 Understanding your request...",
        "validate_inputs": "✅ Validating inputs...",
        "ask_clarification": "💬 Preparing clarification...",
        "search_with_tavily": "🔍 Searching travel information...",
        "generate_response": "✍️ Generating response...",
        "personalization_check": "🎯 Checking personalization...",
        "respond_to_user": "📝 Wrapping up...",
    }
    result = state
    with st.status("Working on it...", expanded=True) as status:
        stream_container = st.empty()
        cb = StreamlitTokenCallback(stream_container)
        for chunk in st.session_state.compiled_graph.stream(
            state,
            stream_mode="updates",
            config={"callbacks": [cb]},
        ):
            for node_name, node_output in chunk.items():
                st.write(_NODE_LABELS.get(node_name, f"Running {node_name}..."))
                result = node_output
        status.update(label="Done!", state="complete", expanded=False)
    return result


# ── sidebar ───────────────────────────────────────────────────────────────────

if _in_streamlit:
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
            duration_days = st.number_input("Duration (days)", min_value=1, max_value=60, step=1, key="sb_duration_days")
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
            key="sb_budget_level",
        )
        pace = st.select_slider(
            "Travel Pace",
            options=["relaxed", "moderate", "packed"],
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
            st.session_state.weather_label = "forecast"
            st.session_state.flight_results = []
            st.session_state.hotel_results = []
            st.session_state.flight_search_done = False
            st.session_state.hotel_search_done = False
            st.session_state["flight_origin_input"] = ""
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
            st.session_state.flight_results = []
            st.session_state.hotel_results = []
            st.session_state.flight_search_done = False
            st.session_state.hotel_search_done = False
            st.session_state["flight_origin_input"] = ""
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
                    st.session_state.flight_results = []
                    st.session_state.hotel_results = []
                    st.session_state.flight_search_done = False
                    st.session_state.hotel_search_done = False
                    st.session_state["flight_origin_input"] = ""
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
            st.session_state.flight_results = []
            st.session_state.hotel_results = []
            st.session_state.flight_search_done = False
            st.session_state.hotel_search_done = False
            st.session_state["flight_origin_input"] = ""
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
        _trip_req = agent_state.get("trip_request") or {}
        _start_date = _trip_req.get("start_date")
        if not _start_date and "sb_start_date" in st.session_state:
            _sb_date = st.session_state["sb_start_date"]
            if _sb_date:
                _start_date = str(_sb_date)

        tab_itin, tab_flights, tab_hotels = st.tabs(["🗺️ Itinerary", "✈️ Flights", "🏨 Hotels"])

        with tab_itin:
            _render_itinerary(
                itinerary_response.get("data", {}),
                weather=st.session_state.weather_data or None,
                weather_label=st.session_state.weather_label,
            )
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
            with btn_b:
                _weather_label = "🌤️ Refresh Weather" if st.session_state.weather_data else "🌤️ Add Weather Forecast"
                _weather_disabled = not bool(_start_date)
                _weather_help = "Set a Start Date in the sidebar to enable weather forecasts." if _weather_disabled else None
                if st.button(_weather_label, use_container_width=True,
                             disabled=_weather_disabled, help=_weather_help):
                    _itin_data = itinerary_response.get("data", {})
                    _dest_for_geo = _itin_data.get("destination", "")
                    _duration = _trip_req.get("duration_days", 7)
                    with st.spinner(
                        "Fetching weather (per city)..."
                        if "," in _dest_for_geo or "+" in _dest_for_geo
                        else "Fetching weather forecast..."
                    ):
                        _weather_map, _wlabel = _build_weather_map(
                            _dest_for_geo, _start_date, int(_duration), _itin_data
                        )
                    if _weather_map:
                        st.session_state.weather_data = _weather_map
                        st.session_state.weather_label = _wlabel
                        if _wlabel == "typical":
                            st.info("Showing typical weather from the same dates last year — your trip is more than 16 days away.")
                        st.rerun()
                    else:
                        st.warning("Could not fetch weather data for this destination and date range.")
            if st.session_state.packing_list:
                st.divider()
                _render_packing_list(st.session_state.packing_list)

        with tab_flights:
            _dest = (_trip_req.get("destination") or "").strip()
            _origin = (_trip_req.get("origin") or "").strip()
            if not _origin:
                _origin = st.text_input(
                    "✈️ Flying from (city or airport code)",
                    key="flight_origin_input",
                    placeholder="e.g. New York",
                )
            # Fires on first render after itinerary generation (not on tab click — Streamlit executes all tab blocks on every rerun)
            if _origin and _dest and not st.session_state.flight_search_done:
                with st.spinner(f"Searching flights from {_origin} to {_dest}..."):
                    st.session_state.flight_results = search_flights(_origin, _dest, _start_date or "")
                st.session_state.flight_search_done = True
            if st.session_state.flight_search_done:
                flights = st.session_state.flight_results
                if not flights:
                    st.info("No flight results found. Try refining your origin or destination.")
                else:
                    st.markdown(
                        "".join(_render_flight_card(f, highlight=(i == 0)) for i, f in enumerate(flights)),
                        unsafe_allow_html=True,
                    )
                    st.caption("⚠️ Prices sourced from web search — verify on the booking site before purchasing. Powered by Tavily.")
                if st.button("🔄 Refresh Flight Search", key="refresh_flights"):
                    st.session_state.flight_search_done = False
                    st.session_state.flight_results = []
                    st.rerun()

        with tab_hotels:
            _dest = (_trip_req.get("destination") or "").strip()
            _duration = int(_trip_req.get("duration_days") or 7)
            _budget = _trip_req.get("budget_level", "mid_range")
            # Fires on first render after itinerary generation (eager, not on-demand)
            if _dest and not st.session_state.hotel_search_done:
                if not _start_date:
                    st.info("Set a Start Date in the sidebar to search for hotels with accurate pricing.")
                else:
                    with st.spinner(f"Searching hotels in {_dest}..."):
                        st.session_state.hotel_results = search_hotels(_dest, _start_date, _duration, _budget)
                    st.session_state.hotel_search_done = True
            if st.session_state.hotel_search_done:
                hotels = st.session_state.hotel_results
                if not hotels:
                    st.info("No hotel results found. Try refining your destination or dates.")
                else:
                    st.markdown(
                        "".join(_render_hotel_card(h, highlight=(i == 0)) for i, h in enumerate(hotels)),
                        unsafe_allow_html=True,
                    )
                    st.caption("⚠️ Prices sourced from web search — verify on the booking site before purchasing. Powered by Tavily.")
                if st.button("🔄 Refresh Hotel Search", key="refresh_hotels"):
                    st.session_state.hotel_search_done = False
                    st.session_state.hotel_results = []
                    st.rerun()

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
