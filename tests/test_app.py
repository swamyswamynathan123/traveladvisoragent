import sys
from unittest.mock import MagicMock

# Mock streamlit BEFORE importing app so _in_streamlit evaluates to False
# and no Streamlit UI code runs at module level.
_st = MagicMock()
_st.runtime.exists.return_value = False
sys.modules["streamlit"] = _st


def test_streaming_callback_accumulates_tokens():
    from app import StreamlitTokenCallback

    container = MagicMock()
    cb = StreamlitTokenCallback(container)
    cb.on_llm_new_token("Day ")
    cb.on_llm_new_token("1")

    assert cb._buffer == "Day 1"
    container.code.assert_called()


def test_streaming_callback_cursor_appended():
    from app import StreamlitTokenCallback

    container = MagicMock()
    cb = StreamlitTokenCallback(container)
    cb.on_llm_new_token("Hello")

    last_call_text = container.code.call_args[0][0]
    assert last_call_text.endswith("▌")


def test_streaming_callback_end_clears_container():
    from app import StreamlitTokenCallback

    container = MagicMock()
    cb = StreamlitTokenCallback(container)
    cb.on_llm_new_token("some token")
    cb.on_llm_end(None)

    container.empty.assert_called_once()


def test_render_itinerary_with_budget_estimate_calls_expander():
    from app import _render_itinerary

    _st.columns.side_effect = lambda *a, **k: [MagicMock(), MagicMock()]

    data = {
        "destination": "Paris",
        "duration": "3 days",
        "itinerary": [],
        "budget_estimate": {
            "line_items": [
                {"category": "Flights", "low_usd": 400, "high_usd": 600, "notes": None},
                {"category": "Hotels", "low_usd": 350, "high_usd": 500, "notes": None},
                {"category": "Food & Dining", "low_usd": 150, "high_usd": 200, "notes": None},
                {"category": "Activities", "low_usd": 80, "high_usd": 120, "notes": None},
            ],
            "per_person_low_usd": 980,
            "per_person_high_usd": 1420,
            "group_low_usd": 1960,
            "group_high_usd": 2840,
            "num_travelers": 2,
            "currency_note": "All estimates in USD.",
        },
    }
    _st.expander.reset_mock()
    _render_itinerary(data)

    expander_calls = [str(c) for c in _st.expander.call_args_list]
    assert any("Budget Estimate" in c for c in expander_calls)


def test_render_itinerary_without_budget_estimate_skips_expander():
    from app import _render_itinerary

    _st.columns.side_effect = lambda *a, **k: [MagicMock(), MagicMock()]

    data = {
        "destination": "Paris",
        "duration": "3 days",
        "itinerary": [],
    }
    _st.expander.reset_mock()
    _render_itinerary(data)

    expander_calls = [str(c) for c in _st.expander.call_args_list]
    assert not any("Budget Estimate" in c for c in expander_calls)
