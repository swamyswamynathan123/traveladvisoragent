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
