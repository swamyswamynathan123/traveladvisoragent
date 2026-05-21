# Streaming Itinerary Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream LLM tokens into the status box while the itinerary is being generated so the user sees real-time progress instead of a blank wait.

**Architecture:** Add a `streaming=True` flag to the `_llm()` helper in `graph.py` and pass it when constructing the `gpt-4o` instance in the two generation functions. In `app.py`, add a `StreamlitTokenCallback(BaseCallbackHandler)` class that writes each token to an `st.empty()` container; pass the callback to the graph via `config={"callbacks": [cb]}`. The token stream clears itself when generation ends and the final formatted card renders normally below.

**Tech Stack:** LangChain `langchain_core.callbacks.BaseCallbackHandler`, LangGraph `compiled_graph.stream(config=...)`, Streamlit `st.empty()`, `st.status()`, `st.code()`.

---

## File Map

| File | Change |
|------|--------|
| `graph.py` | Add `streaming: bool = False` to `_llm()` (line 66); pass `streaming=True` in `_generate_itinerary()` (line 374) and `_generate_answer()` (line 413) |
| `app.py` | Add `from langchain_core.callbacks import BaseCallbackHandler` import (line 16); add `StreamlitTokenCallback` class before `_run_graph()` (line 759); modify `_run_graph()` body (lines 769–775) |
| `tests/test_graph_nodes.py` | Add `test_generate_itinerary_uses_streaming_llm` |
| `tests/test_app.py` | Create with 3 `StreamlitTokenCallback` unit tests |

---

## Task 1: Failing test — `_generate_itinerary` must use `streaming=True`

**Files:**
- Modify: `tests/test_graph_nodes.py`

- [ ] **Step 1: Add the failing test to the bottom of `tests/test_graph_nodes.py`**

Append this function after the last existing test (line 465):

```python
def test_generate_itinerary_uses_streaming_llm():
    from graph import _generate_itinerary
    from schemas import ItineraryResponse, DayPlan, TimeBlock

    minimal_response = ItineraryResponse(
        destination="Paris",
        duration="3 days",
        itinerary=[DayPlan(day_number=1, theme="Arrival",
                           blocks=[TimeBlock(time_of_day="morning", activity="Arrive")])],
        logistics_notes="By train.",
    )

    with patch("graph._llm") as mock_llm_fn:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = minimal_response
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_fn.return_value = mock_llm

        result = _generate_itinerary(_planning_state())

    mock_llm_fn.assert_called_once_with(model="gpt-4o", temperature=0.3, streaming=True)
    assert result["final_response"]["type"] == "itinerary"
```

- [ ] **Step 2: Run the test and confirm it fails**

```
pytest tests/test_graph_nodes.py::test_generate_itinerary_uses_streaming_llm -v
```

Expected output: `FAILED` with `AssertionError: expected call not found` (because `_llm` is called without `streaming=True` today).

---

## Task 2: Implement `streaming=True` in `graph.py`

**Files:**
- Modify: `graph.py` (lines 66–70, 374, 413)

- [ ] **Step 1: Add `streaming` parameter to `_llm()` helper**

Find in `graph.py` (lines 66–70):
```python
def _llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)
    return ChatOpenAI(model=model, temperature=temperature)
```

Replace with:
```python
def _llm(model: str = "gpt-4o-mini", temperature: float = 0.0, streaming: bool = False):
    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)
    return ChatOpenAI(model=model, temperature=temperature, streaming=streaming)
```

- [ ] **Step 2: Use `streaming=True` in `_generate_itinerary()`**

Find in `graph.py` (line 374):
```python
def _generate_itinerary(state: dict) -> dict:
    llm = _llm(model="gpt-4o", temperature=0.3)
```

Replace the `llm = ...` line only:
```python
def _generate_itinerary(state: dict) -> dict:
    llm = _llm(model="gpt-4o", temperature=0.3, streaming=True)
```

- [ ] **Step 3: Use `streaming=True` in `_generate_answer()`**

Find in `graph.py` (line 413):
```python
def _generate_answer(state: dict) -> dict:
    llm = _llm(model="gpt-4o", temperature=0.3)
```

Replace the `llm = ...` line only:
```python
def _generate_answer(state: dict) -> dict:
    llm = _llm(model="gpt-4o", temperature=0.3, streaming=True)
```

Note: `_retry_with_repair()` takes `llm` as a parameter and reuses it — it inherits `streaming=True` automatically since both `_generate_itinerary` and `_generate_answer` pass their `llm` instance to it.

- [ ] **Step 4: Run the Task 1 test and confirm it passes**

```
pytest tests/test_graph_nodes.py::test_generate_itinerary_uses_streaming_llm -v
```

Expected: `PASSED`

- [ ] **Step 5: Run the full test suite and confirm nothing regressed**

```
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```
git add graph.py tests/test_graph_nodes.py
git commit -m "feat: add streaming=True to gpt-4o instances in generate_response_node"
```

---

## Task 3: Failing tests — `StreamlitTokenCallback` behaviour

**Files:**
- Create: `tests/test_app.py`

- [ ] **Step 1: Create `tests/test_app.py` with three failing tests**

```python
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
```

- [ ] **Step 2: Run the tests and confirm they all fail**

```
pytest tests/test_app.py -v
```

Expected: all 3 tests fail with `ImportError: cannot import name 'StreamlitTokenCallback' from 'app'`.

---

## Task 4: Implement `StreamlitTokenCallback` and update `_run_graph()` in `app.py`

**Files:**
- Modify: `app.py` (line 16, lines 759–776)

- [ ] **Step 1: Add `BaseCallbackHandler` import to `app.py`**

Find in `app.py` (line 16):
```python
from langchain_openai import ChatOpenAI
```

Replace with:
```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
```

- [ ] **Step 2: Add `StreamlitTokenCallback` class before `_run_graph()`**

Find in `app.py` (line 759):
```python
def _run_graph(state: dict) -> dict:
```

Insert the new class immediately before that line:
```python
class StreamlitTokenCallback(BaseCallbackHandler):
    def __init__(self, container) -> None:
        self._container = container
        self._buffer = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self._buffer += token
        self._container.code(self._buffer + "▌", language=None)

    def on_llm_end(self, response, **kwargs) -> None:
        self._container.empty()


def _run_graph(state: dict) -> dict:
```

- [ ] **Step 3: Update `_run_graph()` body**

Find in `app.py` (lines 759–776) the current body:
```python
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
        for chunk in st.session_state.compiled_graph.stream(state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                st.write(_NODE_LABELS.get(node_name, f"Running {node_name}..."))
                result = node_output
        status.update(label="Done!", state="complete", expanded=False)
    return result
```

Replace with:
```python
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
```

- [ ] **Step 4: Run the Task 3 tests and confirm they all pass**

```
pytest tests/test_app.py -v
```

Expected: all 3 tests `PASSED`.

- [ ] **Step 5: Run the full test suite**

```
pytest tests/ -v
```

Expected: all tests pass. Note: `test_app.py` imports `app.py`, which triggers all of app.py's module-level imports (`pandas`, `pydeck`, `langchain`, etc.). If any package is missing, install it with `pip install <package>`. All existing tests should continue to pass unchanged.

- [ ] **Step 6: Commit**

```
git add app.py tests/test_app.py
git commit -m "feat: stream LLM tokens into status box during itinerary generation"
```

---

## Final check

After both commits, run the app manually to verify the streaming experience:

```
streamlit run app.py
```

1. Enter a destination and click **Generate Itinerary**.
2. The "Working on it..." status box should expand.
3. While the `generate_response` node runs, a monospace code block inside the status box should fill with token-by-token JSON output.
4. When generation finishes, the code block disappears, the status collapses to "Done!", and the formatted itinerary card renders below.

No API keys are required for the automated tests — all LLM calls remain mocked.
