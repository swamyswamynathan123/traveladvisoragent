# Streaming Itinerary Generation Design

## Goal

Stream LLM tokens into the status box while the itinerary is being generated, so the user sees real-time progress instead of a blank wait.

## Architecture

Three components change; no new files or imports are added to the dependency chain.

```
schemas.py ← tools.py ← search.py ─┐
                                     ├─ app.py   ← StreamlitTokenCallback (new class)
schemas.py ← tools.py ← prompts.py ← graph.py ─┘
                                ↑
                        streaming=True added here
```

**No changes** to `schemas.py`, `tools.py`, `search.py`, or `prompts.py`.

## Component 1 — `StreamlitTokenCallback` (new class in `app.py`)

A `BaseCallbackHandler` subclass placed just above `_run_graph()`:

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
```

Design decisions:
- `st.code(..., language=None)` renders the JSON stream in a monospace box without syntax highlighting. Because `with_structured_output` causes the LLM to emit JSON tokens (not prose), this framing makes the output look deliberate rather than broken.
- `on_llm_end` clears the container entirely. The status box then collapses and the normal `_render_itinerary` card renders below — no double-display.
- The buffer accumulates the full response in memory (~2–4 KB for a typical itinerary). No flush strategy needed.
- No thread-safety guards: Streamlit's script runner is single-threaded per session.

**Known trade-off:** The streamed content is JSON-formatted (field names, braces, quotes mixed with itinerary text). This is a consequence of using `with_structured_output`. The alternative — a two-phase approach that streams readable prose then parses it with a cheap model — would produce better UX but doubles API cost per request. The single-phase approach is chosen here.

## Component 2 — `_run_graph()` changes (in `app.py`)

```python
def _run_graph(state: dict) -> dict:
    _NODE_LABELS = { ... }  # unchanged

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

- `st.empty()` is created once before the loop and handed to the callback.
- `config={"callbacks": [cb]}` passes the callback into every node in the graph for this invocation.
- When the loop ends, `on_llm_end` has already cleared the container, so the status collapses onto an empty placeholder — no flicker.

## Component 3 — `graph.py` changes

Add `streaming=True` to the `ChatOpenAI` instance in `generate_response_node`:

```python
# Before
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# After
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)
```

Also add `streaming=True` to the `ChatOpenAI` instance inside `_retry_with_repair()` so the repair path streams correctly if it fires.

The `gpt-4o-mini` instances in `collect_requirements_node` and in `search.py` are left unchanged — streaming there would send spurious tokens to the callback during non-generation nodes.

## Data Flow

```
user submits request
    → _run_graph() creates st.empty() + StreamlitTokenCallback
    → compiled_graph.stream(..., config={"callbacks": [cb]})
        → collect_requirements_node  (no streaming, callback silent)
        → validate_inputs_node       (no LLM, callback silent)
        → search_with_tavily_node    (no streaming, callback silent)
        → generate_response_node
              → ChatOpenAI(streaming=True).with_structured_output(...)
                   → on_llm_new_token fires per token → st.empty() updates
                   → on_llm_end fires → st.empty() cleared
        → respond_to_user_node       (no LLM, callback silent)
    → status.update("Done!", state="complete")
    → _render_itinerary() renders final card below the collapsed status
```

## Testing

### `tests/test_graph_nodes.py` — streaming flag smoke test

```python
def test_generate_response_node_uses_streaming_llm(monkeypatch):
    captured = {}
    original_init = ChatOpenAI.__init__
    def mock_init(self, **kwargs):
        captured.update(kwargs)
        original_init(self, **kwargs)
    monkeypatch.setattr(ChatOpenAI, "__init__", mock_init)
    # invoke generate_response_node with minimal itinerary state
    assert captured.get("streaming") is True
```

### `tests/test_app.py` — callback unit test

```python
def test_streaming_callback_accumulates_tokens():
    container = MagicMock()
    cb = StreamlitTokenCallback(container)
    cb.on_llm_new_token("Day ")
    cb.on_llm_new_token("1")
    assert cb._buffer == "Day 1"
    cb.on_llm_end(None)
    container.empty.assert_called_once()
```

No API keys required — all LLM calls stay mocked as in the existing suite.

## Error Handling

No new error paths introduced. The callback is passive — if the LLM call raises, the existing exception handling in `generate_response_node` / `_retry_with_repair()` catches it as before. The `st.empty()` container is left in whatever state it was in, which is fine because the status block will close with an error state anyway.

## Out of Scope

- Two-phase prose streaming (better UX, higher cost — future consideration)
- Streaming during `search_with_tavily_node` (Tavily is not an LLM; per-query labels are already shown)
- Streaming during `collect_requirements_node` (fast gpt-4o-mini call, not worth displaying)
