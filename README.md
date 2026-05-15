# Travel Advisor Agent

A multi-step AI travel planning agent built with LangGraph, OpenAI, Tavily, and Streamlit.

## Features

- **Trip Itinerary Generation** — Day-by-day plans with morning/afternoon/evening blocks, logistics notes, and rainy-day alternatives
- **Travel Q&A** — Answer any travel question with real-time web search results and cited sources
- **Real-time Web Search** — Tavily search runs on every request to ground answers in current information
- **Structured Outputs** — All responses validated against Pydantic schemas with automatic retry on failure
- **Chat Interface** — Follow-up questions and plan refinements in a persistent chat session

## Architecture

```
Streamlit UI (app.py)
       │
       ▼
LangGraph Graph (graph.py)
       │
       ├─ collect_requirements   ← parse intent from user message
       ├─ validate_inputs        ← check if enough info to proceed
       │       │
       │  [missing info]         [complete]
       │       │                      │
       ├─ ask_clarification      search_with_tavily  ← up to 6 Tavily calls
       │       │                      │
       │      END              generate_response     ← OpenAI gpt-4o structured output
       │                              │
       └──────────────────── respond_to_user → END
```

## Project Structure

```
traveladvisoragent/
├── app.py           # Streamlit UI — sidebar form, chat, itinerary renderer
├── graph.py         # LangGraph nodes, conditional routing, compiled graph
├── tools.py         # Tavily search wrapper
├── schemas.py       # Pydantic models for all state and response types
├── prompts.py       # LLM prompt templates for each node
├── requirements.txt
├── .env.example
└── tests/
    ├── test_schemas.py
    ├── test_tools.py
    └── test_graph_nodes.py
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env        # macOS/Linux
copy .env.example .env      # Windows
```

Edit `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

- OpenAI API key: https://platform.openai.com/api-keys
- Tavily API key: https://app.tavily.com

### 3. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Usage

### Plan a trip

1. Fill in destination, duration, interests, and budget in the sidebar
2. Click **Generate Itinerary**
3. The agent runs up to 6 Tavily searches, then generates a day-by-day plan
4. Use **Refine Plan** to adjust the result, or type in the chat input for follow-ups

### Ask a travel question

Type any travel question directly in the chat input at the bottom of the page (e.g. *"What is the best time to visit Kyoto for cherry blossoms?"*).

## Running Tests

No API keys required — all external calls are mocked.

```bash
pytest tests/ -v
```

34 tests covering schema validation, Tavily tool error handling, and all graph node logic.

## Models Used

| Node | Model |
|---|---|
| Intent detection, clarification | `gpt-4o-mini` |
| Itinerary generation, Q&A | `gpt-4o` |

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
