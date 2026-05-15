SYSTEM_PROMPT = """You are an expert travel advisor with deep knowledge of destinations worldwide.
You help users plan detailed day-by-day itineraries and answer travel questions accurately.

Core rules:
- Only cite specific facts (prices, hours, seasonal events, neighborhood names) when they appear in provided Tavily search results.
- For any claim NOT backed by search results, mark it as "[General knowledge — verify before traveling]".
- If Tavily results conflict, summarize the conflict and flag it as an open question.
- Be explicit about every assumption you make.
- Never invent URLs, phone numbers, or booking links."""


INTENT_DETECTION_PROMPT = """\
Analyze the user message below and return a JSON object with these exact fields:

User message: {user_message}

Return ONLY valid JSON — no markdown, no explanation — with this structure:
{{
  "intent": "<planning | question | unknown>",
  "trip_request": {{
    "destination": "<string or null>",
    "duration_days": <int or null>,
    "start_date": "<YYYY-MM-DD or null>",
    "origin": "<string or null>",
    "interests": ["<string>"],
    "budget_level": "<budget | mid_range | luxury>",
    "pace": "<relaxed | moderate | packed>",
    "constraints": ["<string>"]
  }},
  "travel_question": "<question string or null>",
  "clarification_needed": ["<missing field name>"]
}}

Rules:
- intent is "planning" if the user wants an itinerary or trip plan.
- intent is "question" if the user is asking a specific travel question.
- For planning, critical fields are destination AND (duration_days OR start_date).
- List field names in clarification_needed only when planning and they are missing.
- budget_level defaults to "mid_range" if not mentioned.
- pace defaults to "moderate" if not mentioned."""


CLARIFICATION_PROMPT = """\
The user wants to plan a trip but key information is missing.

What we already know:
{known_info}

Missing required fields: {missing_fields}

Write a friendly, conversational response asking for the missing information.
Ask at most 3 questions. Keep it short and warm.

Return ONLY valid JSON with this structure:
{{
  "message": "<friendly message asking for missing info>",
  "missing_fields": ["<field name>"],
  "open_questions": ["<specific question to ask the user>"]
}}"""


ITINERARY_GENERATION_PROMPT = """\
Create a detailed day-by-day travel itinerary based on the information below.

Trip Request:
{trip_request}

Traveler Profile:
{user_profile}

Web Search Results (from Tavily):
{tavily_context}

Generate a complete itinerary. Return ONLY valid JSON matching this schema exactly:
{schema}

Instructions:
1. Build morning/afternoon/evening blocks for every day.
2. Only cite specific facts (hours, prices, neighborhoods) if they appear in the search results above.
3. For any claim not in search results, add "[General knowledge]" to the notes field.
4. Include at least one rainy-day alternative under "alternatives".
5. logistics_notes must cover: transport between days, booking lead times, rough budget guidance.
6. List every assumption you made in "assumptions".
7. List unresolved open questions in "open_questions" (e.g., unknown hotel area preference).
8. Populate "sources" from the Tavily results you actually referenced."""


QUESTION_ANSWER_PROMPT = """\
Answer the travel question below using the search results provided.

Question: {question}
Additional context: {context}

Web Search Results (from Tavily):
{tavily_context}

Return ONLY valid JSON matching this schema exactly:
{schema}

Instructions:
1. Give a direct, clear answer.
2. List supporting evidence from search results as bullet points in supporting_points.
3. Note any assumptions or uncertainty in assumptions.
4. Suggest 1-3 useful follow-up questions in follow_up_questions.
5. Populate sources from the Tavily results you actually cited.
6. If search results are empty, answer from general knowledge and note it in assumptions."""


SCHEMA_REPAIR_PROMPT = """\
The previous response did not match the required JSON schema.

Previous response:
{original_response}

Required schema:
{schema}

Return ONLY valid JSON that exactly matches the schema above.
Use empty strings ("") for required string fields you cannot fill.
Use empty arrays ([]) for required list fields you cannot fill."""
