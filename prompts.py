SYSTEM_PROMPT = """You are an expert travel advisor with deep knowledge of destinations worldwide.
You help users plan detailed day-by-day itineraries and answer travel questions accurately.

Core rules:
- Prefer facts from Tavily search results when available; they are more current.
- For any claim from your training knowledge (place names, restaurant names, descriptions), mark it as "[General knowledge — verify before traveling]".
- Never invent URLs, phone numbers, admission prices, or booking links unless they appear in search results.
- If Tavily results conflict, summarize the conflict and flag it as an open question.
- Be explicit about every assumption you make."""


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
- intent is "planning" ONLY if the user explicitly asks for an itinerary, a trip plan, or says "plan my trip". Examples: "Plan a 5-day trip to Rome", "Create an itinerary for Tokyo".
- intent is "question" for ANY other travel-related message — weather, best time to visit, visa, restaurants, transport, costs, safety, recommendations, comparisons. When in doubt, use "question".
- Examples of "question": "What is the best time to visit Barcelona?", "Which restaurants are vegetarian in Madrid?", "Do I need a visa for Japan?", "How do I get from Paris to Lyon?", "What should I pack for Iceland in winter?"
- For planning, critical fields are destination AND (duration_days OR start_date). List only truly missing fields in clarification_needed.
- For question intent, always set travel_question to the full user message text. Never leave it null.
- clarification_needed must be empty [] for question intent.
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
2. For every activity block, always write 1-2 sentences in the notes field describing what the attraction is and why it is worth visiting. Never leave notes empty for sightseeing or cultural activities.
3. Only add "[General knowledge]" at the end of the notes sentence when citing specific operational facts (admission prices, opening hours, booking links) that do NOT appear in the search results. Do not add the marker to descriptive sentences about the attraction itself.
4. For dining activities: use specific restaurant names from the search results (e.g., "Dinner at Teresa Carles"). If no restaurant names appear in search results, set the activity to "Find a [cuisine] restaurant near [neighborhood]" and leave notes empty — NEVER write sentences like "no specific details available", "verify before traveling", or "based on general knowledge" in dining blocks. Do not explain the lack of data; just use the "Find a..." format.
5. Include at least one rainy-day alternative under "alternatives".
6. Populate "hotel_suggestions" with 2-3 hotels per city. Use hotel names from search results when available; otherwise use training knowledge and mark the notes field as "[General knowledge — verify availability before booking]". Each suggestion must include name, city, neighborhood (if known), budget_level (budget/mid_range/luxury), and a 1-sentence description.
7. logistics_notes must cover: transport between days, booking lead times, rough budget guidance.
8. List every assumption you made in "assumptions".
9. List unresolved open questions in "open_questions" (e.g., unknown hotel area preference).
10. Populate "sources" from the Tavily results you actually referenced."""


QUESTION_ANSWER_PROMPT = """\
Answer the travel question below using the search results provided.

Question: {question}
Additional context: {context}

Web Search Results (from Tavily):
{tavily_context}

Return ONLY valid JSON matching this schema exactly:
{schema}

Instructions:
1. Give a direct, clear answer using only information present in the search results above.
2. List supporting evidence from search results as bullet points in supporting_points.
3. Note any assumptions or uncertainty in assumptions.
4. Suggest 1-3 useful follow-up questions in follow_up_questions.
5. Populate sources from the Tavily results you actually cited.
6. If search results contain specific names (restaurants, hotels, attractions), list them by name in the answer. NEVER write "several restaurants can be found" or similar vague phrases — always name specific places.
7. If search results lack specific names, use your training knowledge to provide named recommendations. Mark each training-knowledge item with "[General knowledge — verify current status before visiting]" in the answer. Never respond with "search results did not return names" or redirect the user to Google Maps — always give actionable named recommendations, sourced from search results or training knowledge."""


SCHEMA_REPAIR_PROMPT = """\
The previous response did not match the required JSON schema.

Previous response:
{original_response}

Required schema:
{schema}

Return ONLY valid JSON that exactly matches the schema above.
Use empty strings ("") for required string fields you cannot fill.
Use empty arrays ([]) for required list fields you cannot fill."""
