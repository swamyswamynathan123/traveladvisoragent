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
=== PERSONALIZATION CONTRACT ===
Before writing anything, you MUST honor ALL of the following for every day:
• Budget: {budget_level} — {budget_guidance}
• Pace: {pace} — {pace_description}
• Interests: {interests_list} — at least 60% of activities must directly reflect these interests
• Constraints: {constraints_list} — these are HARD rules, never violate them under any circumstance

=== TRIP REQUEST ===
{trip_request}

=== TRAVELER PROFILE ===
{user_profile}

=== WEB SEARCH RESULTS ===
{tavily_context}

=== HOTEL SEARCH RESULTS ===
{hotel_context}

=== ACTIVITY DEPTH RULES ===
For EVERY activity block:
1. Use the specific venue name — never "a local market", always e.g. "Mercado de San Miguel"
2. Write 2–3 sentences in notes: what it is, why it suits THIS traveler given their interests, one practical tip
3. Specify transport: exact metro line, bus number, or "10-min walk from [landmark]"
4. Set duration_hours to the recommended time at the venue
5. Include: best arrival time (e.g. "arrive before 9am to avoid queues"), book-ahead flag
6. For dining blocks: use specific restaurant names from search results. If none available, write "Find a [cuisine] restaurant near [neighborhood]" — never vague descriptions or "no specific details available"

=== LOGISTICS RULES ===
logistics_notes must cover:
• Exact transport between each day's areas (metro line numbers, bus routes, taxi estimate in local currency)
• Which activities need advance booking and how far ahead (e.g. "Book Alhambra tickets 2–3 weeks ahead")
• Rough daily spend estimate at {budget_level} level in local currency

=== MULTI-CITY RULES (apply when destination contains 2+ cities) ===
1. Cluster days by city — ALL days in the same city must be consecutive. Never split a city across non-consecutive days.
2. Order cities geographically to minimise backtracking (e.g. Madrid → Seville → Barcelona, not Madrid → Barcelona → Seville → Madrid).
3. Insert a dedicated travel day between each pair of cities. Use time_of_day "morning" for the departure block. The activity should name the exact transport: carrier, route, and journey time (e.g. "AVE high-speed train Madrid → Seville, Atocha station, ~2.5 hrs, book on Renfe.com"). Set notes to practical tips (book in advance, baggage, station location). Set duration_hours to the total door-to-door travel time.
4. Allocate remaining days to each city proportionally to the total trip length (e.g. 8 days across 3 cities → roughly 2–3 days each, adjusted for distance and highlights).
5. In logistics_notes, list the full inter-city transport sequence with estimated cost at {budget_level} tier.

Generate a complete itinerary. Return ONLY valid JSON matching this schema exactly:
{schema}

Additional instructions:
1. Build morning/afternoon/evening blocks for every day.
2. Include at least one rainy-day alternative under "alternatives".
3. Populate hotel_suggestions with 2–3 hotels per city at {budget_level} tier. Use hotel names from the Hotel Search Results section first; use training knowledge as fallback and mark notes as "[General knowledge — verify availability before booking]".
4. Populate restaurant_suggestions with 2–3 per city matching the traveler's interests and constraints. Use search results first; training knowledge marked "[General knowledge — verify current status before visiting]" as fallback.
5. List every assumption you made in "assumptions".
6. List unresolved questions in "open_questions".
7. Populate "sources" only from search results you actually referenced."""


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
7. Always give specific named recommendations — never vague phrases like "several restaurants can be found" or "a number of options are available." If search results contain specific names, use them. If search results lack specific names, draw on your training knowledge and mark each recommendation with "[General knowledge — verify current status]". NEVER say you could not find names or redirect the user to search engines — always provide actionable named places."""


SCHEMA_REPAIR_PROMPT = """\
The previous response did not match the required JSON schema.

Previous response:
{original_response}

Required schema:
{schema}

Return ONLY valid JSON that exactly matches the schema above.
Use empty strings ("") for required string fields you cannot fill.
Use empty arrays ([]) for required list fields you cannot fill."""


PERSONALIZATION_CHECK_PROMPT = """\
You are a quality reviewer for a travel itinerary.

The traveler requested:
• Interests: {interests_list}
• Budget: {budget_level} — {budget_guidance}
• Pace: {pace} — {pace_description}
• Constraints: {constraints_list}

Review the itinerary below and fix ONLY violations of the above. For each day, check:
1. Do activities reflect the stated interests? Replace mismatches with interest-aligned alternatives.
2. Are hotel/restaurant tiers consistent with {budget_level}? Correct any tier mismatches.
3. Does the number of activity blocks per day match the stated pace? Add or remove blocks as needed.
4. Are all constraints honored in every block? Fix any violations immediately.

Do NOT rewrite the whole plan — only change fields that violate the contract above.

Itinerary to review:
{itinerary}

Return the corrected itinerary as valid JSON matching this schema exactly:
{schema}"""
