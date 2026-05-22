# Budget Estimator Design

## Goal

Add a collapsible budget estimate card to the itinerary view, showing itemised USD cost ranges for flights, hotels, food, and activities — with both a per-person total and a group total scaled to the traveller type.

## Architecture

Three files change; nothing else in the dependency chain is affected.

```
schemas.py ← tools.py ← prompts.py ← graph.py ─┐
                                                  ├─ app.py
                                       (unchanged)┘
```

| File | Change |
|------|--------|
| `schemas.py` | Add `BudgetLineItem` and `BudgetEstimate` models; add `budget_estimate: Optional[BudgetEstimate]` to `ItineraryResponse` |
| `prompts.py` | Add `BUDGET ESTIMATE RULES` block to `ITINERARY_GENERATION_PROMPT` |
| `app.py` | Render budget card in `_render_itinerary()` after the header |

`graph.py` is **not modified** — it passes `ItineraryResponse.model_json_schema()` to the prompt dynamically, so the new field appears in the schema automatically. All existing tests continue to pass because `budget_estimate` is `Optional`.

## Component 1 — Schema (`schemas.py`)

```python
class BudgetLineItem(BaseModel):
    category: str          # "Flights", "Hotels", "Food & Dining", "Activities"
    low_usd: int
    high_usd: int
    notes: Optional[str] = None   # e.g. "Round-trip from NYC, economy"

class BudgetEstimate(BaseModel):
    line_items: List[BudgetLineItem]   # exactly 4 items in order: Flights, Hotels, Food & Dining, Activities
    per_person_low_usd: int
    per_person_high_usd: int
    group_low_usd: int
    group_high_usd: int
    num_travelers: int     # 1=solo, 2=couple, 3=family, 4=group
    currency_note: str     # "All estimates in USD. Actual costs vary by season and availability."
```

Add to `ItineraryResponse`:
```python
budget_estimate: Optional[BudgetEstimate] = None
```

The LLM fills `per_person_*` as the sum of all four line item lows/highs, and `group_*` as `per_person × num_travelers`. The app renders the group row only when `num_travelers > 1`.

## Component 2 — Prompt changes (`prompts.py`)

Append this block to `ITINERARY_GENERATION_PROMPT` immediately before the final "Generate a complete itinerary" line:

```
=== BUDGET ESTIMATE RULES ===
Populate budget_estimate with realistic USD cost ranges for this trip:
• line_items: exactly 4 items in this order:
  1. "Flights" — round-trip per person. Use the origin field from the trip request above if known; otherwise assume a major international hub.
  2. "Hotels" — total accommodation cost per person for all nights at {budget_level} tier.
  3. "Food & Dining" — total food cost per person for all days at {budget_level} tier.
  4. "Activities" — total admission fees, tours, and local transport per person.
• per_person_low_usd / per_person_high_usd: sum of all four line item lows/highs.
• group_low_usd / group_high_usd: per_person × num_travelers.
• num_travelers: 1 for solo, 2 for couple, 3 for family, 4 for group.
• currency_note: "All estimates in USD. Actual costs vary by season and availability."
• Use Tavily search results for pricing signals where available; fall back to training knowledge.
```

The `{origin}` placeholder is already available in the prompt context from `trip_request`. No new format variables are required.

## Component 3 — UI rendering (`app.py`)

Insert immediately after `st.subheader(header)` in `_render_itinerary()`:

```python
budget = data.get("budget_estimate")
if budget:
    with st.expander("💰 Budget Estimate", expanded=True):
        for item in budget.get("line_items", []):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(item.get("category", ""))
                if item.get("notes"):
                    st.caption(item["notes"])
            with col2:
                low = item.get("low_usd", 0)
                high = item.get("high_usd", 0)
                st.markdown(f"${low:,}–${high:,}")
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Total (per person)**")
        with col2:
            plow = budget.get("per_person_low_usd", 0)
            phigh = budget.get("per_person_high_usd", 0)
            st.markdown(f"**${plow:,}–${phigh:,}**")
        num = budget.get("num_travelers", 1)
        if num > 1:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Total ({num} travelers)**")
            with col2:
                glow = budget.get("group_low_usd", 0)
                ghigh = budget.get("group_high_usd", 0)
                st.markdown(f"**${glow:,}–${ghigh:,}**")
        if budget.get("currency_note"):
            st.caption(budget["currency_note"])
```

The card is collapsible (`st.expander`), expanded by default. If the LLM omits `budget_estimate` (e.g., schema repair fallback), the block is silently skipped.

## Data Flow

```
user submits trip request
    → _generate_itinerary() builds prompt including BUDGET ESTIMATE RULES
    → LLM returns ItineraryResponse with budget_estimate populated
    → final_response["data"] contains budget_estimate dict
    → _render_itinerary() reads budget_estimate and renders expander card
```

## Testing

### `tests/test_schemas.py` — new test

```python
def test_itinerary_response_accepts_budget_estimate():
    from schemas import BudgetEstimate, BudgetLineItem, ItineraryResponse, DayPlan, TimeBlock
    estimate = BudgetEstimate(
        line_items=[
            BudgetLineItem(category="Flights", low_usd=400, high_usd=600),
            BudgetLineItem(category="Hotels", low_usd=350, high_usd=500),
            BudgetLineItem(category="Food & Dining", low_usd=150, high_usd=200),
            BudgetLineItem(category="Activities", low_usd=80, high_usd=120),
        ],
        per_person_low_usd=980, per_person_high_usd=1420,
        group_low_usd=1960, group_high_usd=2840,
        num_travelers=2,
        currency_note="All estimates in USD.",
    )
    resp = ItineraryResponse(
        destination="Paris", duration="5 days",
        itinerary=[DayPlan(day_number=1, blocks=[TimeBlock(time_of_day="morning", activity="Arrive")])],
        logistics_notes="By train.",
        budget_estimate=estimate,
    )
    assert resp.budget_estimate.num_travelers == 2
    assert resp.budget_estimate.line_items[0].category == "Flights"
```

### `tests/test_graph_nodes.py` — update existing fixture

Update `_planning_state()` mock `ItineraryResponse` to include a minimal `BudgetEstimate` so existing tests continue to validate correctly against the updated schema.

## Error Handling

`budget_estimate` is `Optional` on `ItineraryResponse`. If the LLM fails to populate it (schema repair path, structured output failure), it defaults to `None` and the UI silently skips the card. No new error paths are introduced.

## Out of Scope

- Multi-currency display (always USD)
- Dynamic pricing via Expedia or other booking APIs
- Per-day cost breakdown inside each day expander
- User-editable budget inputs
