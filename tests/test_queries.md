# Test queries

Run these in the app and fill in the Result column — same practice as
local-rag-assistant's test log.

| # | Question | Expected tool | Result |
|---|----------|----------------|--------|
| 1 | What is 245 * 38? | `calculate` | ✅ Pass — after adding deterministic pre-routing (see below) |
| 2 | Who was Ada Lovelace? | `search_wikipedia` | ⚠️ Not verified against a real tool call — model answers from memory, no tool trace shown (known limitation, see README) |
| 3 | What time is it right now? | `get_current_datetime` | ✅ Pass — after adding deterministic pre-routing |
| 4 | How does tool calling work in this project? | `search_knowledge_base` | ✅ Pass — after adding silent auto-context injection, correctly grounded in tool_calling.txt |
| 5 | What's the weather in Paris? | none — no weather tool exists; agent should say it can't answer, not guess | ✅ Pass — declined and suggested a real weather source instead of guessing |
| 6 | (chain) What is 12 * 12, and also who was Alan Turing? | two tools in one turn: `calculate` then `search_wikipedia` | ⚠️ Partial — answer was correct and no longer loops (repeat-guard works), but neither tool actually fired: math was embedded in a larger sentence so the deterministic router didn't match it (by design — see below), and the KB auto-check fired a weak, barely-above-threshold false-positive match instead of nothing |

## Findings & decisions

- **qwen2.5-0.5b**, **phi-3.5-mini**, and **phi-4-mini** were all tried as
  `CHAT_MODEL_ALIAS`. `phi-3.5-mini` turned out not to support tool calling
  at all (`supports_tool_calling: False` — a wrong initial guess, corrected
  by actually querying the catalog). Both `qwen2.5-0.5b` and `phi-4-mini`
  report `True` but were unreliable in practice: for arithmetic and
  date/time questions, they sometimes answered in plain text (with wrong
  numbers, in the calculator's case) instead of emitting a real tool call.
- **Fix:** pure-math and date/time questions are now detected in `agent.py`
  with a cheap, unambiguous check and routed straight to the real Python
  function, bypassing the model's judgment entirely for those two cases.
  `search_wikipedia` and `search_knowledge_base` don't have an equally safe
  deterministic detector, so they still depend on the model actually
  choosing to call the tool — call this out if demoing live.
