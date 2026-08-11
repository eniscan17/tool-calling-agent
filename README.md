# Tool-Calling Agent (Foundry Local)

A local AI agent that decides, on its own, which tool to use to answer a
question — a calculator, Wikipedia search, the current date/time, or a
small local knowledge base — instead of only generating text from its
training data. Runs entirely on-device via Microsoft Foundry Local, the
same runtime used in the companion project
[local-rag-assistant](../local-rag-assistant).

**Why this project exists:** local-rag-assistant always does one fixed
step (retrieve → answer). This project generalizes that idea: the model
plans which action to take at each step and can chain multiple tools
together before answering — the core idea behind AI agents.

---

## 0. Prerequisites

If you already set up **local-rag-assistant**, you already have everything
this project needs system-wide (Homebrew, the `foundry` CLI, Python 3.11+).
You only need a fresh Python virtual environment for this project's own
dependencies.

If you're starting fresh (skipped local-rag-assistant), see that project's
README section 0 and 1 first — install Homebrew and `brew install
foundrylocal` before continuing here.

## 1. Set up this project

In Terminal, in this folder (`tool-calling-agent/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Checkpoint:** no red errors; prompt shows `(.venv)`.

## 2. Build the small knowledge base

```bash
python ingest_kb.py
```

Downloads the embedding model on first run (skipped if you already have it
from local-rag-assistant — same local cache). Ends with `Indexed N chunks...`.

## 3. Run it

**Web UI (recommended — shows a live trace of which tools were called):**
```bash
streamlit run app.py
```

**Or command line:**
```bash
python cli.py
```

Try asking:
- `What is 245 * 38?` → uses `calculate`
- `Who was Ada Lovelace?` → uses `search_wikipedia`
- `What time is it right now?` → uses `get_current_datetime`
- `How does tool calling work in this project?` → uses `search_knowledge_base`
- `What's the weather in Paris?` → the agent has no weather tool, so it
  should say it can't answer that rather than guessing — a good test of
  honest failure handling.

---

## Project structure

```
config.py        model names, system prompt, tool-call round limit
llm.py            Foundry Local SDK wrapper (chat + embedding clients)
tools.py          tool JSON schemas + their real implementations
agent.py          the tool-call loop: ask model -> run tools -> feed results back -> repeat
kb.py             tiny SQLite-backed knowledge base (same pattern as local-rag-assistant)
ingest_kb.py      builds the knowledge base from documents/*.txt
app.py            Streamlit UI with a visible tool-call trace
cli.py            plain terminal version, same agent logic
documents/        the two docs the search_knowledge_base tool can search
```

## How the loop works

1. **Deterministic pre-routing first** (see "A real finding" below): if the
   question is unambiguously pure arithmetic, or unambiguously a date/time
   question, `agent.py` calls `calculate` / `get_current_datetime` directly
   in code and skips the model entirely for that turn.
2. Otherwise, the local knowledge base is checked silently; if there's a
   relevant match, it's added as context to the user's message.
3. The app sends the (possibly augmented) conversation plus the list of
   available tools (as JSON schemas) to the model.
4. If the model's response includes `tool_calls`, the app looks up the
   matching Python function in `tools.py`, runs it with the arguments the
   model provided, and appends the result back into the conversation with
   role `"tool"`.
5. The updated conversation is sent back to the model. This repeats (up to
   `MAX_TOOL_ROUNDS` in `config.py`) until the model responds with a plain
   answer instead of another tool call.

## A real finding: tool-calling metadata vs. actual reliability

Foundry Local's catalog exposes a `supports_tool_calling` flag per model.
In practice, testing this project against **three** flagged-as-supported
models (`qwen2.5-0.5b`, `phi-3.5-mini` — which turned out to report `False`,
not a bug, just a wrong first guess — and `phi-4-mini`, which reports
`True`), tool-calling was **not reliable**: for a clearly tool-requiring
question (e.g. a large multiplication), the models sometimes wrote out a
plausible-looking but wrong answer as plain text instead of emitting a real
`tool_calls` response — no different, from the outside, than a model that
doesn't support tool calling at all.

To verify which models on your machine actually claim support before
picking one, run:

```bash
python -c "from foundry_local_sdk import Configuration, FoundryLocalManager; \
FoundryLocalManager.initialize(Configuration(app_name='check')); \
[print(m.alias, '->', m.supports_tool_calling) for m in FoundryLocalManager.instance.catalog.list_models()]"
```

Three different fixes were used here, matched to how detectable each case is:

- **`calculate` and `get_current_datetime`** — pure math and date/time
  questions are unambiguous and cheap to detect with a regex, so `agent.py`
  routes them straight to the real Python function, bypassing the model's
  judgment entirely.
- **`search_knowledge_base`** — almost any phrasing could be a question
  about this project, so a keyword detector would be unreliable. Instead,
  every question is silently checked against the (small, local, cheap)
  knowledge base first; if there's a relevant match, it's folded into the
  user's message as context automatically — the same augmentation pattern
  local-rag-assistant uses — regardless of whether the model would have
  decided to call the tool itself.
- **`search_wikipedia`** — genuinely open-ended (could be about anything),
  so there's no cheap way to pre-check it. This one still depends entirely
  on the model actually choosing to call it, and inherits the same
  reliability caveat described above — worth calling out explicitly if you
  demo this live.

The general lesson: don't trust a small model's judgment for a case you can
already resolve — or at least silently check — in code.

One more small-model failure mode showed up during testing: on longer,
open-ended answers (e.g. combined questions like "what is 12*12 and who was
Alan Turing"), the model occasionally fell into a **repetition loop**,
generating the same paragraph over and over instead of stopping. Two
independent guardrails handle this: `llm.py` caps response length
(`MAX_RESPONSE_TOKENS` in `config.py`), and `agent.py`'s
`_cut_at_first_repeat` walks the answer paragraph by paragraph and truncates
it at the first near-duplicate, regardless of what the model actually
generated. Same underlying lesson as the tool-routing fixes above: bound
small-model failure modes in code rather than hoping the model avoids them.

## Notes on the tools

- **`calculate`** only allows digits, `+ - * / ( )` and spaces before
  evaluating — this blocks code-injection attempts through the expression
  string.
- **`search_wikipedia`** calls Wikipedia's public REST API directly (no
  API key). If the network is unavailable it returns a clean error instead
  of crashing, and the agent is instructed to say so rather than guess.
- **`search_knowledge_base`** reuses the exact retrieval pattern from
  local-rag-assistant (embed the query, cosine-similarity search, a
  minimum relevance threshold) — see that project for more detail on why
  the threshold matters.

## Extending it

Adding a new tool takes two steps: add its JSON schema to `TOOLS` in
`tools.py`, write the Python function, and add it to `TOOL_FUNCTIONS`.
Nothing in `agent.py` needs to change.
