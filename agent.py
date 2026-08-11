"""
The agent loop: sends the conversation to the model with tool definitions,
executes any requested tool calls, feeds results back, and repeats until the
model produces a final answer (or MAX_TOOL_ROUNDS is hit).

Pattern follows Microsoft's official Foundry Local tool-calling tutorial.
"""

import json
import re

import config
import llm
from tools import TOOLS, TOOL_FUNCTIONS, calculate, get_current_datetime, search_knowledge_base

# Strip a small set of common lead-in phrases before checking whether what's
# left is a pure arithmetic expression.
_MATH_LEADIN_RE = re.compile(
    r"^\s*(what\s+is|what's|calculate|compute|solve)\b[:\s]*", re.IGNORECASE
)
_MATH_CHARS = set("0123456789+-*/(). ")


def _extract_pure_math(question: str):
    """
    If `question` is (after stripping a common lead-in phrase and a trailing
    '?') nothing but a math expression, return that expression. Otherwise
    return None.

    Small on-device models were unreliable at actually invoking the
    `calculate` tool for arithmetic — sometimes computing (and getting wrong)
    the answer in plain text instead of emitting a real tool call (a known
    limitation, not specific to one model — see README). For the one case
    that's both unambiguous and cheaply verifiable — pure arithmetic — this
    routes straight to the real calculator in code instead of trusting the
    model's judgment. This is a common production pattern: let the model
    decide for ambiguous cases, hard-route the deterministic ones.
    """
    text = question.strip().rstrip("?").strip()
    text = _MATH_LEADIN_RE.sub("", text).strip()
    if not text or not all(c in _MATH_CHARS for c in text):
        return None
    if not any(c.isdigit() for c in text) or not any(c in "+-*/" for c in text):
        return None
    return text


# Same reasoning as pure-math routing above: "what time/date is it" is
# unambiguous and cheap to detect with keywords, so route it deterministically
# instead of trusting the model to call get_current_datetime.
_DATETIME_RE = re.compile(
    r"\b(what\s+(time|date|day)\b|current\s+(time|date)|today'?s?\s+date|"
    r"what\s+day\s+is\s+it)\b",
    re.IGNORECASE,
)


def _looks_like_datetime_question(question: str) -> bool:
    return bool(_DATETIME_RE.search(question))


def _cut_at_first_repeat(text: str) -> str:
    """
    Small on-device models occasionally degenerate into repeating the same
    paragraph forever. max_tokens (see llm.py) bounds *how much* gets
    generated; this bounds what the user actually sees — walk paragraphs in
    order and stop at the first one that's a near-duplicate of an earlier one.
    """
    paragraphs = text.split("\n\n")
    seen = set()
    kept = []
    for para in paragraphs:
        key = para.strip().lower()
        if key and key in seen:
            break
        if key:
            seen.add(key)
        kept.append(para)
    return "\n\n".join(kept).strip()


def ask(messages, on_tool_call=None):
    """
    messages: full conversation so far (list of {role, content} dicts),
              ending with the latest user message.
    on_tool_call: optional callback(name, arguments, result) — called each
                  time a tool is actually executed, so a UI can show a trace.

    Returns the assistant's final text answer. Mutates `messages` in place
    to include the tool-call round trips, so the caller can keep the full
    conversation for the next turn.
    """
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    expression = _extract_pure_math(last_user_msg)
    if expression is not None:
        result = calculate(expression)
        if on_tool_call:
            on_tool_call("calculate", {"expression": expression}, result)
        if "error" in result:
            final_answer = f"I couldn't evaluate that: {result['error']}"
        else:
            final_answer = f"{result['expression']} = {result['result']}"
        messages.append({"role": "assistant", "content": final_answer})
        return final_answer

    if _looks_like_datetime_question(last_user_msg):
        result = get_current_datetime()
        if on_tool_call:
            on_tool_call("get_current_datetime", {}, result)
        final_answer = f"Right now it's {result['readable']}."
        messages.append({"role": "assistant", "content": final_answer})
        return final_answer

    # Same reliability problem as math/datetime, different fix: rather than
    # trying to keyword-detect "is this question about the knowledge base"
    # (too fuzzy — almost any phrasing could be), always check it silently.
    # It's a local, cheap, embedding-similarity lookup, so trying it costs
    # nothing when it doesn't match. If it finds something relevant, fold the
    # result into the user's message as context (the same augmentation
    # pattern local-rag-assistant uses) so the model has it regardless of
    # whether it would have decided to call search_knowledge_base itself.
    kb_result = search_knowledge_base(last_user_msg)
    if "matches" in kb_result:
        if on_tool_call:
            on_tool_call("search_knowledge_base", {"query": last_user_msg}, kb_result)
        context = "\n\n".join(f"[{m['source']}] {m['content']}" for m in kb_result["matches"])
        for m in reversed(messages):
            if m["role"] == "user":
                m["content"] = (
                    f"{m['content']}\n\n"
                    f"(Relevant internal documentation found automatically:\n{context})"
                )
                break

    response = llm.complete_chat(messages, tools=TOOLS)
    choice = response.choices[0].message

    rounds = 0
    while choice.tool_calls and rounds < config.MAX_TOOL_ROUNDS:
        rounds += 1

        messages.append({
            "role": "assistant",
            "content": choice.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.tool_calls
            ],
        })

        for tool_call in choice.tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            func = TOOL_FUNCTIONS.get(name)
            if func is None:
                result = {"error": f"Unknown tool '{name}'"}
            else:
                try:
                    result = func(**arguments)
                except Exception as e:
                    result = {"error": f"Tool '{name}' raised an error: {e}"}

            if on_tool_call:
                on_tool_call(name, arguments, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        response = llm.complete_chat(messages, tools=TOOLS)
        choice = response.choices[0].message

    final_answer = _cut_at_first_repeat(choice.content or "(no answer produced)")
    messages.append({"role": "assistant", "content": final_answer})
    return final_answer
