"""
Central configuration for the tool-calling agent.

Uses the same Foundry Local runtime already installed for the local-rag-assistant
project (github.com/eniscan17/local-rag-assistant) — no new installs, no API keys,
no cloud costs.

CHAT_MODEL_ALIAS must be a model whose catalog entry reports
supports_tool_calling = True. Run this to check what's available on your
machine (see README "A real finding" section for why the flag alone isn't
enough — tool-calling still needs the deterministic pre-routing in agent.py
for full reliability):

    python -c "from foundry_local_sdk import Configuration, FoundryLocalManager; \
FoundryLocalManager.initialize(Configuration(app_name='check')); \
[print(m.alias, '->', m.supports_tool_calling) for m in FoundryLocalManager.instance.catalog.list_models()]"
"""

import os

CHAT_MODEL_ALIAS = "phi-4-mini"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

APP_NAME = "foundry_local_tool_agent"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
DATA_DIR = os.path.join(BASE_DIR, "data")
KB_DB_PATH = os.path.join(DATA_DIR, "agent_kb.sqlite3")

KB_TOP_K = 2
# Raised from 0.35 after a real false-positive: an unrelated question ("who is
# Eda") scored 0.386 and 0.355 against this small 2-document knowledge base —
# both above the old threshold — and got folded into the model's context as
# if it were relevant. With only two documents, coincidental word overlap
# ("Turing", "computing"-adjacent terms, etc.) is enough to clear a low bar.
# 0.5 was chosen to sit above that observed false-positive band; re-verify
# against a genuinely on-topic question (e.g. "How does tool calling work in
# this project?") after any change here, since raising the threshold too far
# risks the opposite failure — a real match getting silently dropped.
KB_MIN_RELEVANCE_SCORE = 0.5

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Use them when needed to "
    "answer questions accurately."
)

# Max tool-call round-trips per question, to prevent infinite loops if the
# model keeps requesting tools without ever producing a final answer.
MAX_TOOL_ROUNDS = 5

# Caps how long a single answer can be. Small on-device models can fall into
# a repetition loop (same paragraph over and over) — this bounds the damage.
MAX_RESPONSE_TOKENS = 400
