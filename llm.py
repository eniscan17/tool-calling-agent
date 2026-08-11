"""
Thin wrapper around the Foundry Local SDK — same pattern proven in
local-rag-assistant/llm.py, extended with tool-calling support per
Microsoft's official tutorial (see config.py for the link).
"""

from foundry_local_sdk import Configuration, FoundryLocalManager

import config

_manager = None
_chat_model = None
_chat_client = None
_embedding_model = None
_embedding_client = None


def initialize(progress_callback=None, load_embedding_model=True):
    """
    Start the Foundry Local runtime and load the chat model (and, optionally,
    the embedding model used by the knowledge-base tool). Safe to call more
    than once — subsequent calls are no-ops.
    """
    global _manager, _chat_model, _chat_client, _embedding_model, _embedding_client

    if _manager is not None:
        return

    def report(stage, pct):
        if progress_callback:
            progress_callback(stage, pct)
        else:
            print(f"\r{stage}: {pct:.1f}%", end="", flush=True)

    cfg = Configuration(app_name=config.APP_NAME)
    FoundryLocalManager.initialize(cfg)
    _manager = FoundryLocalManager.instance

    _chat_model = _manager.catalog.get_model(config.CHAT_MODEL_ALIAS)
    _chat_model.download(lambda p: report("Downloading chat model", p))
    _chat_model.load()
    _chat_client = _chat_model.get_chat_client()

    if load_embedding_model:
        _embedding_model = _manager.catalog.get_model(config.EMBEDDING_MODEL_ALIAS)
        _embedding_model.download(lambda p: report("Downloading embedding model", p))
        _embedding_model.load()
        _embedding_client = _embedding_model.get_embedding_client()

    if progress_callback:
        progress_callback("Models ready", 100)


def shutdown():
    global _manager, _chat_model, _chat_client, _embedding_model, _embedding_client
    if _chat_model:
        _chat_model.unload()
    if _embedding_model:
        _embedding_model.unload()
    _manager = _chat_model = _chat_client = _embedding_model = _embedding_client = None


def complete_chat(messages, tools=None):
    """Send a chat completion request (non-streaming — needed to inspect tool_calls)."""
    if _chat_client is None:
        raise RuntimeError("Call llm.initialize() first.")
    kwargs = {"tools": tools} if tools else {}
    # Small on-device models can occasionally fall into a repetition loop
    # (the same paragraph generated over and over). Capping response length
    # bounds how bad that can get. Not every SDK version accepts max_tokens
    # on this call, so fail open rather than crash if it doesn't.
    try:
        return _chat_client.complete_chat(messages, max_tokens=config.MAX_RESPONSE_TOKENS, **kwargs)
    except TypeError:
        return _chat_client.complete_chat(messages, **kwargs)


def embed_query(text: str):
    if _embedding_client is None:
        raise RuntimeError("Embedding model not loaded — call initialize(load_embedding_model=True).")
    response = _embedding_client.generate_embedding(text)
    return response.data[0].embedding


def embed_texts(texts):
    if _embedding_client is None:
        raise RuntimeError("Embedding model not loaded — call initialize(load_embedding_model=True).")
    response = _embedding_client.generate_embeddings(texts)
    return [item.embedding for item in response.data]
