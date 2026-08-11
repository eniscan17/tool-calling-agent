"""
Tool definitions (JSON schemas the model sees) and their real Python
implementations. Pattern follows Microsoft's official Foundry Local
tool-calling tutorial — see config.py for the link.
"""

import datetime

import requests

import config
import kb
import llm

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '245 * 38' or '(12 + 8) / 4'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia and return a short summary of the best-matching article. Use for general knowledge, people, places, historical or factual questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for on Wikipedia."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time. Use for any question about 'today', 'now', or the current date/time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search this project's local knowledge base (docs about tool calling and this agent's own architecture). Use when the question is about how this agent itself works.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up."}
                },
                "required": ["query"],
            },
        },
    },
]


def calculate(expression: str):
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return {"error": "Only basic arithmetic (+ - * / and parentheses) is allowed."}
    try:
        # eval is safe here: input is restricted to digits/operators/parens above.
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {e}"}


def search_wikipedia(query: str):
    try:
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": 1,
            },
            timeout=8,
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("query", {}).get("search", [])
        if not hits:
            return {"error": f"No Wikipedia article found for '{query}'."}
        title = hits[0]["title"]

        summary_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            timeout=8,
        )
        summary_resp.raise_for_status()
        data = summary_resp.json()
        return {
            "title": data.get("title", title),
            "summary": data.get("extract", "No summary available."),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
    except requests.RequestException as e:
        return {"error": f"Could not reach Wikipedia: {e}"}


def get_current_datetime():
    now = datetime.datetime.now()
    return {"iso": now.isoformat(timespec="seconds"), "readable": now.strftime("%A, %B %d, %Y %H:%M")}


def search_knowledge_base(query: str):
    query_embedding = llm.embed_query(query)
    results = kb.top_chunks(query_embedding, top_k=config.KB_TOP_K)
    if not results or results[0]["score"] < config.KB_MIN_RELEVANCE_SCORE:
        return {"error": "Nothing relevant found in the local knowledge base."}
    return {
        "matches": [
            {"source": r["source"], "content": r["content"], "score": round(r["score"], 3)}
            for r in results
        ]
    }


TOOL_FUNCTIONS = {
    "calculate": calculate,
    "search_wikipedia": search_wikipedia,
    "get_current_datetime": get_current_datetime,
    "search_knowledge_base": search_knowledge_base,
}
