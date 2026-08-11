"""
Command-line version of the agent — useful for quick testing without
opening a browser. For the full UI with a visible tool-call trace, use
`streamlit run app.py` instead.

Run:
    python cli.py
"""

import agent
import config
import kb
import llm


def _on_tool_call(name, arguments, result):
    print(f"  [tool] {name}({arguments}) -> {result}")


def main():
    kb.init_db()
    if kb.count_chunks() == 0:
        print("Knowledge base is empty — run `python ingest_kb.py` first if you "
              "want the search_knowledge_base tool to have anything to find.\n")

    print("Loading models (first run downloads them, this can take a while)...")
    llm.initialize(load_embedding_model=True)
    print("Ready. Type 'quit' to exit.\n")

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})
        answer = agent.ask(messages, on_tool_call=_on_tool_call)
        print(f"Agent: {answer}\n")

    llm.shutdown()
    print("Goodbye!")


if __name__ == "__main__":
    main()
