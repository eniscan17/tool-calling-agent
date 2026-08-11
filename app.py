"""
Streamlit UI for the tool-calling agent.

Run:
    streamlit run app.py
"""

import streamlit as st

import agent
import config
import ingest_kb
import kb
import llm

st.set_page_config(page_title="Tool-Calling Agent", page_icon="🛠️")


@st.cache_resource(show_spinner=False)
def _init_models():
    llm.initialize(load_embedding_model=True)
    return True


def _ensure_kb_ready():
    kb.init_db()
    return kb.count_chunks()


st.title("🛠️ Tool-Calling Agent")
st.caption("Runs 100% on-device with Microsoft Foundry Local — calculator, Wikipedia, "
           "date/time, and a local knowledge base as tools.")

with st.sidebar:
    st.header("Knowledge base")
    n_chunks = _ensure_kb_ready()
    st.write(f"Indexed chunks: **{n_chunks}**")
    if st.button("🔄 (Re)build knowledge base", use_container_width=True):
        status_box = st.empty()

        def _cb(stage, pct):
            status_box.info(stage if pct < 100 else f"{stage} ✓")

        with st.spinner("Indexing..."):
            try:
                count = ingest_kb.run_ingestion(progress_callback=_cb)
                st.success(f"Indexed {count} chunks.")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.subheader("Available tools")
    st.markdown("- 🧮 `calculate`\n- 📖 `search_wikipedia`\n- 🕒 `get_current_datetime`\n- 📚 `search_knowledge_base`")
    st.divider()
    st.caption(f"Chat model: `{config.CHAT_MODEL_ALIAS}`")

with st.spinner("Loading local models (first run downloads them, this may take a while)..."):
    _init_models()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
if "display_history" not in st.session_state:
    st.session_state.display_history = []

for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("tool_trace"):
            with st.expander("🔧 Tool calls"):
                for t in turn["tool_trace"]:
                    st.markdown(f"**{t['name']}**(`{t['arguments']}`)")
                    st.code(str(t["result"]), language="json")

question = st.chat_input("Ask me anything — I can calculate, search Wikipedia, check the time, or search my docs...")

if question:
    st.session_state.display_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({"role": "user", "content": question})

    trace = []

    def _on_tool_call(name, arguments, result):
        trace.append({"name": name, "arguments": arguments, "result": result})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = agent.ask(st.session_state.messages, on_tool_call=_on_tool_call)
        st.write(answer)
        if trace:
            with st.expander("🔧 Tool calls"):
                for t in trace:
                    st.markdown(f"**{t['name']}**(`{t['arguments']}`)")
                    st.code(str(t["result"]), language="json")

    st.session_state.display_history.append(
        {"role": "assistant", "content": answer, "tool_trace": trace}
    )
