# exercise_1.py
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from typing import TypedDict, Annotated
import operator, json, os
from dotenv import load_dotenv

load_dotenv()

# --- 1. Definisi Tools ---
def web_search(query: str) -> str:
    """Mock web search yang mengembalikan hasil tetap."""
    return json.dumps([
        {"title": f"Hasil Riset tentang {query}", "snippet": f"{query} adalah topik yang sangat penting dalam perkembangan teknologi AI modern saat ini."}
    ])

def summarise(text: str) -> str:
    """Meringkas teks menggunakan LLM terpisah."""
    client = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    res = client.invoke([HumanMessage(content=f"Ringkas teks berikut secara singkat: {text}")])
    return res.content

# Daftar catatan yang tersimpan secara global untuk simulasi save_note
SAVED_NOTES = []

def save_note(content: str) -> str:
    """Menyimpan catatan ke dalam sebuah list."""
    SAVED_NOTES.append(content)
    return f"Catatan berhasil disimpan. Total catatan: {len(SAVED_NOTES)}"

TOOLS = {
    "web_search": web_search,
    "summarise": summarise,
    "save_note": save_note,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Mencari informasi di web berdasarkan query.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise",
            "description": "Meringkas teks panjang menjadi ringkasan yang padat.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Menyimpan catatan hasil riset ke dalam daftar catatan.",
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
        },
    },
]

# --- 2. State & LLM ---
class ResearchState(TypedDict):
    messages: Annotated[list, operator.add]

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
).bind_tools(TOOL_SCHEMAS)

SYSTEM = SystemMessage(content="Anda adalah agen riset. Cari topik menggunakan web_search, ringkas hasilnya menggunakan summarise, lalu simpan pentingnya menggunakan save_note.")

def agent_node(state: ResearchState) -> dict:
    res = llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [res]}

def tool_node(state: ResearchState) -> dict:
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        fn = TOOLS[tc["name"]]
        output = fn(**tc["args"])
        results.append(ToolMessage(content=output, tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state: ResearchState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# --- 3. Build Graph ---
graph = StateGraph(ResearchState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()

# --- 4. Eksekusi ---
res = app.invoke({"messages": [HumanMessage(content="Tolong teliti tentang tren arsitektur RAG, ringkas, dan simpan catatannya.")]})
for m in res["messages"]:
    if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
        print("Final Answer:\n", m.content)
print("\nCatatan Tersimpan:", SAVED_NOTES)