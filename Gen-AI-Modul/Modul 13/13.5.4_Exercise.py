# exercise_4.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated
import operator, os
from dotenv import load_dotenv

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list, operator.add]

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

def chat_node(state: ChatState) -> dict:
    response = llm.invoke([SystemMessage(content="Anda adalah asisten ramah.")] + state["messages"])
    return {"messages": [response]}

# Setup Checkpointer
memory = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Compile dengan checkpointer
app = graph.compile(checkpointer=memory)

# Gunakan thread_id tetap untuk mempertahankan memori percakapan
config = {"configurable": {"thread_id": "user-session-xyz"}}

# --- Percakapan Sesi 1 ---
print("--- Sesi 1 ---")
query_1 = "Halo, nama saya Budi dan saya suka buah mangga."
res_1 = app.invoke({"messages": [HumanMessage(content=query_1)]}, config=config)
print("AI:", res_1["messages"][-1].content)

# --- Percakapan Sesi 2 (Melanjutkan thread_id yang sama) ---
print("\n--- Sesi 2 (Memori Berlanjut) ---")
query_2 = "Siapa nama saya dan apa buah kesukaan saya?"
res_2 = app.invoke({"messages": [HumanMessage(content=query_2)]}, config=config)
print("AI:", res_2["messages"][-1].content)