# exercise_3.py
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
import os
from dotenv import load_dotenv

load_dotenv()

class PlanState(TypedDict):
    input_query: str
    steps: List[str]
    current_step_index: int
    execution_results: List[str]
    final_answer: str

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

def planner_node(state: PlanState) -> dict:
    """Node Planner yang menghasilkan daftar langkah dalam format list Python / baris."""
    prompt = f"Buat daftar 3 langkah singkat untuk menyelesaikan tugas ini: '{state['input_query']}'. Pisahkan setiap langkah dengan baris baru tanpa nomor."
    res = llm.invoke([HumanMessage(content=prompt)])
    steps = [s.strip() for s in res.content.split("\n") if s.strip()]
    print(f"[Planner] Langkah dibuat: {steps}")
    return {"steps": steps, "current_step_index": 0, "execution_results": []}

def executor_node(state: PlanState) -> dict:
    """Node Executor yang menjalankan langkah saat ini berdasarkan index."""
    idx = state["current_step_index"]
    step = state["steps"][idx]
    
    prompt = f"Jalankan atau jawab langkah berikut secara singkat: {step}"
    res = llm.invoke([HumanMessage(content=prompt)])
    
    print(f"[Executor] Menyelesaikan langkah {idx+1}: {step} -> {res.content}")
    
    new_results = state["execution_results"] + [res.content]
    return {
        "current_step_index": idx + 1,
        "execution_results": new_results
    }

def should_continue(state: PlanState):
    if state["current_step_index"] < len(state["steps"]):
        return "continue"
    return "finish"

def synthesizer_node(state: PlanState) -> dict:
    combined = "\n".join(state["execution_results"])
    return {"final_answer": f"Selesai! Hasil gabungan:\n{combined}"}

graph = StateGraph(PlanState)
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("synthesizer", synthesizer_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue, {"continue": "executor", "finish": "synthesizer"})
graph.add_edge("synthesizer", END)

app = graph.compile()

res = app.invoke({
    "input_query": "Bagaimana cara membuat secangkir kopi?",
    "steps": [],
    "current_step_index": 0,
    "execution_results": [],
    "final_answer": ""
})

print("\nFinal Result:\n", res["final_answer"])