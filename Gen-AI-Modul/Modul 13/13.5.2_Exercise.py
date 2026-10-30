# exercise_2.py
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from typing import TypedDict, Annotated
import operator, json, os, math
from dotenv import load_dotenv

load_dotenv()

def calculator(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt}))
    except Exception as e:
        return f"Error: {e}"

TOOL_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Menghitung ekspresi matematika.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    }
}]

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_call_count: int

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
).bind_tools(TOOL_SCHEMAS)

def agent_node(state: AgentState) -> dict:
    res = llm.invoke([SystemMessage(content="Gunakan kalkulator untuk menghitung.")] + state["messages"])
    return {"messages": [res]}

def tool_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    results = []
    count_increment = 0
    for tc in last_msg.tool_calls:
        if tc["name"] == "calculator":
            out = calculator(**tc["args"])
            results.append(ToolMessage(content=out, tool_call_id=tc["id"]))
            count_increment += 1
    return {"messages": results, "tool_call_count": state["tool_call_count"] + count_increment}

def should_continue(state: AgentState):
    # Cek batas iterasi (maksimal 8 tool calls)
    if state["tool_call_count"] >= 8:
        return "limit_reached"
    
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

def limit_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content="Reached step limit.")]}

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("limit_reached", limit_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "limit_reached": "limit_reached", END: END})
graph.add_edge("tools", "agent")
graph.add_edge("limit_reached", END)

app = graph.compile()

res = app.invoke({
    "messages": [HumanMessage(content="Hitung 1+1")],
    "tool_call_count": 0
})

for m in res["messages"]:
    if isinstance(m, AIMessage):
        print("Output:", m.content)