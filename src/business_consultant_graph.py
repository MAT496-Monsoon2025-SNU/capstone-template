import os
import uuid
from typing import TypedDict, Dict, Any
from dotenv import load_dotenv

# If your langgraph/langchain imports differ, adapt them accordingly.
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

# Load .env API keys if present (works silently if no .env)
load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# --- State ---
class BizState(TypedDict):
    business_description: str
    goal: str
    analysis: str

# --- LLM Setup ---
# Adjust model name if needed (or to a model available to you)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- Node ---
def hormozi_node(state: BizState) -> BizState:
    system_prompt = (
        "You are Alex Hormozi — direct, tactical, value-driven. "
        "You must return output ONLY in JSON format."
    )

    user_prompt = f"""
Business Description:
{state['business_description']}

Goal:
{state['goal']}

Respond STRICTLY in this JSON format:

{{
  "bottlenecks": [
    {{
      "name": "",
      "diagnosis": "",
      "tactical_fix": ["", ""]
    }},
    {{
      "name": "",
      "diagnosis": "",
      "tactical_fix": ["", ""]
    }}
  ],
  "summary": ""
}}
"""

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    import json
    try:
        state["analysis"] = json.loads(resp.content)
    except Exception:
        state["analysis"] = {"raw_text": getattr(resp, "content", str(resp))}

    return state


# --- build_graph updated to enable memory persistence ---
def build_graph():
    g = StateGraph(BizState)
    g.add_node("hormozi_analysis", hormozi_node)
    g.add_edge(START, "hormozi_analysis")
    g.add_edge("hormozi_analysis", END)

    # Use the MemorySaver available in your langgraph version
    memory = MemorySaver()

    graph = g.compile(checkpointer=memory)
    return graph, memory


# --- runner updated to call graph and then fetch memory info ---
def run_cli_and_save():
    """
    CLI runner that invokes the graph and provides a 'thread' object
    required by the MemorySaver checkpointer.
    """
    print("=== BizScale AI (Single Coach with MemorySaver) ===")
    desc = input("Describe your business: ").strip()
    goal = input("Primary goal: ").strip()

    initial_state: Dict[str, Any] = {
        "business_description": desc,
        "goal": goal,
        "analysis": ""
    }

    # build_graph must return (graph, memory)
    graph, memory = build_graph()

    # create a unique thread id for this run
    thread_id = f"biz-{uuid.uuid4().hex[:8]}"

    # 'thread' shape expected by LangGraph: {"configurable": {"thread_id": "<id>"}}
    thread = {"configurable": {"thread_id": thread_id}}

    # invoke the graph with the thread argument (important!)
    final_state = graph.invoke(initial_state, thread)

    print("\n\n===== ALEX HORMOZI ANALYSIS =====\n")
    # print structured analysis if available
    analysis = final_state.get("analysis")
    print(analysis)

    # Example: if MemorySaver allows listing / retrieving runs
    try:
        if hasattr(memory, "list_runs"):
            runs = memory.list_runs()
            print("\nSaved runs count (memory.list_runs):", len(runs))
        elif hasattr(memory, "get_runs"):
            runs = memory.get_runs()
            print("\nSaved runs count (memory.get_runs):", len(runs))
        else:
            print("\nMemorySaver present. Check its API to list saved runs.")
    except Exception:
        print("\nMemory saved (unable to inspect via MemorySaver API).")


if __name__ == "__main__":
    run_cli_and_save()
