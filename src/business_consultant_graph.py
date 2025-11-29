"""
STEP 2 — Basic LangGraph App With a Single Coach (No RAG)
Coach Used: Alex Hormozi
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

# If your langgraph/langchain imports differ, adapt them accordingly.
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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
        "Give short, powerful analysis focusing on offer, pricing, and bottlenecks."
    )

    user_prompt = f"""
Business Description:
{state['business_description']}

Goal:
{state['goal']}

Please provide:
1) Top 2 bottlenecks (funnel: acquisition → conversion → retention).
2) One-sentence diagnosis for each.
3) A 2-step tactical fix per bottleneck.
Format with headings and bullets.
"""

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    state["analysis"] = getattr(resp, "content", str(resp))
    return state

# --- Graph builder ---
def build_graph():
    g = StateGraph(BizState)
    g.add_node("hormozi_analysis", hormozi_node)
    g.add_edge(START, "hormozi_analysis")
    g.add_edge("hormozi_analysis", END)
    memory = None
    try:
        # If your langgraph flow needs a checkpoint/memory, wire it here.
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
    except Exception:
        memory = None
    return g.compile(checkpointer=memory) if memory else g.compile()

# --- Runner ---
def run_cli():
    print("=== BizScale AI (Step 2: Single Coach) ===")
    desc = input("Describe your business (one paragraph):\n> ").strip()
    goal = input("Primary goal (e.g., double revenue in 6 months):\n> ").strip()

    initial_state: BizState = {
        "business_description": desc,
        "goal": goal,
        "analysis": ""
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": "1"}})

    print("\n\n===== ALEX HORMOZI ANALYSIS =====\n")
    print(final_state["analysis"])


if __name__ == "__main__":
    run_cli()
    