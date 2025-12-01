# src/business_consultant_graph.py
import os
import uuid
import json
from typing import TypedDict, Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# LangGraph / LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load env
load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# ---- State types (can be flexible — nodes return partial dicts) ----
class BizState(TypedDict, total=False):
    business_description: str
    goal: str
    kpis: Dict[str, Any]
    # Not using 'analyses' as concurrent write target anymore
    analysis_dan: Optional[Dict[str, Any]]
    analysis_sam: Optional[Dict[str, Any]]
    analysis_alex: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]

# ensure data dir exists
Path("data").mkdir(parents=True, exist_ok=True)

# ---- LLM ----
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ---- Persona prompts ----
DAN_SYSTEM = "You are Dan Martell — systems, delegation, and operational scaling expert. Return JSON only."
SAM_SYSTEM = "You are Sam Ovens — positioning, niche, and client-acquisition expert. Return JSON only."
ALEX_SYSTEM = "You are Alex Hormozi — offer creation and pricing expert. Return JSON only."

COACH_JSON_SCHEMA = """
Respond ONLY in this JSON format:

{
  "bottlenecks": [
    {
      "name": "",
      "diagnosis": "",
      "tactical_fix": ["", ""],
      "priority": "low|medium|high"
    }
  ],
  "top_recommendation": "",
  "kpis_to_track": ["kpi_name1", "kpi_name2"],
  "summary": ""
}
"""

# ---- helpers ----
def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        if start != -1:
            try:
                return json.loads(text[start:])
            except Exception:
                pass
        return {"raw_text": text}

def build_coach_prompt(system_text: str, business_desc: str, goal: str, kpis: Dict[str, Any]):
    kpi_block = "\n".join([f"- {k}: {v}" for k, v in (kpis or {}).items()]) or "none"
    user_text = f"""
Business Description:
{business_desc}

Goal:
{goal}

KPIs:
{kpi_block}

{COACH_JSON_SCHEMA}
"""
    return [SystemMessage(content=system_text), HumanMessage(content=user_text)]

# -------- Coach nodes each return a UNIQUE key (partial updates) --------

def dan_node(state: BizState) -> Dict[str, Any]:
    msgs = build_coach_prompt(DAN_SYSTEM, state["business_description"], state["goal"], state.get("kpis", {}))
    resp = llm.invoke(msgs)
    parsed = safe_parse_json(getattr(resp, "content", str(resp)))
    # Return partial state: only this coach's analysis
    return {"analysis_dan": parsed}

def sam_node(state: BizState) -> Dict[str, Any]:
    msgs = build_coach_prompt(SAM_SYSTEM, state["business_description"], state["goal"], state.get("kpis", {}))
    resp = llm.invoke(msgs)
    parsed = safe_parse_json(getattr(resp, "content", str(resp)))
    return {"analysis_sam": parsed}

def alex_node(state: BizState) -> Dict[str, Any]:
    msgs = build_coach_prompt(ALEX_SYSTEM, state["business_description"], state["goal"], state.get("kpis", {}))
    resp = llm.invoke(msgs)
    parsed = safe_parse_json(getattr(resp, "content", str(resp)))
    return {"analysis_alex": parsed}

# ---- Merge node: read unique analysis_* keys and return final_report (partial) ----
def merge_node(state: BizState) -> Dict[str, Any]:
    # Collect coach analyses from the unique keys
    analyses = []
    if "analysis_dan" in state and state["analysis_dan"] is not None:
        analyses.append({"coach": "dan_martell", "analysis": state["analysis_dan"]})
    if "analysis_sam" in state and state["analysis_sam"] is not None:
        analyses.append({"coach": "sam_ovens", "analysis": state["analysis_sam"]})
    if "analysis_alex" in state and state["analysis_alex"] is not None:
        analyses.append({"coach": "alex_hormozi", "analysis": state["analysis_alex"]})

    merged = {
        "business_snapshot": {
            "description": state.get("business_description", ""),
            "goal": state.get("goal", ""),
            "kpis": state.get("kpis", {})
        },
        "coach_insights": {},
        "consensus_bottlenecks": [],
        "action_plan": [],
        "kpis_to_track": [],
        "final_summary": ""
    }

    # populate coach_insights and gather bottlenecks
    all_b = []
    for a in analyses:
        merged["coach_insights"][a["coach"]] = a["analysis"]
        if isinstance(a["analysis"], dict):
            for b in a["analysis"].get("bottlenecks", []):
                b_copy = dict(b)
                b_copy["source"] = a["coach"]
                all_b.append(b_copy)

    # rank by priority
    priority_map = {"high": 3, "medium": 2, "low": 1}
    all_b_sorted = sorted(all_b, key=lambda x: priority_map.get(x.get("priority", "medium"), 2), reverse=True)
    merged["consensus_bottlenecks"] = all_b_sorted

    # action plan from top fixes
    action_plan = []
    for b in all_b_sorted:
        for fix in b.get("tactical_fix", []):
            action_plan.append({"fix": fix, "from": b.get("source", "")})
    merged["action_plan"] = action_plan[:8]

    # KPIs union
    kpis_set = set()
    for a in analyses:
        if isinstance(a["analysis"], dict):
            for k in a["analysis"].get("kpis_to_track", []):
                kpis_set.add(k)
    merged["kpis_to_track"] = list(kpis_set)

    # summaries
    summaries = [a["analysis"].get("summary", "") for a in analyses if isinstance(a["analysis"], dict)]
    merged["final_summary"] = " || ".join([s for s in summaries if s])

    # Return partial update with only the final_report key
    return {"final_report": merged}

# ---- Graph wiring (edges added individually) ----
def build_graph():
    g = StateGraph(BizState)
    g.add_node("dan_analysis", dan_node)
    g.add_node("sam_analysis", sam_node)
    g.add_node("alex_analysis", alex_node)
    g.add_node("merge_report", merge_node)

    # START -> each coach node (separately)
    g.add_edge(START, "dan_analysis")
    g.add_edge(START, "sam_analysis")
    g.add_edge(START, "alex_analysis")

    # each coach -> merge_report
    g.add_edge("dan_analysis", "merge_report")
    g.add_edge("sam_analysis", "merge_report")
    g.add_edge("alex_analysis", "merge_report")

    # merge -> END
    g.add_edge("merge_report", END)

    memory = MemorySaver()
    graph = g.compile(checkpointer=memory)
    return graph, memory

# ---- Runner (threaded, MemorySaver) ----
def run_all_coaches_and_save():
    print("=== BizScale AI (Multi-coach) ===")
    desc = input("Describe your business: ").strip()
    goal = input("Primary goal: ").strip()

    kpis = {}
    leads = input("Leads per month (leave blank if unknown): ").strip()
    if leads:
        try:
            kpis['leads_per_month'] = float(leads.replace(",", ""))
        except Exception:
            kpis['leads_per_month'] = leads

    initial_state: BizState = {
        "business_description": desc,
        "goal": goal,
        "kpis": kpis
        # analysis_* keys absent initially
    }

    graph, memory = build_graph()
    thread_id = f"biz-{uuid.uuid4().hex[:8]}"
    thread = {"configurable": {"thread_id": thread_id}}

    # invoke
    final_state = graph.invoke(initial_state, thread)

    # final merged report should be present
    print("\n\n===== FINAL MERGED REPORT =====\n")
    print(json.dumps(final_state.get("final_report", {}), indent=2, ensure_ascii=False))

    # try to inspect memory (best-effort)
    try:
        if hasattr(memory, "list_runs"):
            runs = memory.list_runs()
            print("\nMemorySaver runs count:", len(runs))
        elif hasattr(memory, "get_runs"):
            runs = memory.get_runs()
            print("\nMemorySaver runs count:", len(runs))
        else:
            print("\nMemorySaver present; check its API if you need to export runs.")
    except Exception:
        print("\nMemorySaver used; could not inspect runs via API.")

if __name__ == "__main__":
    run_all_coaches_and_save()
