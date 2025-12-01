# src/business_consultant_graph.py
import os
import sys
import time
import uuid
import json
import inspect
import traceback
from typing import TypedDict, Dict, Any, Optional, List, Tuple
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

# ========== RAG SETUP ==========
# Try imports in a robust order
EmbeddingClass = None
ChromaClass = None
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    EmbeddingClass = OpenAIEmbeddings
    ChromaClass = Chroma
except Exception:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.vectorstores import Chroma
        EmbeddingClass = OpenAIEmbeddings
        ChromaClass = Chroma
    except Exception:
        try:
            from langchain.embeddings.openai import OpenAIEmbeddings
            from langchain.vectorstores import Chroma
            EmbeddingClass = OpenAIEmbeddings
            ChromaClass = Chroma
        except Exception:
            print("WARNING: Could not import OpenAIEmbeddings/Chroma. RAG will be disabled.")
            EmbeddingClass = None
            ChromaClass = None

# RAG config
CHROMA_PERSIST_DIR = "chroma_persist"
RAG_TOP_K = 3
EMB_OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def _build_chroma_vectorstore(coach_collection_name: str):
    """Construct a Chroma vectorstore instance."""
    if not EmbeddingClass or not ChromaClass:
        raise RuntimeError("Chroma/Embeddings not available")
    emb = EmbeddingClass(openai_api_key=EMB_OPENAI_KEY)
    try:
        return ChromaClass(persist_directory=CHROMA_PERSIST_DIR, collection_name=coach_collection_name, embedding_function=emb)
    except TypeError:
        try:
            return ChromaClass(persist_directory=CHROMA_PERSIST_DIR, collection_name=coach_collection_name, embeddings=emb)
        except TypeError:
            return ChromaClass(persist_directory=CHROMA_PERSIST_DIR, collection_name=coach_collection_name)

def get_top_k_evidence_with_meta(coach: str, query: str, k: int = RAG_TOP_K) -> List[Tuple[str, Dict[str, Any]]]:
    """Returns list of tuples: (text, metadata) for top-k retrieved chunks."""
    vect = _build_chroma_vectorstore(coach)
    docs = vect.similarity_search(query, k=k)
    results: List[Tuple[str, Dict[str, Any]]] = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "text", None) or str(d)
        # try different metadata attributes
        meta = {}
        if hasattr(d, "metadata"):
            meta = getattr(d, "metadata") or {}
        elif hasattr(d, "meta"):
            meta = getattr(d, "meta") or {}
        results.append((text, meta or {}))
    return results

# ========== STATE DEFINITION ==========
class BizState(TypedDict, total=False):
    business_description: str
    goal: str
    kpis: Dict[str, Any]
    analysis_dan: Optional[Dict[str, Any]]
    analysis_sam: Optional[Dict[str, Any]]
    analysis_alex: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]

# ========== LLM ==========
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ========== PERSONA PROMPTS ==========
DAN_SYSTEM = (
    "You are Dan Martell — systems, delegation, and operational scaling expert.\n"
    "You MUST return ONLY a single JSON object matching the schema provided. "
    "No commentary, no markdown, no code fences. "
    "If KPIs are missing, include a 'proposed_kpis' array with exactly 3 items of the form {\"kpi\": \"\", \"why\": \"\"}. "
    "If you cannot determine a value, set it to the string \"I_DONT_KNOW\"."
)

SAM_SYSTEM = (
    "You are Sam Ovens — positioning, niche, and client-acquisition expert.\n"
    "You MUST return ONLY a single JSON object matching the schema provided. "
    "No commentary, no markdown, no code fences. "
    "If KPIs are missing, include a 'proposed_kpis' array with exactly 3 items of the form {\"kpi\": \"\", \"why\": \"\"}. "
    "If you cannot determine a value, set it to the string \"I_DONT_KNOW\"."
)

ALEX_SYSTEM = (
    "You are Alex Hormozi — offer creation and pricing expert.\n"
    "You MUST return ONLY a single JSON object matching the schema provided. "
    "No commentary, no markdown, no code fences. "
    "If KPIs are missing, include a 'proposed_kpis' array with exactly 3 items of the form {\"kpi\": \"\", \"why\": \"\"}. "
    "If you cannot determine a value, set it to the string \"I_DONT_KNOW\"."
)

COACH_JSON_SCHEMA = """
Respond ONLY in this JSON format (exact keys; additional keys are allowed but the listed keys should be present):

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
  "proposed_kpis": [
    {"kpi": "", "why": ""}
  ],
  "summary": ""
}
"""

# ========== HELPERS ==========
import re

def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Try to parse JSON from LLM output robustly:
    - strip triple/backtick code fences
    - extract first {...} block
    - return {"raw_text": original} on failure
    """
    if not isinstance(text, str):
        try:
            return dict(text)
        except Exception:
            return {"raw_text": str(text)}

    s = text.strip()

    # remove markdown code fences ```json ... ``` or ``` ... ```
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # remove leading/trailing single backticks
    s = s.strip("` \n")

    # try to find the first JSON object substring
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # final attempt: direct json.loads
    try:
        return json.loads(s)
    except Exception:
        return {"raw_text": text}

def validate_and_fix_json(parsed: Dict[str, Any], llm_instance: ChatOpenAI, msgs: List[Any], max_retries: int = 1) -> Dict[str, Any]:
    """
    Ensure required keys exist. If not, re-prompt the model (one retry) with a strict instruction
    to output only valid JSON and to fill missing keys.
    msgs is the original message list [SystemMessage, HumanMessage].
    """
    required = ["bottlenecks", "top_recommendation", "kpis_to_track", "summary"]
    if not isinstance(parsed, dict):
        parsed = {"raw_text": str(parsed)}

    missing = [k for k in required if k not in parsed]
    if not missing:
        return parsed

    # prepare re-prompt
    system_msg = msgs[0] if len(msgs) > 0 else SystemMessage(content="You are a strict JSON assistant.")
    human_msg = msgs[-1] if len(msgs) > 0 else HumanMessage(content="")
    repair_instruction = (
        human_msg.content
        + "\n\nIMPORTANT: Your previous response was missing keys: "
        + ", ".join(missing)
        + ". Return ONLY a valid JSON object that includes these keys. "
        + "Do not include any text outside the JSON. If you can't determine a value, use \"I_DONT_KNOW\"."
    )
    re_msgs = [system_msg, HumanMessage(content=repair_instruction)]
    try:
        resp = llm_instance.invoke(re_msgs)
        repaired = safe_parse_json(getattr(resp, "content", str(resp)))
        if isinstance(repaired, dict):
            # merge: repaired wins for missing keys
            parsed.update(repaired)
        return parsed
    except Exception:
        return parsed

def build_coach_prompt_with_rag(system_text: str, business_desc: str, goal: str, kpis: dict, coach: str, k: int = RAG_TOP_K) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Build the System+Human messages for the coach including retrieved evidence. Returns (msgs, provenance)."""
    query = f"{business_desc}\nGoal: {goal}"
    evidence = []
    try:
        evidence = get_top_k_evidence_with_meta(coach, query, k=k)
    except Exception as e:
        # retrieval not available or failed
        # print for debug but continue
        print(f"RAG retrieval failed for {coach}: {e}")
        evidence = []

    evidence_block = ""
    provenance: List[Dict[str, Any]] = []
    if evidence:
        pieces = []
        for i, (txt, meta) in enumerate(evidence, start=1):
            src = meta.get("source", meta.get("source_file", "unknown"))
            cid = meta.get("chunk_id", meta.get("chunk", ""))
            pieces.append(f"--- EVIDENCE {i} (source={src}, chunk_id={cid}) ---\n{txt}")
            provenance.append({"evidence_rank": i, "source": src, "chunk_id": cid})
        evidence_block = "\n\n".join(pieces)
    else:
        evidence_block = "NO_RETRIEVED_EVIDENCE"

    kpi_block = "\n".join([f"- {kk}: {vv}" for kk, vv in (kpis or {}).items()]) or "none"

    human_text = f"""
Business Description:
{business_desc}

Goal:
{goal}

KPIs:
{kpi_block}

Retrieved Evidence (top {len(evidence)} from coach collection '{coach}'):
{evidence_block}

Respond STRICTLY in this JSON format (no extra commentary):
{COACH_JSON_SCHEMA}
"""
    msgs = [SystemMessage(content=system_text), HumanMessage(content=human_text)]
    return msgs, provenance

# ========== COACH NODES ==========
def dan_node(state: BizState) -> Dict[str, Any]:
    msgs, provenance = build_coach_prompt_with_rag(DAN_SYSTEM, state.get("business_description", ""), state.get("goal", ""), state.get("kpis", {}), "dan_martell")
    resp = llm.invoke(msgs)
    parsed = safe_parse_json(getattr(resp, "content", str(resp)))
    parsed = validate_and_fix_json(parsed, llm, msgs)
    return {"analysis_dan": {"analysis": parsed, "provenance": provenance}}

def sam_node(state: BizState) -> Dict[str, Any]:
    msgs, provenance = build_coach_prompt_with_rag(SAM_SYSTEM, state.get("business_description", ""), state.get("goal", ""), state.get("kpis", {}), "sam_ovens")
    resp = llm.invoke(msgs)
    parsed = safe_parse_json(getattr(resp, "content", str(resp)))
    parsed = validate_and_fix_json(parsed, llm, msgs)
    return {"analysis_sam": {"analysis": parsed, "provenance": provenance}}

def alex_node(state: BizState) -> Dict[str, Any]:
    msgs, provenance = build_coach_prompt_with_rag(ALEX_SYSTEM, state.get("business_description", ""), state.get("goal", ""), state.get("kpis", {}), "alex_hormozi")
    resp = llm.invoke(msgs)
    parsed = safe_parse_json(getattr(resp, "content", str(resp)))
    parsed = validate_and_fix_json(parsed, llm, msgs)
    return {"analysis_alex": {"analysis": parsed, "provenance": provenance}}

# ========== MERGE NODE ==========
def merge_node(state: BizState) -> Dict[str, Any]:
    # Collect analyses from unique per-coach keys
    analyses_list: List[Dict[str, Any]] = []
    if "analysis_dan" in state and state["analysis_dan"] is not None:
        a = state["analysis_dan"]
        # handle both shapes: either {'analysis': {...}, 'provenance': [...] } or direct dict
        if isinstance(a, dict) and "analysis" in a:
            analyses_list.append({"coach": "dan_martell", "analysis": a["analysis"], "provenance": a.get("provenance", [])})
        else:
            analyses_list.append({"coach": "dan_martell", "analysis": a, "provenance": []})
    if "analysis_sam" in state and state["analysis_sam"] is not None:
        a = state["analysis_sam"]
        if isinstance(a, dict) and "analysis" in a:
            analyses_list.append({"coach": "sam_ovens", "analysis": a["analysis"], "provenance": a.get("provenance", [])})
        else:
            analyses_list.append({"coach": "sam_ovens", "analysis": a, "provenance": []})
    if "analysis_alex" in state and state["analysis_alex"] is not None:
        a = state["analysis_alex"]
        if isinstance(a, dict) and "analysis" in a:
            analyses_list.append({"coach": "alex_hormozi", "analysis": a["analysis"], "provenance": a.get("provenance", [])})
        else:
            analyses_list.append({"coach": "alex_hormozi", "analysis": a, "provenance": []})

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
        "proposed_kpis": [],
        "final_summary": ""
    }

    all_b = []
    for a in analyses_list:
        merged["coach_insights"][a["coach"]] = {
            "analysis": a["analysis"],
            "provenance": a.get("provenance", [])
        }
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

    # KPIs to track (union)
    kpis_set = set()
    for a in analyses_list:
        if isinstance(a["analysis"], dict):
            for k in a["analysis"].get("kpis_to_track", []):
                kpis_set.add(k)
    merged["kpis_to_track"] = list(kpis_set)

    # Collect proposed_kpis from coaches (if any)
    proposed = []
    for a in analyses_list:
        if isinstance(a["analysis"], dict):
            for pk in a["analysis"].get("proposed_kpis", []):
                proposed.append(pk)
    merged["proposed_kpis"] = proposed

    # final summary: concat coach summaries
    summaries = [a["analysis"].get("summary", "") for a in analyses_list if isinstance(a["analysis"], dict)]
    merged["final_summary"] = " || ".join([s for s in summaries if s])

    return {"final_report": merged}

# ========== GRAPH BUILDER ==========
def build_graph():
    g = StateGraph(BizState)
    g.add_node("dan_analysis", dan_node)
    g.add_node("sam_analysis", sam_node)
    g.add_node("alex_analysis", alex_node)
    g.add_node("merge_report", merge_node)

    # START -> each coach
    g.add_edge(START, "dan_analysis")
    g.add_edge(START, "sam_analysis")
    g.add_edge(START, "alex_analysis")

    # each coach -> merge
    g.add_edge("dan_analysis", "merge_report")
    g.add_edge("sam_analysis", "merge_report")
    g.add_edge("alex_analysis", "merge_report")

    # merge -> END
    g.add_edge("merge_report", END)

    memory = MemorySaver()
    graph = g.compile(checkpointer=memory)
    return graph, memory

# ========== VERBOSE RUNNER ==========
def run_all_coaches_and_save_verbose():
    """Verbose runner with diagnostics."""
    print("=== BizScale AI (Verbose Runner) ===")
    print("OPENAI_API_KEY loaded?:", bool(os.getenv("OPENAI_API_KEY")))
    print("Python executable:", sys.executable)
    print("Working dir:", os.getcwd())

    for coach in ["alex_hormozi","dan_martell","sam_ovens"]:
        p = Path("data/processed")/coach/"chunks.jsonl"
        print(f"Processed chunks for {coach}: exists={p.exists()}", end="")
        if p.exists():
            print(", size=", p.stat().st_size)
        else:
            print("")

    print("chroma_persist exists?:", Path("chroma_persist").exists())

    interactive = sys.stdin.isatty()
    if interactive:
        print("Running in interactive mode (will prompt for input).")
        desc = input("Describe your business: ").strip()
        goal = input("Primary goal: ").strip()
        # optional KPI preflight
        kpis = {}
        want_kpis = input("Do you want to provide any KPIs? (y/n): ").strip().lower()
        if want_kpis.startswith("y"):
            print("Enter up to 3 KPIs (press enter to skip):")
            for i in range(3):
                name = input(f"KPI {i+1} name (or blank to stop): ").strip()
                if not name:
                    break
                val = input(f"Value for {name} (optional): ").strip()
                kpis[name] = val or None
    else:
        print("Non-interactive environment detected (no stdin). Using default sample inputs.")
        desc = "Sample small gym business with declining monthly revenue."
        goal = "Increase monthly recurring revenue by 30% in 6 months."
        kpis = {}

    print("Business description (len):", len(desc))
    print("Goal:", goal)
    print("Initial KPIs provided:", kpis)

    initial_state: BizState = {
        "business_description": desc,
        "goal": goal,
    }
    if kpis:
        initial_state["kpis"] = kpis

    try:
        t0 = time.time()
        graph, memory = build_graph()
        print("Graph built successfully.")
        thread_id = f"biz-{uuid.uuid4().hex[:8]}"
        thread = {"configurable": {"thread_id": thread_id}}
        print("Invoking graph (this may take some time if LLM calls are made)...")
        final_state = graph.invoke(initial_state, thread)
        t1 = time.time()
        print(f"Graph invoked. Duration: {t1-t0:.2f}s")
        print("Final state keys:", list(final_state.keys()))
        print("\n===== FINAL MERGED REPORT (pretty-print) =====")
        try:
            print(json.dumps(final_state.get("final_report", {}), indent=2, ensure_ascii=False))
        except Exception:
            print(final_state.get("final_report"))
        
        # Persist run metadata for auditing
        try:
            run_meta = {
                "thread_id": thread_id,
                "final_report": final_state.get("final_report", {}),
                "timestamp": time.time()
            }
            Path("data/metadata").mkdir(parents=True, exist_ok=True)
            with open("data/metadata/runs.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(run_meta, ensure_ascii=False) + "\n")
        except Exception:
            pass

        try:
            if hasattr(memory, "list_runs"):
                runs = memory.list_runs()
                print("\nMemorySaver.list_runs -> count:", len(runs))
            elif hasattr(memory, "get_runs"):
                runs = memory.get_runs()
                print("\nMemorySaver.get_runs -> count:", len(runs))
            else:
                print("\nMemorySaver available; no list/get methods found.")
        except Exception as e:
            print("Memory inspection failed:", repr(e))

    except Exception as exc:
        print("\n!!! Exception during run !!!")
        traceback.print_exc()
        print("\nPlease paste the above traceback into the chat.")

if __name__ == "__main__":
    run_all_coaches_and_save_verbose()
