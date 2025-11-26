import re, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Set
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from prompts import build_subgoal_prompts, build_entity_prompts
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
from utils import API_KEY, BASE_URL, GPT_MODEL

# ===================== Paths =====================
RAW  = Path("data")
OUT  = Path("outputs")
LOG  = Path("logs")
for p in (RAW, OUT, LOG):
    p.mkdir(parents=True, exist_ok=True)

# ===================== LLM Call Stub =====================

@retry(stop=stop_after_attempt(50), wait=wait_random_exponential(multiplier=1, max=600))
def call_llm_chat(client, prompt, model=GPT_MODEL):
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
    )
    return response.choices[0].message.content

# ================== UTILITIES ==================

FENCE_RE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)

def extract_code_block(text: str, fence_name: str) -> str:
    m = re.search(rf"```{fence_name}\n(.*?)```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""

def read_sources() -> Dict[str,str]:
    return {
        "paper": (RAW / "paper.txt").read_text(encoding="utf-8"),
        "code" : (RAW / "code.txt").read_text(encoding="utf-8"),
    }

def snake_case(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\s-]", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", "_", s)
    return s.lower()

def parse_jsonl(block: str) -> List[Dict[str,Any]]:
    objs=[]
    lines = [l for l in block.splitlines() if l.strip()]
    for i,line in enumerate(lines):
        try:
            obj=json.loads(line)
            objs.append(obj)
        except Exception as e:
            print(f"[WARN] JSON parse error line {i}: {e} :: {line[:180]}")
    return objs

def graph_text(nodes, edges, independent: Set[str], *, sep="; ") -> str:
    """
    nodes        : iterable of all node ids
    edges        : list from extract_edges_and_independent
    independent  : set from extract_edges_and_independent
    """

    incoming = defaultdict(lambda: {"AND": [], "OR": []})
    for e in edges:
        key = "OR" if e["logic"] == "OR" else "AND"  
        incoming[e["to"]][key].append(e["from"])

    parts = []
    for node in sorted(nodes):
        dep = incoming.get(node)
        is_root = (node in independent) or (not dep) 

        if is_root:
            parts.append(node)

        if dep:
            if dep["AND"]:
                lhs = " & ".join(sorted(dep["AND"]))
                parts.append(f"{lhs} -> {node}")

            for src in sorted(dep["OR"]):
                parts.append(f"{src} -> {node}")

    return sep.join(parts)

def extract_edges_and_independent(
    subgoals: Dict[str, Dict]
) -> Tuple[List[Dict[str, str]], Set[str]]:
    edges, independent = [], set()
    for sg in subgoals.values():
        for dep in sg.get("subgoals", []):
            dep_name = dep.get("subgoal")
            logic = dep.get("logic", "SINGLE").upper()

            if dep_name is None or str(dep_name).lower() == "null":
                if logic == "OR":
                    independent.add(sg["name"])
                continue

            if dep_name in subgoals:
                edges.append({"from": dep_name, "to": sg["name"], "logic": logic})
    return edges, independent

def depth_layers(nodes, edges, independent: Set[str]) -> Dict[str, int]:
    incoming = defaultdict(list)  
    outgoing = defaultdict(list)

    for e in edges:
        incoming[e["to"]].append(e["from"])
        outgoing[e["from"]].append(e["to"])

    roots = {n for n in nodes if (n in independent) or (not incoming.get(n))}
    depth = {n: 0 for n in roots}

    q = deque(roots)
    while q:
        u = q.popleft()
        for v in outgoing.get(u, []):
            if all(pred in depth for pred in incoming[v]):
                depth[v] = max(depth[pred] + 1 for pred in incoming[v])
                q.append(v)

    max_d = max(depth.values(), default=0)
    for n in nodes:
        depth.setdefault(n, max_d + 1)

    return depth

def node_repr(node: str, incoming_map: Dict[str, Dict[str, List[str]]],
              is_root: bool) -> List[str]:
    if is_root or node not in incoming_map:
        return [node]

    inc = incoming_map[node]
    parts = []
    if inc["AND"]:
        lhs = " & ".join(sorted(inc["AND"]))
        parts.append(f"{lhs} -> {node}")
    for src in sorted(inc["OR"]):
        parts.append(f"{src} -> {node}")
    return parts

def graph_text_by_depth(
    nodes: List[str],
    edges: List[Dict[str, str]],
    independent: Set[str],
    *,
    sep="; "
) -> str:
    incoming_map = defaultdict(lambda: {"AND": [], "OR": []})
    for e in edges:
        key = "OR" if e["logic"] == "OR" else "AND" 
        incoming_map[e["to"]][key].append(e["from"])

    depth = depth_layers(nodes, edges, independent)

    max_depth = max(depth.values())
    lines = []
    for d in range(max_depth + 1):
        layer_nodes = sorted([n for n, dep in depth.items() if dep == d])
        if not layer_nodes:
            continue
        layer_parts = []
        for n in layer_nodes:
            layer_parts.extend(
                node_repr(n, incoming_map, n in independent or not incoming_map.get(n))
            )
        lines.append(sep.join(layer_parts))

    return "\n".join(lines)

# ================== NORMALIZATION & VALIDATION ==================

def normalize_subgoal(obj: Dict[str,Any]) -> Dict[str,Any]:
    # 'name' mandatory; internal id = name
    if "name" not in obj:
        raise ValueError("Subgoal missing 'name'")
    obj["name"] = snake_case(obj["name"])
    obj.setdefault("description","")
    obj.setdefault("prerequisites", {})
    pre = obj["prerequisites"]
    for k in ("materials","tools","observations"):
        pre.setdefault(k, [])
        if pre[k] is None: pre[k] = []
        if not isinstance(pre[k], list):
            pre[k] = [pre[k]]

    raw_subs = obj.get("subgoals", [])
    if not isinstance(raw_subs, list):
        raw_subs = []
    cleaned_subs = []
    for entry in raw_subs:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("subgoal", "")
        if not sid:
            continue
        cleaned_subs.append({
            "subgoal": snake_case(sid),
            "logic": entry.get("logic", "SINGLE").upper(), 
            "relationship_description": entry.get(
                "relationship_description", ""
            ).strip()
        })
    obj["subgoals"] = cleaned_subs

    obj.setdefault("effects", {})
    eff = obj["effects"]
    for k in ("inventory","status","obsevations"):
        eff.setdefault(k, [])
        if eff[k] is None: eff[k] = []
        if not isinstance(eff[k], list):
            eff[k] = [eff[k]]

    return obj

def validate_subgoal(obj: Dict[str,Any]) -> List[str]:
    errs=[]
    if not re.match(r"^[a-z0-9_]+$", obj["name"]):
        errs.append("name_not_snake_case")
    for path, cond in [
        ("description", bool(obj.get("description","").strip()))
    ]:
        if not cond: errs.append(f"missing_{path}")
    if "prerequisites" not in obj:
        errs.append("missing_prerequisites")
    return errs

def normalize_entity(obj: Dict[str,Any]) -> Dict[str,Any]:
    if "name" not in obj:
        raise ValueError("Entity missing name")
    obj["name"] = snake_case(obj["name"])
    obj.setdefault("type","resource")
    if not isinstance(obj["knowledge"], list):
        obj["knowledge"] = [str(obj["knowledge"])]
    obj.setdefault("related_subgoals", [])
    if not isinstance(obj["related_subgoals"], list):
        obj["related_subgoals"] = [obj["related_subgoals"]]
    return obj

def validate_entity(obj: Dict[str,Any], subgoal_ids:set) -> List[str]:
    errs=[]
    if not re.match(r"^[a-z0-9_]+$", obj["name"]):
        errs.append("name_not_snake_case")
    if "knowledge" not in obj:
        errs.append("missing_knowledge")
    if "related_subgoals" not in obj:
        errs.append("missing_related_subgoals")
    else:
        for sid in obj["related_subgoals"]:
            if sid not in subgoal_ids:
                errs.append(f"unknown_subgoal:{sid}")
    return errs

# ================== STAGE 1 ==================

def stage_subgoals(client, model: str):
    src = read_sources()
    messages = build_subgoal_prompts(src["paper"], src["code"])
    raw = call_llm_chat(client, messages, model=model)
    (LOG / "stage1_raw.txt").write_text(raw, encoding="utf-8")

    block = extract_code_block(raw, "jsonl_subgoals")
    if not block:
        raise RuntimeError("Missing fenced block: jsonl_subgoals")
    parsed = parse_jsonl(block)
    if not parsed:
        raise RuntimeError("No subgoals parsed")

    dedup: Dict[str,Dict[str,Any]] = {}
    for obj in parsed:
        try:
            obj = normalize_subgoal(obj)
            if obj["name"] not in dedup:
                dedup[obj["name"]] = obj
        except Exception as e:
            print(f"[SKIP SUBGOAL] parse/normalize error: {e}")

    total_errors=0
    for sg in dedup.values():
        errs = validate_subgoal(sg)
        if errs:
            total_errors += 1
            print(f"[SUBGOAL ERROR] {sg['name']}: {errs}")

    edges, independent = extract_edges_and_independent(dedup)
    graph_str = graph_text_by_depth(list(dedup.keys()), edges, independent)

    with (OUT / "subgoals.jsonl").open("w", encoding="utf-8") as f:
        for sg in dedup.values():
            f.write(json.dumps(sg, ensure_ascii=False)+"\n")

    with (OUT / "subgoal_graph_edges.jsonl").open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False)+"\n")

    subgoal_names = sorted(dedup.keys()) 
    subgoal_str = "{" + ", ".join(subgoal_names) + "}"
    (LOG / "subgoal_name.txt").write_text(subgoal_str, encoding="utf-8")

    with (OUT / "subgoals_read.json").open("w", encoding="utf-8") as f:
        for sg in dedup.values():
            f.write(json.dumps(sg, indent=2, ensure_ascii=False))
            f.write("\n\n")
    
    (LOG / "subgoal_graph.txt").write_text(graph_str, encoding="utf-8")


    stats = {
        "count": len(dedup),
        "edge_count": len(edges),
        "validation_error_count": total_errors,
    }
    (OUT / "stats_subgoals.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] Stage 1 complete:", stats)

# ================== STAGE 2 ==================

def stage_entities(client, model: str):
    src = read_sources()
    subgoals_text = (OUT / "subgoals.jsonl").read_text(encoding="utf-8")
    messages = build_entity_prompts(src["paper"], src["code"], subgoals_text)
    raw = call_llm_chat(client, messages, model=model)
    (LOG / "stage2_raw.txt").write_text(raw, encoding="utf-8")

    block = extract_code_block(raw, "jsonl_entities")
    if not block:
        raise RuntimeError("Missing fenced block: jsonl_entities")

    parsed = parse_jsonl(block)
    if not parsed:
        raise RuntimeError("No entities parsed")

    subgoal_ids = {json.loads(l)["name"] for l in subgoals_text.splitlines() if l.strip()}
    dedup: Dict[str,Dict[str,Any]] = {}
    for obj in parsed:
        try:
            if "__summary__" in obj:
                continue
            obj = normalize_entity(obj)
            if obj["name"] not in dedup:
                dedup[obj["name"]] = obj
        except Exception as e:
            print(f"[SKIP ENTITY] normalize error: {e}")

    total_errors=0
    for ent in dedup.values():
        errs = validate_entity(ent, subgoal_ids)
        if errs:
            total_errors += 1
            print(f"[ENTITY ERROR] {ent['name']}: {errs}")

    with (OUT / "entity_kb.jsonl").open("w", encoding="utf-8") as f:
        for ent in dedup.values():
            f.write(json.dumps(ent, ensure_ascii=False)+"\n")
    
    with (OUT / "entity_kb_read.json").open("w", encoding="utf-8") as f:
        for ent in dedup.values():
            f.write(json.dumps(ent, indent=2, ensure_ascii=False))
            f.write("\n\n")

    stats = {
        "count": len(dedup),
        "validation_error_count": total_errors
    }
    (OUT / "stats_entities.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] Stage 2 complete:", stats)

# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_subgoals", default=GPT_MODEL)
    ap.add_argument("--model_entities", default=GPT_MODEL)
    ap.add_argument("--stage", choices=["all","subgoals","entities"], default="all")
    args = ap.parse_args()

    client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )

    if args.stage in ("all","subgoals"):
        stage_subgoals(client, args.model_subgoals)
    if args.stage in ("all","entities"):
        stage_entities(client, args.model_entities)

if __name__ == "__main__":
    main()
