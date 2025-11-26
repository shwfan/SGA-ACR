import numpy as np
import torch
import json
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Iterable, Optional
from pathlib import Path




API_KEY = '' # Your API key
BASE_URL = '' # URL for your API
GPT_MODEL = '' # GPT model name
LOCAL_MODEL_PATH = '' # Your local llm path
SBERT_PATH = '' # SentenceBert path  

QUERY_INTERVAL = 100
OBEY_R = 0.2 # extra reward


TYPE_DICT = {
    1: 'water',
    2: 'grass',
    3: 'stone',
    4: 'path',
    5: 'sand',
    6: 'tree',
    7: 'lava',
    8: 'coal',
    9: 'iron',
    10: 'diamond',
    11: 'table',
    12: 'furnace',
    13: 'player',
    14: 'cow',
    15: 'zombie',
    16: 'skeleton',
    17: 'arrow',
    18: 'plant'
}

ACTIONS_NAME = [
    'noop',
    'move_left',
    'move_right',
    'move_up',
    'move_down',
    'do',
    'sleep',
    'place_stone',
    'place_table',
    'place_furnace',
    'place_plant',
    'make_wood_pickaxe',
    'make_stone_pickaxe',
    'make_iron_pickaxe',
    'make_wood_sword',
    'make_stone_sword',
    'make_iron_sword'
  ]

def l_func(x):
    return 1 / (1 + torch.exp(-10 * (x - 0.1)))

def compute_l_score(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    result = l_func(torch.mm(a_norm, b_norm.transpose(0, 1)))
    
    return result[0].detach().cpu().numpy()

def compute_bin_score(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    result = l_func(torch.mm(a_norm, b_norm.transpose(0, 1)))[0].detach().cpu().numpy()
    
    if result >= 0.5:
        return 1
    else:
        return 0

def get_fov(info):
    pos = info['player_pos']
    obs = info['semantic']

    fov_size = np.array([9, 7])
    top_left = np.maximum(pos - fov_size // 2, 0)
    bottom_right = np.minimum(pos + fov_size // 2 + 1, obs.shape)
    fov = obs[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    pad_top = top_left[0] - pos[0] + fov_size[0] // 2
    pad_bottom = pos[0] + fov_size[0] // 2 + 1 - bottom_right[0]
    pad_left = top_left[1] - pos[1] + fov_size[1] // 2
    pad_right = pos[1] + fov_size[1] // 2 + 1 - bottom_right[1]
    fov = np.pad(fov, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    types = np.unique(fov)
    type_strings = [TYPE_DICT[t] for t in types if t != 13 and t != 0]
    type_message = ', '.join(type_strings)
    
    return type_message, type_strings

def crafter_state_text(info: dict) -> str:
    gauge_key = ['health', 'food', 'drink', 'energy']
    gauge_parts = [f"{k} {v}/9" for k, v in info['inventory'].items() if k in gauge_key]
    gauge_parts = []
    inv_parts = []
    inv_entity = []
    for k, v in info['inventory'].items():
        if k in gauge_key:
            gauge_parts.append(f"{k} {v}(max=9)")
        else:
            if v > 0:
                inv_parts.append(f"{k} {v}")
                inv_entity.append(f"{k}")
    status_text = ', '.join(gauge_parts)
    inv_text = ', '.join(inv_parts) if inv_parts else "none"

    return status_text, inv_text, inv_entity


def see_delta(prev_fov_str: str, curr_fov_str: str) -> str:
    prev_set = {s.strip() for s in prev_fov_str.split(",") if s.strip()}
    curr_set = {s.strip() for s in curr_fov_str.split(",") if s.strip()}
    new_objs = curr_set - prev_set
    return ", ".join(sorted(new_objs))


def state_delta(prev_inv: dict, curr_inv: dict, gauge_keys=('health', 'food', 'drink', 'energy')):
    status_delta = []
    inv_delta = []
    # gauges (0‥9 scale)
    for k in gauge_keys:
        d = curr_inv[k] - prev_inv[k]
        if d:
            sign = '+' if d > 0 else ''
            status_delta.append(f"{k} {sign}{d}")
    # normal items
    for k, v_now in curr_inv.items():
        if k in gauge_keys:
            continue
        d = v_now - prev_inv.get(k, 0)
        if d:
            sign = '+' if d > 0 else ''
            inv_delta.append(f"{k} {sign}{d}")
    status_delta_text = ', '.join(status_delta)
    inv_delta_text = ', '.join(inv_delta)
    return status_delta_text, inv_delta_text

def get_time(info):
    time_step = info['time_step']
    hours = int(((90 + time_step) % 300) / 300 * 24)
    if 5 < hours < 20:
        time_description = "Daytime"
    else:
        time_description = "Nighttime"
    return time_description

def text_transition(prev_info, curr_info, action_name, prev_fov=None):
    """
    Generate text form transition 
    """
    status_delta, inv_delta = state_delta(prev_info['inventory'], curr_info['inventory'])
    curr_fov, _ = get_fov(curr_info)
    if prev_fov is not None:
        delta_see = see_delta(prev_fov, curr_fov)
    else:
        prev_fov, _ = get_fov(prev_info)
        delta_see = see_delta(prev_fov, curr_fov)
    status, inv, _ = crafter_state_text(curr_info)
    text_transition = f"Action: <{action_name}>; Status delta: <{status_delta or 'none'}>; Inventory delta: <{inv_delta or 'none'}>; See delta: <{delta_see or 'none'}>; Status: <{status}>; Inventory: <{inv}>; See: <{curr_fov}>;"
    return text_transition, curr_fov

def get_fov_goal(info):
    """
    Generate text form observation for sub-goals generation 
    """
    time_description = get_time(info)
    fov, fov_entity = get_fov(info)
    status, inv, inv_entity = crafter_state_text(info)
    text_obs = f"Time: <{time_description}>. Player sees: <{fov}>. Player status: <{status}>. Player inventory: <{inv}>"
    entity_list = fov_entity + inv_entity
    return text_obs, fov, entity_list


def fulfilled_score(subgoal_set, fulfilled_set) -> float:

    hit, total = 0.0, 0.0
    for _, g in enumerate(subgoal_set):
        w = 1.0
        total += w
        if g in fulfilled_set:
            hit += w

    return hit / total if total else 0.0


class RuleBasedSubgoalChecker:

    def __init__(self, subgoal_set):
        
        self.subgoal_set = subgoal_set
        self.ach_direct = {"eat_cow", "eat_plant", "defeat_zombie", "defeat_skeleton", "sleep", "place_stone"}
        self.undone_set = self.subgoal_set.copy()
        self.clear_state()

    # ---------- 外部接口 ----------
    def reset_plan(self, plan: List[str], base_info: dict):
        self.active: Set[str] = set(plan)
        self.done = set()
        self.prev_inv = base_info["inventory"].copy()
        self.prev_ach = base_info["achievements"].copy()
        self.plan_steps = 0
        self.just_completed: List[str] = []

    def step(self, info: dict) -> tuple[list[str], float]:

        self.just_completed = []
        self.plan_steps += 1
        cur_inv, cur_ach, nearby = info["inventory"], info["achievements"], info["nearby"]
        text_obs, _, entity_list= get_fov_goal(info)

        total_r = 0.0

        for sg in list(self.active):
            flag = self._check_done(sg, cur_inv, cur_ach, nearby)
            if not flag:
                continue

            total_r += OBEY_R

            self.active.remove(sg)
            self.just_completed.append(sg)
            self.done.add(sg)


        self.prev_inv, self.prev_ach = (
            cur_inv.copy(), cur_ach.copy()
        )
        return total_r, text_obs, entity_list

    def clear_state(self):
        self.active, self.done = set(), set()
        self.prev_inv, self.prev_ach = {}, {}
        self.undone_set = self.subgoal_set.copy()
        self.has_ach = set()

    def pop_just_completed(self) -> List[str]:
       out = self.just_completed
       self.just_completed = []
       return out

    def _check_done(self, sg, inv, ach, nearby) -> bool:
        flag = False
        if sg in self.ach_direct:
            if sg == 'sleep':
                subgoal = 'wake_up'
            else:
                subgoal = sg
            if ach[subgoal] > self.prev_ach[subgoal]:
                flag = True
                if (sg == 'defeat_zombie' or sg == 'defeat_skeleton' or sg == 'place_stone'):
                    self.undone_set.discard(sg)
            else:
                flag = False
            return flag
            
        verb, obj = sg.split("_", 1)
        if sg == "collect_water":
            obj = "drink"
        
        if verb == "place":
            if obj in nearby:
                flag = True
                if sg == 'place_plant':
                    self.undone_set.discard(sg)
            else:
                flag = False
            return flag
        
        if verb == "collect" or verb == "make":
            if inv[obj] > self.prev_inv[obj]:
                flag = True
                if (obj == 'wood_pickaxe' or obj == 'stone_pickaxe' or obj == 'iron_pickaxe' or sg =='wood_sword' or sg == 'stone_sword' or sg =='iron_sword' or obj == 'sapling'):
                    self.undone_set.discard(sg)
            else:
                flag = False
            return flag
        
        return flag



_WEAPON_PRIORITY = ["iron", "stone", "wood"]
def _inv_key_to_make_goal(inv_key: str) -> Optional[str]:
    if inv_key.endswith("_sword"):
        return f"make_{inv_key}"
    return None

class WeightedSubgoalGraph:
    def __init__(self, graph_path, data_path):
        self.graph_path = graph_path
        self.independent_nodes: Set[str] = set()
        self.incoming_edges: Dict[str, List[Tuple[Tuple[str, ...], str]]] = defaultdict(list)
        self.edges_all: List[str] = [] 
        self._parse_graph()
        self.save_path = Path(data_path) / ("graph_weight_data.jsonl")


        self.node_planned: Dict[str, int] = defaultdict(int)
        self.node_success: Dict[str, int] = defaultdict(int)
        self.edge_planned: Dict[str, int] = defaultdict(int)
        self.edge_success: Dict[str, int] = defaultdict(int)


        self.st_node_planned: Dict[str, int] = defaultdict(int)
        self.st_node_success: Dict[str, int] = defaultdict(int)
        self.st_edge_planned: Dict[str, int] = defaultdict(int)
        self.st_edge_success: Dict[str, int] = defaultdict(int)

    def _parse_graph(self):
        from pathlib import Path
        raw = Path(self.graph_path).read_text(encoding="utf-8").strip()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        first = lines[0]
        self.independent_nodes = set([p.strip() for p in first.split(';') if p.strip()])

        self.layer_edges: list[list[str]] = []   
        self.edges_all = []                     

        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(';') if p.strip()]
            self.layer_edges.append(parts)
            for edge_str in parts:
                self.edges_all.append(edge_str)
                lhs, rhs = [x.strip() for x in edge_str.split('->')]
                if '&' in lhs:
                    srcs = tuple([s.strip() for s in lhs.split('&')])
                else:
                    srcs = (lhs,)
                self.incoming_edges[rhs].append((srcs, edge_str))


    @staticmethod
    def _rate(success: int, planned: int) -> str:
        if planned <= 0:
            return "-%"
        pct = round(100.0 * success / float(planned), 3)
        return f"{pct:.3f}%"

    def stage_plan(self, plan_sgs: Iterable[str]):
        for g in plan_sgs:
            if g in self.independent_nodes:
                self.st_node_planned[g] += 1
            for (_, edge_str) in self.incoming_edges.get(g, []):
                self.st_edge_planned[edge_str] += 1

    def _resolve_or_edge_for_defeat(self, target: str, inv) -> Optional[str]:
        chosen_make_goal = None
        for mat in _WEAPON_PRIORITY:
            key = f"{mat}_sword"
            if inv.get(key, 0) and inv[key] > 0:
                chosen_make_goal = _inv_key_to_make_goal(key)
                break

        if not chosen_make_goal:
            return None

        for (srcs, edge_str) in self.incoming_edges.get(target, []):
            if chosen_make_goal in srcs:
                return edge_str
        return None

    def stage_success(self, subgoal: str, inv):
        if subgoal in self.independent_nodes:
            self.st_node_success[subgoal] += 1

        incoming = self.incoming_edges.get(subgoal, [])
        if not incoming:
            return


        if subgoal.startswith("defeat_") and inv is not None:
            edge_str = self._resolve_or_edge_for_defeat(subgoal, inv)
            if edge_str is not None:
                self.st_edge_success[edge_str] += 1
                return 

        for (_, edge_str) in incoming:
            self.st_edge_success[edge_str] += 1

    def apply_staged(self):
        for k, v in self.st_node_planned.items():
            self.node_planned[k] += v
        for k, v in self.st_node_success.items():
            self.node_success[k] += v
        for k, v in self.st_edge_planned.items():
            self.edge_planned[k] += v
        for k, v in self.st_edge_success.items():
            self.edge_success[k] += v
        self.st_node_planned.clear()
        self.st_node_success.clear()
        self.st_edge_planned.clear()
        self.st_edge_success.clear()

    def render_text(self) -> str:
        ind_parts = []
        for n in sorted(self.independent_nodes):
            rate = self._rate(self.node_success[n], self.node_planned[n])
            ind_parts.append(f"{n} ({rate})")
        first_line = "; ".join(ind_parts)

        edge_lines = []
        for edge_list in self.layer_edges:
            decorated = []
            for e in edge_list:
                rate = self._rate(self.edge_success[e], self.edge_planned[e])
                decorated.append(f"{e} ({rate})")
            edge_lines.append("; ".join(decorated))

        return "\n".join([first_line] + edge_lines)


    def dump_jsonl(self, episode: int):
        obj = {
            "episode": episode,
            "node_planned": dict(self.node_planned),
            "node_success": dict(self.node_success),
            "edge_planned": dict(self.edge_planned),
            "edge_success": dict(self.edge_success),
        }

        with open(self.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class KnowledgeBase:
    def __init__(self, kb_dir: str = "./crafter_kb"):
        kb_dir = Path(kb_dir)
        self.subgoals = {sg["name"]: sg
                         for sg in self._read_jsonl(kb_dir /"outputs"/ "subgoals.jsonl")}
        self.entities = {e["name"]: e
                         for e in self._read_jsonl(kb_dir /"outputs"/ "entity_kb.jsonl")}
        self.graph_text: str = (kb_dir / "logs" / "subgoal_graph.txt").read_text(encoding="utf-8").strip()

        sg_raw = (kb_dir / "logs" / "subgoal_name.txt").read_text(encoding="utf-8")

        clean = sg_raw.strip()

        items = clean.lstrip("{").rstrip("}").split(",")

        # strip() 每个字符串并丢进集合；顺手过滤空串
        self.subgoal_set = {item.strip() for item in items if item.strip()}

        self.subgoal_text_set = ','.join(self.subgoal_set)

    # ---------- public API ----------
    def feasible_subgoals(self,
                          inventory: Dict[str, int],
                          visible: List[str]) -> List[str]:
        vis_set = {v.strip() for v in visible}
        feas = []
        for name, sg in self.subgoals.items():
            if self._check_prereq(sg["prerequisites"], inventory, vis_set):
                feas.append(name)
        return feas

    def entity_rag(self, entity_names: List[str]) -> str:
        lines = []
        for name in entity_names:
            info = self.entities.get(name)
            if info:
                lines.append(info)
        entity_text = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in lines)
        return entity_text

    def subgoal_rag(self, subgoal_names: Set[str]) -> str:
        """汇总子目标描述，用于给 Critic."""
        parts = []
        for n in subgoal_names:
            sg = self.subgoals.get(n)
            if sg:
                parts.append(sg)
        sg_text = "\n".join(json.dumps(obj, ensure_ascii=False) for obj in parts)
        return sg_text

    # ---------- helpers ----------
    @staticmethod
    def _read_jsonl(fp: Path) -> List[dict]:
        if not fp.exists():
            return []
        return [json.loads(line) for line in fp.open("r", encoding="utf-8")]

    @staticmethod
    def _parse_mat(mat: str) -> Tuple[str, int]:
        m = re.match(r"(\w+)\s+(\d+)", mat)
        return (m.group(1), int(m.group(2))) if m else (mat, 1)

    def _check_prereq(self, pre: Dict, inv: Dict[str, int], vis: set) -> bool:
        # materials
        for mat in pre.get("materials", []):
            item, cnt = self._parse_mat(mat)
            if inv.get(item, 0) < cnt:
                return False
        # tools
        for tool in pre.get("tools", []):
            if inv.get(tool, 0) <= 0:
                return False
        # observations
        for obs in pre.get("observations", []):
            # obs 里可能是 'grass or sand' 形式
            if " or " in obs:
                if not any(o in vis for o in obs.split(" or ")):
                    return False
            elif obs not in vis:
                return False
        return True


