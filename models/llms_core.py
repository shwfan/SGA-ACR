import json, re
from typing import List
from pathlib import Path
from typing import Tuple, List
import torch

from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from openai import OpenAI

from utils import API_KEY, BASE_URL, GPT_MODEL, LOCAL_MODEL_PATH, KnowledgeBase
from .prompt_templates import actor_first_template_ada, critic_template_ada, actor_refine_template_ada

def _grab(text: str, tag: str, default: str = "") -> str:
    """Grab body inside Tag<...>. Return default if not found."""
    m = re.search(rf"{re.escape(tag)}\s*<([^>]*)>", text, flags=re.S | re.I)
    return m.group(1).strip() if m else default


_SPLIT_RE = re.compile(r"[,\n]|(?:\s+and\s+)", re.I)
def _clean_sg_list(all_subgoals: List[str], raw_body: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for token in _SPLIT_RE.split(raw_body):
        sg = token.strip()
        if sg and sg in all_subgoals and sg not in seen:
            out.append(sg)
            seen.add(sg)
    return out


# ----------------- LLM Core -----------------
class LLMs_Core:
    def __init__(self,
                 data_path: str,
                 kb: KnowledgeBase,
                 save_interval: int = 10,
                 mode: str = "Train"):

        self.kb = kb
        self.save_interval = save_interval
        self.gen_count = 0
        self.mode = mode
        if self.mode == 'Test':
            self.save_interval = 1
        if mode == 'Test':
            self.file_path = None
        else:
            self.file_path = Path(data_path) / (
            "llms_train_data.jsonl" if mode == "Train" else "llms_test_data.jsonl"
        )


        self.actor_provider = "local"
        self.critic_provider = "local"
        self.refiner_provider = "local"

        if self.actor_provider != "local" or self.critic_provider != "local" or self.refiner_provider != "local":
            self.gpt_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        if self.actor_provider == "local" or self.critic_provider == "local" or self.refiner_provider == "local":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
            self.mod = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH, quantization_config=bnb_config, torch_dtype="auto", device_map="auto"
            )

            self.llama_pipe = pipeline("text-generation",
                                        model=self.mod,
                                        tokenizer=self.tok,
                                        max_new_tokens=500,
                                        return_full_text=False)

    # ---------- unified chat ----------
    def _chat(self, provider: str, prompt_msgs: List[dict], temperature: float) -> str:
        if provider == "local":
            prompt = self.tok.apply_chat_template(
                prompt_msgs,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False
            ).to(self.mod.device)

            outputs = self.mod.generate(prompt, max_new_tokens=500, temperature=temperature)
            out = self.tok.decode(outputs[0], skip_special_tokens=True)
        else:
            out = self._chat_gpt(prompt_msgs, temperature)
        return out
    
    @retry(stop=stop_after_attempt(50), wait=wait_random_exponential(multiplier=1, max=600))
    def _chat_gpt(self, prompt_msgs: List[dict], temperature: float) -> str:
        response = self.gpt_client.chat.completions.create(
            model=GPT_MODEL,
            messages=prompt_msgs,
            temperature=temperature,
        )
        return response.choices[0].message.content

    # ---------- wrappers ----------
    def actor_first_gpt(self, prompt):    
        return self._chat(self.actor_provider, prompt, 0.6)

    def critic_gpt(self, prompt):    
        return self._chat(self.critic_provider, prompt, 0.1)

    def actor_refine_gpt(self, prompt):    
        return self._chat(self.refiner_provider, prompt, 0.2)

    # ---------- public high‑level -----------
    def actor_first_generate(self, text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, fulfilled):
        prompt = actor_first_template_ada(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, fulfilled)
        raw = self.actor_first_gpt(prompt)
        candidate_plans, all_subgoals = self._extract_actor_blocks(raw)
        return raw, candidate_plans, all_subgoals

    def critic_generate(self, text_obs, entity_text, unachieved_text, subgoal_text_set, graph_text, actor_output,subgoal_details_text, plan_last, fulfilled):
        prompt = critic_template_ada(text_obs, entity_text, unachieved_text, subgoal_text_set,graph_text, actor_output, subgoal_details_text, plan_last, fulfilled)
        raw = self.critic_gpt(prompt)
        modify_flag, critic_feedback, top_plan, tplan_list = self._extract_critic_blocks(raw)
        return raw, modify_flag, critic_feedback, top_plan, tplan_list

    def actor_refine_generate(self, text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, top_plan, tplan_subgoals_info, critic_feedback):
        prompt = actor_refine_template_ada(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, top_plan, tplan_subgoals_info, critic_feedback)
        raw = self.actor_refine_gpt(prompt)
        final_plan, final_plan_list = self._extract_refiner_blocks(raw)
        return raw, final_plan, final_plan_list

    def ac_generate(self, text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, achieve_last) -> str:
        self.gen_count += 1

        # ---- actor ----
        actor_raw, _, all_subgoals = self.actor_first_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, achieve_last)
        subgoals_info = self.kb.subgoal_rag(all_subgoals)

        # ---- critic ----
        critic_raw, need_modify, critic_feedback, top_plan, tplan_list = self.critic_generate(text_obs, entity_text, unachieved_text, subgoal_text_set, graph_text, actor_raw, subgoals_info, plan_last, achieve_last)
        
        # ---- refiner ----
        if need_modify == 'no':
            final_plan = top_plan
            final_plan_list = tplan_list
        else:
            tplan_subgoals_info = self.kb.subgoal_rag(set(tplan_list))
            refiner_raw, final_plan, final_plan_list = self.actor_refine_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, top_plan, tplan_subgoals_info, critic_feedback)

        # ---- logging ----
        if self.gen_count % self.save_interval == 0:
            if need_modify == 'no':
                data = {
                    "text_obs": text_obs,
                    "past_plan": plan_last,
                    "ful_sg": achieve_last,
                    "unachieved": unachieved_text,
                    "graph": graph_text,
                    "actor_output": actor_raw,
                    "critic_output": critic_raw
                }

            else:
                data = {
                    "text obs": text_obs,
                    "past_plan": plan_last,
                    "ful_sg": achieve_last,
                    "unachieved": unachieved_text,
                    "graph": graph_text,
                    "actor_output": actor_raw,
                    "critic_output": critic_raw,
                    "refiner_output": refiner_raw
                }

            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, indent=4) + '\n')
            
            self.summary_text = ""

        return final_plan, final_plan_list
    
    # -------------- utils --------------
    def _extract_summary_blocks(self, raw: str) -> Tuple[str, str]:
        analysis_str = _grab(raw, "Analysis")
        fulfilled_str = _grab(raw, "Fulfilled")
        return analysis_str, fulfilled_str
    
    def _extract_actor_blocks(self, raw: str) -> tuple[str, list[str]]:
        TAGS = ("PlanA", "PlanB", "PlanC")
        plan_map, lines = {}, []

        for tag in TAGS:
            body = _grab(raw, tag)
            sg_list = _clean_sg_list(self.kb.subgoal_set, body)[:3]
            if sg_list:
                plan_map[tag] = sg_list
                lines.append(f"{tag}<{','.join(sg_list)}>")
            else:
                lines.append(f"{tag}<invalid>")

        candidate_text = "\n".join(lines)
        all_sg = sorted({sg for lst in plan_map.values() for sg in lst})

        self._last_actor_plan_map = plan_map or {}
        return candidate_text, all_sg


    def _extract_critic_blocks(self, raw: str) -> tuple[str, str, str]:
        need_modify = _grab(raw, "Need_Modify").lower()
        need_modify = "yes" if need_modify not in {"yes", "no"} else need_modify

        ranking_body = _grab(raw, "Ranking")
        first_plan = ranking_body.split(",")[0].strip() if ranking_body else ""

        sg_list = self._last_actor_plan_map.get(first_plan, [])
        if not sg_list and first_plan:
            for k, v in self._last_actor_plan_map.items():
                if k.lower() == first_plan.lower():
                    sg_list = v
                    break

        sg_str = ",".join(sg_list)

        feedback = ""
        if first_plan:
            tags = [
                rf"{re.escape(first_plan)}\s*_\s*feedback"
            ]
            pat = rf"(?i)(?:{'|'.join(tags)})\s*<\s*([^>]*)\s*>"
            m_fb = re.search(pat, raw, flags=re.S)
            if m_fb:
                feedback = m_fb.group(1).strip()

        return need_modify, feedback, sg_str, sg_list

    def _extract_refiner_blocks(self, raw: str) -> str:
        """
        Returns:
            final_plan_str: 'sg1,sg2,sg3'
        """
        final_plan_raw = _grab(raw, "Final_Plan")
        sg_list = _clean_sg_list(self.kb.subgoal_set, final_plan_raw)[:3]
        sg_text = ",".join(sg_list)
        return sg_text ,sg_list





