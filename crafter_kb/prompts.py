def build_subgoal_prompts(paper: str, code: str) -> list:
    system_prompt = (
        "You are an expert reinforcement learning knowledge engineer.\n"
        "Task: From provided Crafter game paper & code, first derive a FINITE yet SUFFICIENT set of subgoals for planning and guiding RL policy to get higher rewards in this game, and then complete subgoal information as required.\n"
        "Strictly follow instructions.\n\n"
        "[OBJECTIVE]\n"
        f"- Produce at most 35 subgoals (ideal: 20-35). Minimize redundancy while covering full game progression.\n"
        "- Each subgoal should be atomic and executional so that a 40-step macro plan can combine 3 subgoals.\n"
        "- Dependencies of each subgoal may be AND-only, OR-only, or a mix; represent each prerequisite pairwise as AND or OR.\n\n"
        "[FIELD DEFINITIONS]\n"
        "- name: lowercase snake_case unique identifier\n"
        "- description: One sentence to describe this subgoal.\n"
        "- prerequisites: The prerequisites for executing this subgoal.(Can be empty list if none) \n"
        "- subgoals: List of all the prerequisite subgoals of this subgoal. If just one prerequisite subgoal the \"logic\" use \"SINGLE\". If no need to any prerequisite subgoals, use empty list. If this subgoal can either depend on prerequisites or be executed independently, you can add an optional field in the \"subgoals\" list: {\"subgoal\": \"null\", \"logic\": \"OR\", \"relationship_description\": \"this subgoal can also be independent\"}\n\n"
        "- effects: State changes after completing this subgoal.(Can be empty list if none)\n"
        "[CONSTRAINTS]\n"
        "- ***Your output and reasoning must strictly adhere to the provided paper and code content. Do not introduce any external information!***\n"
        "- Make sure all of the subgoals are useful to guiding RL policy to gain higher rewards.\n"
        "- Merge semantic duplicates; choose canonical naming.\n"
        "- Avoid ambiguous, overlapping or ill-defined subgoals(e.g. maintain somrthing), ensure crisp success boundary.\n"
        "- Do NOT wrap JSON objects in an array. Output one JSON per line.\n"
        "[SCHEMA REMINDER]\n"
        "{id,description,prerequisites: {materials:[],tools:[],observations:[]},subgoals:[{subgoal,logic, relationship_description}], effects:{inventory:[],status:[],obsevations:[]}}\n"
        "[OUTPUT FORMAT STRICT]\n"
        "```jsonl_subgoals_names\\n<ONE JSON OBJECT PER LINE>\\n```\n"
        "```jsonl_subgoals\\n<ONE JSON OBJECT PER LINE>\\n```\n"
        "\n[OUTPUT EXAMPLES]\n"
        "```jsonl_subgoal_names\n"
        "{\"name\":\"collect_wood\"}\n"
        "{\"name\":\"place_table\"}\n"
        "{\"name\":\"collect_water\"}\n"
        "{\"name\":\"make_wood_pickaxe\"}\n"
        "```\n"
        "```jsonl_subgoals\n"
        "{\"name\":\"collect_wood\",\"description\":\"Gather basic wood logs\",\"prerequisites\":{\"materials\":[],\"tools\":[],\"observations\":[\"tree\"]},\"subgoals\":[],\"effects\":{\"inventory\":[\"wood increased\"],\"status\":[],\"obsevations\":[]}}\n"
        "{\"name\":\"place_table\",\"description\":\"crafer a table and place it, a basic workbench\",\"prerequisites\":{\"materials\":[\"wood 2\"],\"tools\":[],\"observations\":[\"grass or sand or path\"]},\"subgoals\":[{\"subgoal\":\"Collect Wood\",\"logic\":\"SINGLE\",\"relationship_description\":\"wood is required to craft a table\"}],\"effects\":{\"inventory\":[\"wood decreased 2\"],\"status\":[],\"obsevations\":[\"table\"]}}\n"
        "{\"name\":\"collect_water\",\"description\":\"find and approch water resource and collect water.\",\"prerequisites\":{\"materials\":[],\"tools\":[],\"observations\":[]},\"subgoals\":[],\"effects\":{\"inventory\":[],\"status\":[\"water increased\"],\"obsevations\":[\"water\"]}}\n"
        "{\"name\":\"make_wood_pickaxe\",\"description\":\"craft a wood pickaxe, a basic tool for resource collection.\",\"prerequisites\":{\"materials\":[\"wood 1\"],\"tools\":[],\"observations\":[\"table\"]},\"subgoals\":[{\"subgoal\":\"collect_wood\",\"logic\":\"AND\",\"relationship_description\":\"wood is required to craft a wood pickaxe.\"},{\"subgoal\":\"place_table\",\"logic\":\"AND\",\"relationship_description\":\"table is required to craft a wood pickaxe.\"}],\"effects\":{\"inventory\":[\"wood pickaxe increase 1\",\"wood decreased 1\"],\"status\":[],\"obsevations\":[]}}\n"
        "```\n"
        "[END OF OUTPUT EXAMPLES]\n"
        "Follow the structure & style of examples but scale to the full environment."
    )

    user_prompt = (
        "[PAPER]\n"
        f"{paper}\n\n"
        "[CODE]\n"
        f"{code}\n\n"
    )

    gpt_prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    ]

    return gpt_prompt


def build_entity_prompts(paper: str, code: str, subgoals_jsonl: str) -> list:

    system_prompt = (
        "You are an expert game environment knowledge distiller.\n"
        "Task: Using Crafter game paper and codes + the final subgoal JSONL, build an entity knowledge base for planning.\n\n"
        "[ENTITY DEFINITION]\n"
        "Entity = any observable or interactable environmental element\n\n"
        "[FIELD DEFINITIONS]\n"
        "- name: lowercase snake_case unique identifier\n"
        "- type: one of {material, resource, object, and creature}.\n"
        "- knowledge: List of concise,clear tips or usages useful for planning subgoals.\n"
        "- related_subgoals: list of subgoal name where this entity is **required, produced, or strategically affects subgoal planning**.\n\n"
        "[CONSTRAINTS]\n"
        "- ***Your output and reasoning must strictly adhere to the provided paper and code content. Do not introduce any external information!***\n"
        "- Only include entities that actually influence at least one subgoal decision (avoid noise).\n"
        "- No duplicate names.\n"
        "- Every related_subgoal MUST exist in provided subgoals\n"
        "- Keep each knowledge tip atomic; no multi-sentence paragraphs.\n\n"
        "- Do NOT wrap JSON objects in an array. Output one JSON per line.\n"
        "[SCHEMA REMINDER]\n"
        "{name,type,knowledge:[],related_subgoals:[]}\n"
        "[OUTPUT FORMAT STRICT]\n"
        "```jsonl_entities\\n<ONE JSON OBJECT PER LINE>\\n```\n"
        "\n[OUTPUT EXAMPLES]\n"
        "```jsonl_entities\n"
        "{\"name\":\"tree\",\"type\":\"resources\",knowledge:[\"Wood can be collected from trees.\",\"Wood collected from trees can be used to craft wooden objects.\"],related_subgoals:[\"collect_wood\",\"place_table\",\"make_wood_pickaxe\",\"make_wood_sword\"]}\n"
        "```\n"
        "[END OF OUTPUT EXAMPLES]\n"
    )

    user_prompt = (
        "[PAPER]\n"
        f"{paper}\n\n"
        "[CODE]\n"
        f"{code}\n\n"
        "[AVAILABLE_SUBGOALS_JSONL]\n"
        f"{subgoals_jsonl}\n\n"
    )

    gpt_prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    ]

    return gpt_prompt