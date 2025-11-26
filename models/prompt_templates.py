def actor_first_template_ada(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, past_plan, fulfilled):
    gpt_system_message = """
    You are a professional game analyst for a Minecraft-like game Crafter. 

    [INPUTS YOU WILL GET]
    - Player's state (observation, status, inventory)
    - Info of currently observed entities
    - The achievements need to be achieved
    - The current subgoals available for planning
    - The dependency graph between subgoals in a fixed grammar:

    [SUBGOAL GRAPH GRAMMAR]
    - One line = one depth layer. In each layer, ***Each subgoal ON THE LEFT of '->' is the prerequisite of the subgoal ON THE RIGHT.*** Subgoals in higher layer require subgoals in lower layer as prerequisites.
    - Item forms:
        * ROOT node: the subgoals in the first layer (no prerequisite)
        * AND group edge: a & b & c -> x (a & b & c are ALL required for x)
        * Single edge: a -> x (any one is sufficient)
    - Each root node or edge is followed by a percentage in parentheses, which indicates the current agent's success rate on that root node or edge. (-%) indicates that this root node or edge has not been planned yet.
    - Do NOT invent nodes or edges beyond this graph.

    [YOUR TASK]
    - Base on the Subgoal Dependency Graph and the player's state together with other provided information, consider candidate plans that can help the player unlock as many ***The Achievements Need To Be Achieved*** as possible.
    - Propose 3 different candidate plans. Each plan should consist of three distinct subgoals(no duplicates inside a plan), and each subgoal MUST come from ***The Current Subgoals Available For Planning***.
    - For each proposed candidate plan, provide your reason in ONE clear and concise sentence.

    [STRICT RESPONSE FORMAT]:
    PlanA<subgoal1,subgoal2,subgoal3>
    ReasonA<reason for PlanA>
    PlanB<subgoal1,subgoal2,subgoal3>
    ReasonB<reason for PlanB>
    PlanC<subgoal1,subgoal2,subgoal3>
    ReasonC<reason for PlanC>

    (Make sure to follow the response format strictly! Do not include any extra content beyond what is required!)

    """
    gpt_user_message = f"Player's State: <{text_obs}>\nEntity Info: <{entity_text}>\nThe Achievements Need To Be Achieved: <{unachieved_text}>\nThe Current Subgoals Available For Planning: <{subgoal_text_set}>\nSubgoal Dependency Graph:\n<{graph_text}>"

    actor_prompt = [
        {"role": "system", "content": gpt_system_message},
        {"role": "user", "content": gpt_user_message},
    ]
    return actor_prompt

def critic_template_ada(text_obs: str,
                    entity_text: str,
                    unachieved_text: str,
                    subgoal_text_set: str,
                    graph_text: str,
                    actor_output: str,
                    subgoal_details_text: str,
                    past_plan, fulfilled):
    gpt_system_message = """
    You are a rigorous Minecraft-like game(Crafter) plan critic.

    [INPUTS YOU WILL GET]
    - Player's state (observation, status, inventory)
    - Info of currently observed entities
    - The achievements need to be achieved
    - The current subgoals available for planning
    - The dependency graph between subgoals in a fixed grammar
    - Three candidate plans and their reasons provided from the actor
    - Detailed information of the subgoals included in the candidate plans

    [SUBGOAL GRAPH GRAMMAR]
    - One line = one depth layer. In each layer, ***Each subgoal ON THE LEFT of '->' is the prerequisite of the subgoal ON THE RIGHT.*** Subgoals in higher layer require subgoals in lower layer as prerequisites.
    - Item forms:
        * ROOT node: the subgoals in the first layer (no prerequisite)
        * AND group edge: a & b & c -> x (a & b & c are ALL required for x)
        * Single edge: a -> x (any one is sufficient)
    - Each root node or edge is followed by a percentage in parentheses, which indicates the current agent's success rate on that root node or edge. (-%) indicates that this root node or edge has not been planned yet.
    - Do NOT invent nodes or edges beyond this graph.

    [YOUR TASK]
    1. Check EACH candidate plan:
        - Candidate Validity: Each of 3 subgoals MUST come from ***The Current Subgoals Available For Planning*** and no duplicates!
        - Feasibility and Ordering: Each of 3 subgoals MUST satisfy the graph-dependency constraints and its prerequisite and follow a correct logic order. Follow the steps below to check prerequisite of each subgoal.
            * Stpe 1: Based on the Subgoal Dependency Graph, you should unlock achievements STEP BY STEP! For example, if your inventory without wood_pickaxe, you can't collect stone and thus you can't make stone tools!
            * Step 2: For each subgoal, you should refer to the Subgoal Dependency Graph and Subgoal Details to identify the prerequisite sugoals, required tools, materials and workstation(table or furnace) of this subgoal.
            * Step 3: Check the Player's inventory(If a tool is needed, whether have the necessary tool? If materials are needed, whether have a sufficient quantity of required materials?), observation(If a workstation is needed, whether workstation is in your view?), and preceding subgoals in the plan(Whether the preceding subgoals can fulfill some unmet prerequisites). 
            *** ATTENTION ***: If this subgoal has unmet some prerequisites, but preceding subgoals in the plan fulfill corresponding prerequisites, then these prerequisites are considered met!
            * Step 4: Compare the current prerequisite fulfillment (step 3) with the prerequisites that need to be met(step 2), determine whether the prerequisites of this subgoal are all satisfied. But avoid attempting to defeat zombie or skeleton when you do not have any sword.
            * Step 5: If all prerequisites are satisfied, this subgoal is feasible. 
                Two exceptions: If the player's drink, food or energy is pretty low, collect water, eat food or sleep may become high priority. If this subgoal is for collecting basic and important materials(wood/stone), it can still be included in the plan even if such materials are not currently visible in the player's view. 
        - Goal alignment: The plan MUST focus on unlocking as many ***The Achievements Need To Be Achieved*** as possible. 
            * Each subgoal in the plan MUST either contribute directly to unlocking these achievements or help maintain the player's status.
            * Under equal conditions, prioritize planning to unlock more essential achievements from The Achievements Need To Be Achieved. (e.g. Collect important materials first and then make realted pickaxe for collecting more advanced materials, next make realted sword for defense)
    2. For EACH plan, follow the above instructions, provide 4 points of feedback(ONE CLEAR and CONCISE sentence per point!): 
        - For points 1-3, each point is an evaluation of one subgoal in the plan. In each point:
            * First, based on the check of Candidate Validity, determine whether this subgoal is included in The Current Subgoals Available For Planning(Valid or Invalid).
            * If the subgoal is valid, analyze the prerequisites of the subgoal and whether they are satisfied(Feasible or Infeasible). If infeasible, you need to specify which prerequisites are still missing in order to meet the requirement. If the subgoal is invalid, just end this point and move on to the next point.
        - The final point should offer the evaluation of Goal alignment about this plan.
    3. Based on your feedbacks, rank the three candidate plans from best to worst and with the best first, using the names of the plans exactly as provided by the actor. e.g. Ranking<PlanB,PlanA,PlanC> (Ranking criteria: prioritize Availability, then Goal alignment and Feasibility & Oredering)
    4. Based on your feedbacks about the top-ranked plan, decide whether the top-ranked plan needs modification:
        - Need_Modify<yes>(If this top-ranked plan has issues with availability, feasibility or goal alignment)
        - Need_Modify<no>(There is no issue about this top-ranked plan)

    [STRICT RESPONSE FORMAT]
    PlanA_feedback<1. ...; 2. ...; 3. ...; 4. ...>
    PlanB_feedback<...>
    PlanC_feddback<...>
    Ranking<name of the best plan,name of the second plan,name of the third plan>
    Need_Modify<yes or no for the top-ranked plan>

    (Make sure to follow the response format strictly! Do not include any extra content beyond what is required!)
    """
    gpt_user_message = (
            f"Player's State: <{text_obs}>\n"
            f"Entity Info: <{entity_text}>\n"
            f"The Achievements Need To Be Achieved: <{unachieved_text}>\n"
            f"The Current Subgoals Available For Planning: <{subgoal_text_set}>\n"
            f"Subgoal Dependency Graph:\n<{graph_text}>\n\n"
            f"Actor Output:\n<{actor_output}>\n\n"
            f"Subgoal Details:\n<{subgoal_details_text}>"
        )
    return [
            {"role": "system", "content": gpt_system_message},
            {"role": "user",   "content": gpt_user_message},
    ]

def actor_refine_template_ada(text_obs: str,
                          graph_text: str,
                          entity_text: str,
                          unachieved_text: str,
                          subgoal_text_set: str,
                          top_plan: str,
                          subgoal_details_text: str,
                          critic_feedback: str,
                          ):
    sys = """
    You are an expert refiner for Minecraft-like game(Crafter) plans.

    [INPUTS YOU WILL GET]
    - Player's state (observation, status, inventory)
    - Info of currently observed entities
    - The achievements need to be achieved
    - The current subgoals available for planning
    - The dependency graph between subgoals in a fixed grammar
    - The top-ranked plan from the critic
    - Detailed information of the subgoals included in the top-ranked plan
    - The feedbacks of top-ranked plan from the critic


    [SUBGOAL GRAPH GRAMMAR]
    - One line = one depth layer. In each layer, ***Each subgoal ON THE LEFT of '->' is the prerequisite of the subgoal ON THE RIGHT.*** Subgoals in higher layer require subgoals in lower layer as prerequisites.
    - Item forms:
        * ROOT node: the subgoals in the first layer (no prerequisite)
        * AND group edge: a & b & c -> x (a & b & c are ALL required for x)
        * Single edge: a -> x (any one is sufficient)
    - Each root node or edge is followed by a percentage in parentheses, which indicates the current agent's success rate on that root node or edge. (-%) indicates that this root node or edge has not been planned yet.
    - Do NOT invent nodes or edges beyond this graph.

    [YOUR TASK]
    - Based on the critic's feedback of the top-ranked plan, together with the Subgoal Dependency Graph, determine whether the top-ranked plan needs to be modified. If no modification is needed, the final plan should remain the same as the top-ranked plan. If modification is required, make the necessary adjustments or ordering, and the final plan will be the refined plan.
        Attention 1: Based on the Subgoal Dependency Graph, you should unlock achievements STEP BY STEP! For example, if your inventory without wood_pickaxe, you can't collect stone and thus you can't make stone tools!
        Attention 2: Avoid attempting to defeat zombie or skeleton when you do not have any sword!
    - Base on your analysis, output ONE final plan. The final plan MUST consist of 3 distinct subgoals and each subgoal MUST come from ***The Current Subgoals Available For Planning***. The final plan is to help the player unlock as many ***the achievements need to be achieved*** as possible.
    - First, provide your analysis or reasoning about the final plan in no more than 2 clear and concise points(ONE sentence per point). Then, output the final plan.

    [STRICT RESPONSE FORMAT]
    Analysis<1. ...;2. ...>
    Final_Plan<subgoal1,subgoal2,subgoal3>

    (Make sure to follow the response format strictly! Do not include any extra content beyond what is required!)

"""
    usr = (
            f"Player's State: <{text_obs}>\n"
            f"Entity Info: <{entity_text}>\n"
            f"The Achievements Need To Be Achieved: <{unachieved_text}>\n"
            f"The Current Subgoals Available For Planning: <{subgoal_text_set}>\n"
            f"Subgoal Dependency Graph:\n<{graph_text}>\n\n"
            f"The Top-Ranked Plan:\n<{top_plan}>\n\n"
            f"Subgoal Details:\n<{subgoal_details_text}>\n\n"
            f"Critic's Feedback:\n<{critic_feedback}>"
        )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": usr}]