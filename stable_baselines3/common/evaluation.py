import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from utils import QUERY_INTERVAL, ACTIONS_NAME, text_transition, get_fov_goal, fulfilled_score, RuleBasedSubgoalChecker

def evaluate_ada_policy(
    llms_core,
    kb,
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Use for AdaRefiner ppo policy(---Need To Query!---)
    """

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    sg_checker = RuleBasedSubgoalChecker(kb.subgoal_set)
    if model.graph_weight == 'ada':
        graph_text = model.sg_graph.render_text()
    else:
        graph_text = kb.graph_test 
    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_sim_score = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    obs = env.reset()
    info = {
        'inventory': env.envs[0].env._env._env._player.inventory.copy(),
        'semantic': env.envs[0].env._env._env._sem_view(),
        'player_pos': env.envs[0].env._env._env._player.pos,
        'time_step': env.envs[0].env._env._env._step,
        'achievements': env.envs[0].env._env._env._player.achievements.copy()
    }

    sg_checker.clear_state()
    subgoal_text_set = ','.join(sg_checker.undone_set.copy())
    text_obs, _, entity_list = get_fov_goal(info)
    entity_text = kb.entity_rag(entity_list)

    unachieved = [name for name, achieved in info['achievements'].items() if achieved == 0]
    unachieved_text = ",".join(unachieved)
    plan_last = ''
    fulfilled = []

    inst, subgoal_list = llms_core.ac_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, fulfilled)
    sg_checker.reset_plan(subgoal_list, info)

    sim_buffer = []
    env_steps = 0

    while (episode_counts < episode_count_targets).any():
        actions, _, _ = model.policy.predict(obs, instruction=inst, deterministic=deterministic, text_obs=text_obs)
        obs, rewards, dones, infos = env.step(actions)
        env_steps += 1

        # Rule-based subgoal checker
        _, text_obs, entity_list = sg_checker.step(infos[0])
        if len(sg_checker.done) >= 3 or sg_checker.plan_steps >= QUERY_INTERVAL:
            fulfilled = list(sg_checker.done)
            entity_text = kb.entity_rag(entity_list)
            subgoal_text_set = ','.join(sg_checker.undone_set.copy())
            unachieved = [name for name, achieved in infos[0]['achievements'].items() if achieved == 0]
            unachieved_text = ",".join(unachieved)

            inst, plan_sg_list = llms_core.ac_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, inst, fulfilled)
            sg_checker.reset_plan(plan_sg_list, infos[0])
            score = len(fulfilled) / 3.0
            sim_buffer.append(score)

        
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    # Rule-based subgoal checker
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_sim_score.append(np.mean(sim_buffer))
                    sim_buffer = []
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    env_steps = 0
                    info = {
                        'inventory': env.envs[0].env._env._env._player.inventory.copy(),
                        'semantic': env.envs[0].env._env._env._sem_view(),
                        'player_pos': env.envs[0].env._env._env._player.pos,
                        'time_step': env.envs[0].env._env._env._step,
                        'achievements': env.envs[0].env._env._env._player.achievements.copy()
                    }
                    sg_checker.clear_state()
                    text_obs, _, entity_list = get_fov_goal(info)
                    entity_text = kb.entity_rag(entity_list)
                    subgoal_text_set = ','.join(sg_checker.undone_set.copy())
                    unachieved = [name for name, achieved in info['achievements'].items() if achieved == 0]
                    unachieved_text = ",".join(unachieved)
                    plan_last = ''
                    fulfilled = []

                    inst, plan_sg_list= llms_core.ac_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, fulfilled)
                    sg_checker.reset_plan(plan_sg_list, info)

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    mean_sim = np.mean(episode_sim_score)
    std_sim = np.std(episode_sim_score)
    return mean_reward, std_reward, mean_length, std_length, mean_sim, std_sim

def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Use for pure ppo policy
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_observations = []
    episode_attn_maps = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states, attn_maps = model.policy.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        episode_observations.append(torch.Tensor(observations).to(model.device) / 255.0)
        episode_attn_maps.append(attn_maps)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_observations, episode_attn_maps
    return mean_reward, std_reward, episode_observations, episode_attn_maps