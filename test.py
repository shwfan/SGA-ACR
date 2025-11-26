import torch
import crafter
import framework
from config.default import register_args
import stable_baselines3 as sb3
from utils import QUERY_INTERVAL, get_fov_goal, RuleBasedSubgoalChecker, KnowledgeBase
from models.llms_core import LLMs_Core
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import json
import imageio.v3 as iio


N_EPISODE = 20
MODEL_PATH = ""
TIME_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join("./results", f"test_{TIME_TAG}")

save_image = False
run_path = Path("./results") / f"test_{TIME_TAG}"
os.makedirs(RUN_DIR, exist_ok=True)

kb = KnowledgeBase("./crafter_kb")
sg_checker = RuleBasedSubgoalChecker(kb.subgoal_set)
llms_core = LLMs_Core(data_path='', save_interval=1, mode='Test', kb=kb)


helper = framework.helpers.TrainingHelper(register_args=register_args)
if not hasattr(helper.args, "wrap_normalizeimg"):
    helper.args.wrap_normalizeimg = False

def extract_with_imageio(video_path, indices, out_dir, prefix="frame"):
    out = Path(out_dir)
    with iio.imopen(video_path, "r", plugin="pyav") as v:
        for i in indices:
            frame = v.read(index=i)
            iio.imwrite(out / f"{prefix}_{i:06d}.png", frame)

def make_episode_env(ep_dir):
    env = crafter.Env()
    env = crafter.Recorder(
        env,
        ep_dir,
        helper=None,
        save_stats=True,
        save_video=True,
        save_episode=False,
        log_every_n_episodes=1,
    )
    return env

_tmp_env = crafter.Env()
model = sb3.PPO(helper, _tmp_env, verbose=0)
ckpt = torch.load(MODEL_PATH)
model.policy.load_state_dict(ckpt["model_state_dict"])
_tmp_env.close()                            
del _tmp_env

def run_one_episode(ep_dir):
    env = make_episode_env(ep_dir)
    model.set_env(env)                         

    llms_core.file_path = Path(ep_dir) / ("llms_test_data.jsonl")

    obs = env.reset()
    info = {
        'inventory': env._env._player.inventory.copy(),
        'semantic': env._env._sem_view(),
        'player_pos': env._env._player.pos,
        'time_step': env._env._step,
        'achievements': env._env._player.achievements.copy()
    }
    sg_checker.clear_state()
    subgoal_text_set = kb.subgoal_text_set
    text_obs, _, entity_list = get_fov_goal(info)
    graph_text = kb.graph_text
    entity_text = kb.entity_rag(entity_list)
    unachieved = [name for name, achieved in info['achievements'].items() if achieved == 0]
    unachieved_text = ",".join(unachieved)

    plan_last = ''
    fulfilled = []
    inst, plan_sg_list = llms_core.ac_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, fulfilled)
    sg_checker.reset_plan(plan_sg_list, info)

    sim_buffer = []
    env_steps, rew, done = 0, 0.0, False
    frame_list = []
    frame_list.append(env_steps)

    while not done:
        action, _, _ = model.policy.predict(obs, instruction=inst, deterministic=True, text_obs=text_obs)
        obs, reward, done, info = env.step(action)
        env_steps += 1
        rew += reward
        _, text_obs, entity_list = sg_checker.step(info)


        if len(sg_checker.done) >= 3 or sg_checker.plan_steps >= QUERY_INTERVAL:
            frame_list.append(env_steps)
            fulfilled = list(sg_checker.done)
            entity_text = kb.entity_rag(entity_list)
            unachieved = [name for name, achieved in info['achievements'].items() if achieved == 0]
            unachieved_text = ",".join(unachieved)

            inst, plan_sg_list = llms_core.ac_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, inst, fulfilled)
            sg_checker.reset_plan(plan_sg_list, info)
            score = len(fulfilled) / 3.0
            sim_buffer.append(score)


    ep_stats = {'length': env_steps, 'reward': round(rew, 1)}
    for key, value in info['achievements'].items():
        ep_stats[f'achievement_{key}'] = value
    
    video_path = str(env._directory / (env._env.episode_name + '.mp4'))
    env.close()

    if save_image: 
        extract_with_imageio(video_path, frame_list, ep_dir)

    ep_sim = np.mean(sim_buffer)
    return rew, ep_sim, env_steps, ep_stats

def trimmed(arr, trim_ratio=1):
    arr = np.asarray(arr)
    if arr.size <= 2 * trim_ratio: 
        return arr
    sorted_arr = np.sort(arr)
    return sorted_arr[trim_ratio:-trim_ratio]

def main():
    ep_rew_buffer, ep_sim_buffer, ep_len_buffer = [], [], []
    stats_file = (run_path / 'stats.jsonl').open('a')

    for ep_idx in range(1, N_EPISODE + 1):
        ep_dir = os.path.join(RUN_DIR, f"episode_{ep_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        rew, sim, length, ep_status = run_one_episode(ep_dir)

        ep_rew_buffer.append(rew)
        ep_sim_buffer.append(sim)
        ep_len_buffer.append(length)

        stats_file.write(json.dumps(ep_status) + '\n')

        print(f"[EP {ep_idx:03d}] reward={rew:.2f}, sim={sim:.3f}, length={length}")

    trimmed_rew = trimmed(ep_rew_buffer)
    trimmed_sim = trimmed(ep_sim_buffer)
    trimmed_len = trimmed(ep_len_buffer)

    mean_reward, std_reward = np.mean(trimmed_rew), np.std(trimmed_rew)
    mean_sim, std_sim = np.mean(trimmed_sim), np.std(trimmed_sim)
    mean_len, std_len = np.mean(trimmed_len), np.std(trimmed_len)

    summary = (
        f"Mean reward  = {mean_reward:.3f} ± {std_reward:.3f}\n"
        f"Mean sim     = {mean_sim:.3f} ± {std_sim:.3f}\n"
        f"Mean length  = {mean_len:.3f} ± {std_len:.3f}\n"
    )
    print("\n" + summary)

    with open(os.path.join(RUN_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

if __name__ == "__main__":
    main()