import datetime
import json
import pathlib
import warnings

import imageio
import numpy as np
from framework.visualize.plot import Video


class Recorder:

    def __init__(self, env, directory, helper=None, env_prefix='', save_stats=True, save_video=True, save_episode=True, video_size=(512, 512), log_every_n_episodes=None):
        if directory and save_stats:
            env = StatsRecorder(env, directory, helper, env_prefix, log_every_n_episodes)
        if directory and save_video:
            env = VideoRecorder(env, directory, helper, env_prefix, video_size)
        if directory and save_episode:
            env = EpisodeRecorder(env, directory, helper)
        self._env = env
    
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)


class StatsRecorder:

    def __init__(self, env, directory, helper=None, env_prefix='', log_every_n_episodes=None):
        self._env = env
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._file = (self._directory / 'stats.jsonl').open('a')
        self._length = None
        self._reward = None
        self._unlocked = None
        self._stats = None
        self._env_prefix = env_prefix
        self._helper = helper
        if self._helper is not None:
            if log_every_n_episodes is None:
                self._log_every_n_episodes = helper.args.log_every_n_episodes
            else:
                self._log_every_n_episodes = log_every_n_episodes
        self._episode_cnt = 0


        self._logs_cnt = 0
        self._logs_scr = {}

        #----------------------------- Recover -------------------------#
        # self._logs_scr = {
        #     "ach_scr_collect_coal": 28.147983062179637,
        #     "ach_scr_collect_diamond": 0.0,
        #     "ach_scr_collect_drink": 55.4045018943616,
        #     "ach_scr_collect_iron": 0.958324047247604,
        #     "ach_scr_collect_sapling": 84.31022955203915,
        #     "ach_scr_collect_stone": 50.45687541787376,
        #     "ach_scr_collect_wood": 92.80142634276812,
        #     "ach_scr_defeat_skeleton": 15.73434365946067,
        #     "ach_scr_defeat_zombie": 57.78916870960537,
        #     "ach_scr_eat_cow": 57.38800980610652,
        #     "ach_scr_eat_plant": 0.0445732114998886,
        #     "ach_scr_make_iron_pickaxe": 0.0,
        #     "ach_scr_make_iron_sword": 0.0,
        #     "ach_scr_make_stone_pickaxe": 10.965010028972593,
        #     "ach_scr_make_stone_sword": 2.4515266324938674,
        #     "ach_scr_make_wood_pickaxe": 71.18341876532197,
        #     "ach_scr_make_wood_sword": 64.0517049253397,
        #     "ach_scr_place_furnace": 28.28170269667927,
        #     "ach_scr_place_plant": 81.5912636505465,
        #     "ach_scr_place_stone": 47.49275685313121,
        #     "ach_scr_place_table": 87.27434811678195,
        #     "ach_scr_wake_up": 44.595498105638576
        # }
        # self._logs_cnt = 4487

        self._logs_scr = {
            "ach_scr_collect_coal": 10.921217173822411,
            "ach_scr_collect_diamond": 0.0,
            "ach_scr_collect_drink": 49.1871613172156,
            "ach_scr_collect_iron": 0.08336807002917877,
            "ach_scr_collect_sapling": 89.66235931638182,
            "ach_scr_collect_stone": 28.220091704877046,
            "ach_scr_collect_wood": 87.11963318049236,
            "ach_scr_defeat_skeleton": 4.001667361400588,
            "ach_scr_defeat_zombie": 44.31012922050858,
            "ach_scr_eat_cow": 43.476448520216785,
            "ach_scr_eat_plant": 0.0,
            "ach_scr_make_iron_pickaxe": 0.0,
            "ach_scr_make_iron_sword": 0.0,
            "ach_scr_make_stone_pickaxe": 2.2926219258024174,
            "ach_scr_make_stone_sword": 0.29178824510212614,
            "ach_scr_make_wood_pickaxe": 53.105460608586824,
            "ach_scr_make_wood_sword": 51.146310962901175,
            "ach_scr_place_furnace": 6.127553147144644,
            "ach_scr_place_plant": 85.9524802000838,
            "ach_scr_place_stone": 26.385994164235054,
            "ach_scr_place_table": 78.1575656523554,
            "ach_scr_wake_up": 66.52771988328468
        }
        self._logs_cnt = 2399

        # self._logs_scr = {
        #     "ach_scr_collect_coal": 22.854561878952055,
        #     "ach_scr_collect_diamond": 0.0,
        #     "ach_scr_collect_drink": 54.268292682926756,
        #     "ach_scr_collect_iron": 0.5645889792231248,
        #     "ach_scr_collect_sapling": 90.37940379403803,
        #     "ach_scr_collect_stone": 46.36404697380314,
        #     "ach_scr_collect_wood": 91.12466124661228,
        #     "ach_scr_defeat_skeleton": 10.049683830171649,
        #     "ach_scr_defeat_zombie": 59.55284552845522,
        #     "ach_scr_eat_cow": 56.63956639566398,
        #     "ach_scr_eat_plant": 0.13550135501354993,
        #     "ach_scr_make_iron_pickaxe": 0.0,
        #     "ach_scr_make_iron_sword": 0.0,
        #     "ach_scr_make_stone_pickaxe": 10.207768744354105,
        #     "ach_scr_make_stone_sword": 5.171635049683828,
        #     "ach_scr_make_wood_pickaxe": 69.10569105691036,
        #     "ach_scr_make_wood_sword": 64.92773261065953,
        #     "ach_scr_place_furnace": 9.078590785907878,
        #     "ach_scr_place_plant": 88.32429990966602,
        #     "ach_scr_place_stone": 45.05420054200558,
        #     "ach_scr_place_table": 85.41102077687421,
        #     "ach_scr_wake_up": 87.10478771454339
        # }
        # self._logs_cnt = 4428


        #----------------------------------------------------------------#


        self._reward_mean = 0
        self._length_mean = 0

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)
    
    def reset(self):
        obs = self._env.reset()
        self._length = 0
        self._reward = 0
        self._unlocked = None
        self._stats = None
        return obs
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._length += 1
        self._reward += info['reward']
        if done:
            self._log_jsonl(info)
            self._episode_cnt += 1
            if self._helper is not None:
                self._log_helper_record(info)  # recorded at every step to match manual scores/rewards
                if self._episode_cnt % self._log_every_n_episodes == 0:
                    self._log_helper(info)
        return obs, reward, done, info
    
    def _save(self):
        self._file.write(json.dumps(self._stats) + '\n')
        self._file.flush()

    def _log_jsonl(self, info):
        self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
        for key, value in info['achievements'].items():
            self._stats[f'achievement_{key}'] = value
        self._save()
    
    def _log_helper_record(self, info):
        if self._logs_cnt == 0:
            for key, value in info['achievements'].items():
                self._logs_scr[f'ach_scr_{key}'] = 0
        self._logs_cnt += 1
        n = self._logs_cnt
        self._reward_mean = self._reward_mean * (n - 1) / n + round(self._reward, 1) / n
        self._length_mean = self._length_mean * (n - 1) / n + self._length / n
        for key, value in info['achievements'].items():
            scr_key = f'ach_scr_{key}'
            self._logs_scr[scr_key] = (n - 1) / n * self._logs_scr[scr_key] + int(value>=1) * 100 / n

    def _log_helper(self, info):
        logs = {
            'length': self._length,
            'length_mean': self._length_mean,
            'reward': round(self._reward, 1),
            'reward_mean': self._reward_mean,
            'episodes': self._episode_cnt,
        }
        for key, value in info['achievements'].items():
            logs[f'ach_cnt_{key}'] = value
            scr_key = f'ach_scr_{key}'
            logs[scr_key] = self._logs_scr[scr_key]
        logs['crf_scr'] = self._crafter_score()
        prefix_logs = {f"{self._env_prefix}/{k}": v for k, v in logs.items()}
        self._helper.log(prefix_logs, step=self._helper.state.step)
    
    def _crafter_score(self):
        # Geometric mean with an offset of 1%.
        with warnings.catch_warnings():  # Empty seeds become NaN.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            percentages = np.array(list(self._logs_scr.values()))
            scores = np.exp(np.nanmean(np.log(1 + percentages))) - 1
        return scores


class VideoRecorder:

    def __init__(self, env, directory, helper, env_prefix, size=(512, 512)):
        if not hasattr(env, 'episode_name'):
            env = EpisodeName(env)
        self._env = env
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._size = size
        self._frames = None
        self._env_prefix = env_prefix
        self._helper = helper
    
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)
    
    def reset(self):
        obs = self._env.reset()
        self._frames = [self._env.render(self._size)]
        return obs
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)    
        self._frames.append(self._env.render(self._size))
        if done:
            self._save()
        return obs, reward, done, info
    
    def _save(self):
        # Write to disc
        filename = str(self._directory / (self._env.episode_name + '.mp4'))
        imageio.mimsave(filename, self._frames)
        # Wandb/tensorboard write
        if self._helper:
            logs = {f'{self._env_prefix}/video_render_{self._env.episode_name}': Video(np.array(self._frames, dtype=np.uint8))}
            self._helper.log(logs, step=self._helper.state.step)


class EpisodeRecorder:

    def __init__(self, env, directory, helper):
        if not hasattr(env, 'episode_name'):
            env = EpisodeName(env)
        self._env = env
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(exist_ok=True, parents=True)
        self._episode = None
    
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self._episode = [{'image': obs}]
        return obs
    
    def step(self, action):
        # Transitions are defined from the environment perspective, meaning that a
        # transition contains the action and the resulting reward and next
        # observation produced by the environment in response to said action.
        obs, reward, done, info = self._env.step(action)
        transition = {
            'action': action, 'image': obs, 'reward': reward, 'done': done,
        }
        for key, value in info.items():
            if key in ('inventory', 'achievements'):
                continue
            transition[key] = value
        for key, value in info['achievements'].items():
            transition[f'achievement_{key}'] = value
        for key, value in info['inventory'].items():
            transition[f'ainventory_{key}'] = value  # TODO: ainventory or inventory?
        self._episode.append(transition)
        if done:
            self._save()
        return obs, reward, done, info
    
    def _save(self):
        filename = str(self._directory / (self._env.episode_name + '.npz'))
        # Fill in zeros for keys missing at the first time step.
        for key, value in self._episode[1].items():
            if key not in self._episode[0]:
                self._episode[0][key] = np.zeros_like(value)
        episode = {
            k: np.array([step[k] for step in self._episode]) for k in self._episode[0]
        }
        np.savez_compressed(filename, **episode)
    

class EpisodeName:

    def __init__(self, env):
        self._env = env
        self._timestamp = None
        self._unlocked = None
        self._length = None
    
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self._timestamp = None
        self._unlocked = None
        self._length = 0
        return obs
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._length += 1
        if done:
            self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            self._unlocked = sum(int(v >= 1) for v in info['achievements'].values())
        return obs, reward, done, info
    
    @property
    def episode_name(self):
        return f'{self._timestamp}-ach{self._unlocked}-len{self._length}'
