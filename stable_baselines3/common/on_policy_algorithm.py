import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from collections import deque
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from models.llms_core import LLMs_Core
from utils import QUERY_INTERVAL, get_fov_goal, RuleBasedSubgoalChecker, WeightedSubgoalGraph, KnowledgeBase


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param policy_base: The base policy used by this method
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        helper,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        llm_path=None,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            helper,
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.sim_buffer = deque(maxlen=50)

        if _init_setup_model:
            self._setup_model()
        
        self.kb = KnowledgeBase("./crafter_kb")
        self.llms_core = LLMs_Core(data_path=self.helper.dirs.base, kb=self.kb)
        self.sg_checker = RuleBasedSubgoalChecker(self.kb.subgoal_set)
        self.sg_graph = WeightedSubgoalGraph(graph_path="./crafter_kb/logs/subgoal_graph.txt", data_path=self.helper.dirs.base)
        self._graph_text_cache = self.sg_graph.render_text()
        self._episode_counter = 0


    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.helper.args,
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        env_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        obs = env.reset()
        info = {
            'inventory': env.envs[0].env._env._env._player.inventory.copy(),
            'semantic': env.envs[0].env._env._env._sem_view(),
            'player_pos': env.envs[0].env._env._env._player.pos,
            'time_step': env.envs[0].env._env._env._step,
            'achievements': env.envs[0].env._env._env._player.achievements.copy()
        }
        self._last_obs = obs

        self.sg_checker.clear_state()
        subgoal_text_set = self.kb.subgoal_text_set
        text_obs, _, entity_list = get_fov_goal(info)
        entity_text = self.kb.entity_rag(entity_list)
        unachieved = [name for name, achieved in info['achievements'].items() if achieved == 0]
        unachieved_text = ",".join(unachieved)

        graph_text = self._graph_text_cache


        plan_last = ''
        fulfilled = []
        inst, plan_sg_list = self.llms_core.ac_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, fulfilled)
        self.sg_checker.reset_plan(plan_sg_list, info)
        self.sg_graph.stage_plan(plan_sg_list)

        inst_last = inst
        text_obs_last = text_obs
        r_flag = False
        new_inst = False


        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor, instruction=inst_last, text_obs=text_obs_last)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            env_steps += 1
            self.num_timesteps += env.num_envs

            # Rule-based subgoal checker
            r_in, text_obs, entity_list = self.sg_checker.step(infos[0])
            completed_now = self.sg_checker.pop_just_completed()
            if completed_now:
                for sg_done in completed_now:
                    self.sg_graph.stage_success(sg_done, infos[0]["inventory"])
            if r_in > 0:
                r_flag = True
            if len(self.sg_checker.done) >= 3 or self.sg_checker.plan_steps >= QUERY_INTERVAL:
                fulfilled = list(self.sg_checker.done)
                entity_text = self.kb.entity_rag(entity_list)
                unachieved = [name for name, achieved in infos[0]['achievements'].items() if achieved == 0]
                unachieved_text = ",".join(unachieved)


                graph_text = self._graph_text_cache

                inst, plan_sg_list = self.llms_core.ac_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, inst_last, fulfilled)
                self.sg_checker.reset_plan(plan_sg_list, infos[0])
                self.sg_graph.stage_plan(plan_sg_list)
                score = len(fulfilled) / 3.0
                print(f"Past sub-goals: {inst_last}; Fulfilled: {fulfilled}; Score: {score}; New sub-goals: {inst}")
                self.sim_buffer.append(score)
                new_inst = True
              

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs, instruction=inst_last, text_obs=text_obs_last)[0]
                    rewards[idx] += self.gamma * terminal_value

                if done:
                    self._episode_counter += 1

                    if self._episode_counter % 10 == 0:
                        self.sg_graph.apply_staged()
                        self._graph_text_cache = self.sg_graph.render_text()

                    # if self._episode_counter % 500 == 0:
                    #     self.sg_graph.dump_jsonl(self._episode_counter)
                    #     print(f"[SubgoalGraph] Successfully save weights snapshot!")
                
                    info = {
                        'inventory': env.envs[0].env._env._env._player.inventory.copy(),
                        'semantic': env.envs[0].env._env._env._sem_view(),
                        'player_pos': env.envs[0].env._env._env._player.pos,
                        'time_step': env.envs[0].env._env._env._step,
                        'achievements': env.envs[0].env._env._env._player.achievements.copy()
                    }

                    self.sg_checker.clear_state()
                    text_obs, _, entity_list = get_fov_goal(info)
                    entity_text = self.kb.entity_rag(entity_list)
                    unachieved = [name for name, achieved in info['achievements'].items() if achieved == 0]
                    unachieved_text = ",".join(unachieved)
                    graph_text = self._graph_text_cache

                    plan_last = ''
                    fulfilled = []
     
                    inst, plan_sg_list = self.llms_core.ac_generate(text_obs, graph_text, entity_text, unachieved_text, subgoal_text_set, plan_last, fulfilled)
                    self.sg_checker.reset_plan(plan_sg_list, info)
                    self.sg_graph.stage_plan(plan_sg_list)
                    new_inst = True

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, inst_last, text_obs_last)
            if r_flag:
                rollout_buffer.rewards[-1, 0] += r_in
                r_flag = False

            if new_inst:
                inst_last = inst
                new_inst = False
            self._last_obs = new_obs
            self._last_episode_starts = dones
            text_obs_last = text_obs

        with th.no_grad():
            # Compute value for the last timestep
            if self.helper.args.use_pure_ppo is not True:
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), instruction=inst, text_obs=text_obs)
            elif self.helper.args.use_pure_ppo:
                values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), instruction='')

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env_det: Optional[GymEnv] = None,
        eval_env_sto: Optional[GymEnv] = None,
        eval_freq: int = -1,
        save_freq: int = -1,
        save_path: str = None,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env_det, eval_env_sto, callback, eval_freq, save_freq, save_path,n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                # Display training infos (native stable-baselines3)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_ret_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.record("rollout/sim_score_mean", np.mean(self.sim_buffer))
                self.logger.dump(step=self.num_timesteps)
            
                # Display training infos (mine via helper)
                self.helper.state.step = int(self.num_timesteps)
                helper_logs = {
                    "step": self.helper.state.step,
                    "time/iterations": iteration,
                    "time/fps": fps,
                    "time/time_elapsed": int(time.time() - self.start_time),
                    "time/total_timesteps": self.num_timesteps,
                    "rollout/sim_score_mean": np.mean(self.sim_buffer), 
                }
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logs_add = {
                        "rollout/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                        "rollout/ep_len_mean": safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    }
                    helper_logs.update(logs_add)
                self.helper.log(helper_logs, step=self.helper.state.step)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
