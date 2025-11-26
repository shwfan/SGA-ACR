import os
import framework
import torch

import crafter
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_ada_policy
from stable_baselines3.common.base_class import BaseAlgorithm

class RLTaskCRF:
    def __init__(self, helper: framework.helpers.TrainingHelper, checkpoint_path=None):
        self.helper = helper
        self.env_train = self.create_train_env()
        # self.env_valid_deterministic = self.create_valid_env(env_prefix='valid_det', log_every_n_episodes=1)
        # self.env_valid_stochastic = self.create_valid_env(env_prefix='valid_sto', log_every_n_episodes=1)
        self.env_valid_deterministic = None
        self.env_valid_stochastic = None # just eval deterministic env
        self.checkpoint_path = checkpoint_path
        self.save_freq = self.helper.args.save_freq
        self.save_path = os.path.join(self.helper.dirs.base, "checkpoint")

        self.model = self.create_model()
        framework.helpers.model_info.print_model_size(self.model.policy)

    
    def create_train_env(self):
        return self.create_single_env(env_prefix='train', el_vars=self.helper.args.el_vars, el_freq=self.helper.args.el_freq_train, el_app_freq=self.helper.args.el_app_freq_train)
    
    def create_valid_env(self, env_prefix, save_video=False, log_every_n_episodes=None):
        return self.create_single_env(env_prefix=env_prefix, seed=0, save_video=save_video, log_every_n_episodes=log_every_n_episodes, 
                                        el_vars=self.helper.args.el_vars, el_freq=self.helper.args.el_freq_valid, el_app_freq=self.helper.args.el_app_freq_valid)

    def create_single_env(self, env_prefix='train', seed=None, save_video=False, log_every_n_episodes=None, el_vars='', el_freq='100,0,0,0', el_app_freq='sssss'):
        env = crafter.Env(
            size=(self.helper.args.crf.size, self.helper.args.crf.size), 
            render_scoreboard=self.helper.args.crf.render_scoreboard,
            seed=seed,
            length=self.helper.args.crf.max_ep_len,
            el_vars=el_vars,
            el_freq=el_freq,
            el_app_freq=el_app_freq,
        )
        env = crafter.Recorder(
            env,
            os.path.join(self.helper.save_dir, f'crafter-episodes-{env_prefix}'),
            self.helper,
            env_prefix=env_prefix,
            save_stats=True,
            save_video=save_video,
            save_episode=False,
            log_every_n_episodes=log_every_n_episodes,
        )
        env = BaseAlgorithm._wrap_env(env)
        env = VecFrameStack(env, n_stack=1, n_skip=1)
        return env

    def create_model(self):
        if self.helper.args.ppo.recurrent:
            model = sb3.RecurrentPPO(
                self.helper, 
                self.env_train, 
                verbose=1
            )
        else:
            model = sb3.PPO(
                self.helper, 
                self.env_train, 
                verbose=1
            )
            if self.checkpoint_path is not None:
                checkpoint = torch.load(self.checkpoint_path)
                model.policy.load_state_dict(checkpoint['model_state_dict'])
        return model

    def train(self):
        try:
            print('-' * 89)
            print(f'Starting training for max {self.helper.args.max_train_steps} steps')
            print("At any point you can hit Ctrl + C to break out of the training loop early.")

            self.model.learn(
                total_timesteps=self.helper.args.max_train_steps,
                eval_env_det=self.env_valid_deterministic,
                eval_env_sto=self.env_valid_stochastic,
                eval_freq=self.helper.args.eval_n_steps,
                n_eval_episodes=self.helper.args.eval_n_episodes,
                save_freq=self.save_freq,
                save_path=self.save_path
            )

        except KeyboardInterrupt:
            print('-' * 89)
            print('KeyboardInterrupt signal received. Exiting early from training.')
    
    def test(self):
        self.env_test_deterministic = self.create_valid_env(env_prefix='test_det', save_video=True, log_every_n_episodes=1)
        # self.env_test_stochastic = self.create_valid_env(env_prefix='test_sto', save_video=True, log_every_n_episodes=1)
        kb = self.model.kb
        llms_core = self.model.llms_core
        llms_core.mode = 'Test'

        # Evaluate deterministic policy
        mean_reward, std_reward, mean_length, std_length, mean_sim, std_sim = evaluate_ada_policy(llms_core,kb, self.model, self.env_test_deterministic, n_eval_episodes=20, deterministic=True)
        
        print(f"Mean deterministic reward = {mean_reward} +/- {std_reward}")
        helper_logs = {
            'eval_final/reward_det_mean': mean_reward,
            'eval_final/reward_det_std': std_reward,
            'eval_final/length_det_mean': mean_length,
            'eval_final/length_det_std': std_length,
            'eval_final/sim_det_mean': mean_sim,
            'eval_final/sim_det_std': std_sim,
        }
        self.helper.log(helper_logs, step=self.helper.state.step)

        # Evaluate stochastic policy
        # reward_mean, reward_std, _, _ = evaluate_policy(
        #         self.client,
        #         self.llm_model,
        #         self.tokenizer,
        #         self.text_enc,
        #         self.query_interval,
        #         self.model, self.env_test_stochastic, n_eval_episodes=20, deterministic=False)
        # print(f"Mean stochastic reward = {reward_mean} +/- {reward_std}")
        # helper_logs = {
        #     'eval_final/reward_stoch_mean': reward_mean,
        #     'eval_final/reward_stoch_std': reward_std,
        # }
        # self.helper.log(helper_logs, step=self.helper.state.step)
