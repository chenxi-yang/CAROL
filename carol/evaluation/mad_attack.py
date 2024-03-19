import sys, os
sys.path.append(f"{os.getcwd()}/../../../rllab/sandbox")
import numpy as np
import torch
from typing import cast
import pickle
import hydra

import carol
import carol.util.env
import carol.planning
from carol.planning.sac_wrapper import SACAgent
import carol.base_algorithm.pytorch_sac_pranz24 as pytorch_sac_pranz24
# PPO
from carol.base_algorithm.pytorch_ppo.ppo import PPOAttack, PPOMADAttack
from carol.base_algorithm.pytorch_ppo.network import FeedForwardNN
# RADIAL
from cox.store import Store
from carol.evaluation.utils import (
    CtsPolicy, # the policy used in RADIAL policy training
    )

class MADAttack(object):
    '''
    Applies an attack to the trained policy in the given environment.
    
    Args:
    test_env: the environment
    attack_position: list of bool, e.g. [True, False]: attack over the states, and not over the actions
    attack_idx: list of list of bool
    disturbances:
    \epsilon used for magnitude,
    max_disturbance_steps:
    '''
    def __init__(self, attack_position=None, attack_idx=None, \
        disturbance=None, max_disturbance_steps=None, cfg=None, attack_steps=50_000_0):
        self.env, *_ = carol.util.env.EnvHandler.make_env(cfg)
        self.test_env, *_ = carol.util.env.EnvHandler.make_env(cfg)
        self.external_envs = None # For RADIAL or SA's framework
        self.attack_position = attack_position
        self.attack_idx = attack_idx
        self.disturbance = disturbance
        self.max_disturbance_steps = max_disturbance_steps
        self.cfg = cfg
        self.hyperparameters = {
            'timesteps_per_batch': 2048, 
            'max_timesteps_per_episode': cfg.overrides.trial_length, 
            'gamma': 0.99, 
            'n_updates_per_iteration': 10,
            'lr': 3e-3,
            'clip': 0.2,
            'render': False,
            'render_every_i': 10
            }
        self.attack_steps = attack_steps

    def train_attacker(self, m, ori_policy_name, run_env=None, state_filter=None, reward_filter=None): # train the attacker
        # train the attacker
        # attack over the observation
        # attacker's input: state; attacker's output: noise
        # m is differentiable through the pytorch version
        # should add a wrapper over the ones not coming from PyTorch
        attacker_model = PPOMADAttack(policy_class=FeedForwardNN, 
            ori_policy=m, 
            env=self.env,
            run_env=run_env,
            disturbance=self.disturbance,
            ori_policy_name=ori_policy_name,
            log=True,
            state_filter=state_filter,
            reward_filter=reward_filter,
            **self.hyperparameters,
        )
        attacker_model.learn(total_timesteps=self.attack_steps)
        return attacker_model

    def add_disturbance(self, data):
        # sample from the surrounding of data with disturbance
        # data: np.array
        shape = data.shape
        noise = np.random.uniform(low=-1.0, high=1.0, size=shape)
        disturbed_data = data + noise * self.disturbance
        return disturbed_data
    
    def load_agent(self, path, model_name, expr_name=None):
        # load the agent, return the agent model/action function
        # return f: f(obs)
        if model_name == 'CAROL':
            configs_agent = expr_name.split('_')
            self.cfg.overrides.sac_policy = "SymGaussian"
            carol.planning.complete_agent_cfg(self.test_env, self.cfg.algorithm.agent)
            agent = SACAgent(
                cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(self.cfg.algorithm.agent))
            )
            if path[-4:] == ".pth":
                agent.sac_agent.load_checkpoint(path, ckpt_path=path)
            else:
                agent.sac_agent.load_checkpoint(path)
            policy = agent # .sac_agent
        return policy
    
    def reward_evaluate(self, path, model_name, expr_name=None):
        agent = self.load_agent(path, model_name, expr_name=expr_name)
        if self.disturbance > 0.0:
            if model_name == 'CAROL':
                attacker = self.train_attacker(agent, model_name)
            min_bound = np.array(self.env.observation_space.low.tolist())
            max_bound = np.array(self.env.observation_space.high.tolist())

        print(f"{'-'*10}Start Reward Evaluation{'-'*10}")

        avg_episode_reward = 0
        with torch.no_grad():
            episode_count = 0
            for episode in range(self.cfg.evaluation.concrete_num_eval_episodes):
                obs = self.test_env.reset()
                done = False
                episode_reward = 0
                steps = 0
                while not done:
                    if self.disturbance > 0.0:
                        noise, _ = attacker.get_action(obs)
                        obs_noise = obs + noise * self.disturbance
                        min_ = np.maximum(min_bound, obs-self.disturbance)
                        max_ = np.minimum(max_bound, obs+self.disturbance)
                        obs = np.clip(obs_noise, min_, max_)
                    try:
                        # this is from sampling with evaluate=True (no sampling)
                        action = agent.act(obs) # batched=False
                    except:
                        print(f"obs: {obs}; action: {action}")
                        break
                    obs, reward, done, info = self.test_env.step(action)
                    episode_reward += reward
                    steps += 1
                episode_count += 1
                avg_episode_reward += episode_reward
        print(f"{'-'*10}Attack on {model_name} of {self.cfg.overrides.env}{'-'*10}")
        print(f"Disturbance: {self.disturbance}")
        print(f"Reward: {avg_episode_reward / episode_count}")
        print(f"{'-'*10}Attack Ends{'-'*10}")
        return avg_episode_reward / episode_count


