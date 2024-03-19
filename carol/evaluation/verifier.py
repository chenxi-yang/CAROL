from typing import cast
import hydra
import itertools

import torch

import carol
import carol.util
import carol.util.common
import carol.util.env
from carol.domain.box import Box
from carol.planning.sac_wrapper import SACAgent
import carol.base_algorithm.pytorch_sac_pranz24 as pytorch_sac_pranz24

from carol.evaluation.utils import CtsPolicy

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from cox.store import Store

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class tmpModelCAROLFormat(nn.Module):
    def __init__(self, model_env, agent, horizon, device):
        super().__init__()
        self.horizon = horizon
        self.model_env_hidden_layers = model_env.dynamics_model.model.hidden_layers.to(device)
        # Transfer EnsembleLinearLayer to LinearLayer (As LiRPA does not support EnsembleLinearLayer)
        for component in self.model_env_hidden_layers:
            component[0].weight.data = torch.squeeze(component[0].weight.data) # as the ensemble keeps inXout and linear keeps outXin
            component[0].bias.data = torch.squeeze(component[0].bias.data)
        
        self.model_env_mean = model_env.dynamics_model.model.mean.to(device)
        self.model_env_mean.weight.data = torch.squeeze(self.model_env_mean.weight.data)
        self.model_env_mean.bias.data = torch.squeeze(self.model_env_mean.bias.data)

        # For non-deterministic environment modeling
        ###
        self.model_env_logvar = torch.squeeze(model_env.dynamics_model.model.logvar).to(device).detach()
        self.model_env_max_logvar = torch.squeeze(model_env.dynamics_model.model.max_logvar).to(device).detach()
        self.model_env_min_logvar = torch.squeeze(model_env.dynamics_model.model.min_logvar).to(device).detach()
        ###

        self.model_env_input_normalizer = model_env.dynamics_model.input_normalizer
        if self.model_env_input_normalizer:
            self.model_env_input_normalizer.mean = torch.squeeze(self.model_env_input_normalizer.mean).to(device).float()
            self.model_env_input_normalizer.std = torch.squeeze(self.model_env_input_normalizer.std).to(device).float()
            self.model_env_input_normalizer.device = device
        self.model_env_target_is_delta = model_env.dynamics_model.target_is_delta

        self.agent_action_scale = agent.action_scale.to(device)
        self.agent_action_bias = agent.action_bias.to(device)
        self.agent_linear1 = agent.linear1.to(device)
        self.agent_linear2 = agent.linear2.to(device)
        self.agent_mean_linear = agent.mean_linear.to(device)

        self.relu = nn.ReLU()

        std_ones = torch.ones((model_env.observation_space.shape[0] + 1)).to(device)
        mean_zeros = torch.zeros((model_env.observation_space.shape[0] + 1)).to(device)

        torch.manual_seed(0)
        self.horizon_eps_list = []
        for i in range(self.horizon):
            eps = torch.normal(mean_zeros, std_ones)
            self.horizon_eps_list.append(eps)
    
    def forward(self, x, attack, model_error): 
        episode_reward = None
        origin_x = x
        x = x + attack
        i = 0
        while i < self.horizon:
            i += 1
            x = F.relu(self.agent_linear1(x))
            x = F.relu(self.agent_linear2(x))
            mean = self.agent_mean_linear(x)
            action = torch.tanh(mean) * self.agent_action_scale + self.agent_action_bias
            obs = origin_x
            model_in = torch.cat([obs, action], dim=obs.ndim - 1)
            if self.model_env_input_normalizer:
                model_in = (model_in - self.model_env_input_normalizer.mean) / self.model_env_input_normalizer.std
            model_output = self.model_env_hidden_layers(model_in)
            # deterministic model
            # preds = self.model_env_mean(model_output)

            # stochastic model
            preds_mean = self.model_env_mean(model_output)
            preds_logvar = self.model_env_logvar
            preds_logvar = self.model_env_max_logvar - F.softplus(self.model_env_max_logvar - preds_logvar)
            preds_logvar = self.model_env_min_logvar + F.softplus(preds_logvar - self.model_env_min_logvar)
            preds_vars = preds_logvar.exp()
            preds_stds = torch.sqrt(preds_vars)
            eps = self.horizon_eps_list[i-1]
            # default: \mu + N(0, 1) * \sigma
            preds = preds_mean.add(eps * preds_stds)

            # preds = preds[0] # when using Linear Layer, no need to select the first member
            next_obs = preds[:, :-1]
            rewards = preds[:, -1:]
            if self.model_env_target_is_delta:
                tmp_ = next_obs.add(obs)
                next_obs = tmp_
            origin_x = next_obs
            x = origin_x + attack
            # for model error
            x = x + model_error

            if episode_reward is None:
                episode_reward = rewards
            else:
                episode_reward += rewards
        return episode_reward


class Verifier(object):
    def __init__(self, cfg=None, horizon_steps=None):
        self.env, self.term_fn, _ = carol.util.env.EnvHandler.make_env(cfg)
        self.test_env, *_ = carol.util.env.EnvHandler.make_env(cfg)
        self.obs_shape = self.env.observation_space.shape
        self.act_shape = self.env.action_space.shape

        self.model_name = cfg.evaluation.model_name

        self.horizon_steps = horizon_steps
        self.eps = float(cfg.evaluation.eps)
        self.cfg = cfg
        self.torch_generator = torch.Generator(device=self.cfg.device)
        
    def load_agent(self, path):
        # assume all the loaded agent for now are from CRRL
        # If not, should convert them into the CRRL model version (for symbolic computation)
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
    
    def load_model_env(self, path):
        dynamics_model = carol.util.common.create_one_dim_tr_model(
            self.cfg, 
            self.env.observation_space.shape, 
            self.env.action_space.shape,
            model_dir=path,
            model_normalization=self.cfg.overrides.model_normalization,
        )
        model_env = carol.models.ModelEnv(
            self.env,
            dynamics_model,
            termination_fn=self.term_fn,
            reward_fn=None,
            generator=self.torch_generator,
        )
        return model_env

    # with auto_LiRPA
    def extract_symbolic_reward(self, policy_dir, model_dir, model_error, normalization=False):
        agent = self.load_agent(policy_dir)
        model_env = self.load_model_env(model_dir)
        if not normalization:
            model_env.dynamics_model.input_normalizer = None

        verified_model = tmpModelCAROLFormat(
            agent=agent.sac_agent.policy, 
            model_env=model_env, 
            horizon=self.horizon_steps, 
            device="cpu",
            ).cpu()
        verified_model.eval()

        print(f"{'-'*10}Start Verification{'-'*10}")
        with torch.no_grad():
            episode_count = 0
            min_reward = 0.0
            min_reward_list = []
            for episode in range(self.cfg.evaluation.concrete_num_eval_episodes):
                obs = self.test_env.reset()
                state_batch = torch.FloatTensor(obs) 

                # Verification Algorithm
                # ptb = zero_like(state) + eps
                # Loop
                # # s, s# = state, state + ptb
                # # a# = agent(s#)
                # # s'# = env(s, a#)
                # # state = s'#

                my_input_state = torch.unsqueeze(state_batch, 0)
                my_input_attack = torch.zeros_like(my_input_state)
                my_input_model_error = torch.zeros_like(my_input_state)
                model = BoundedModule(verified_model, (my_input_state, my_input_attack, my_input_model_error), 
                    bound_opts={
                        'conv_mode': 'matrix',
                        'sparse_features_alpha': False,
                        }
                    )
                ptb_1 = PerturbationLpNorm(norm=np.inf, eps=self.eps)
                ptb_0 = PerturbationLpNorm(norm=np.inf, eps=0.0)
                ptb_2 = PerturbationLpNorm(norm=np.inf, eps=model_error)
                my_input_state = BoundedTensor(my_input_state, ptb_0)
                my_input_attack = BoundedTensor(my_input_attack, ptb_1)
                my_input_model_error = BoundedTensor(my_input_model_error, ptb_2)
                prediction = model(my_input_state, my_input_attack, my_input_model_error)
                lb, _ = model.compute_bounds(x=(my_input_state, my_input_attack, my_input_model_error,), method="backward")
                print(f"one episode worst reward/T: {lb/self.horizon_steps}")
                print(lb/self.horizon_steps)
                min_reward += lb.item()
                min_reward_list.append(lb.item())

                episode_count += 1

        print(f"{'-'*10}Verifier of {self.cfg.overrides.env}{'-'*10}")
        print(f"Disturbance: {self.eps}")
        print(f"Min Reward: {sum(min_reward_list) / ((episode_count) * self.horizon_steps)}")
        print(f"{'-'*10}Verifier Ends{'-'*10}")
        return sum(min_reward_list) / ((episode_count) * self.horizon_steps)
    