import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from carol.base_algorithm.pytorch_sac_pranz24.model import (
    GaussianPolicy,
    QNetwork,
    SymbolicQNetwork,
    SymbolicGaussianPolicy,
)
from carol.base_algorithm.pytorch_sac_pranz24.utils import hard_update, soft_update
from carol.domain.box import Box
from carol.domain.partitions import partition_and_sample
import numpy as np


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.args = args
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.symbolic_alpha = args.alpha

        self.log_pi = None
        self.symbolic_log_pi = None

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = args.device

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            device=self.device
        )
        self.symbolic_critic = SymbolicQNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(
            device=self.device
        )
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.symbolic_critic_optim = Adam(self.symbolic_critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(
            num_inputs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        self.symbolic_critic_target = SymbolicQNetwork(
            num_inputs, action_space.shape[0], args.hidden_size
        ).to(self.device)
        hard_update(self.critic_target, self.critic)
        hard_update(self.symbolic_critic_target, self.symbolic_critic)
        if self.policy_type == "SymGaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                if args.target_entropy is None:
                    self.target_entropy = -torch.prod(
                        torch.Tensor(action_space.shape).to(self.device)
                    ).item()
                else:
                    self.target_entropy = args.target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
                self.symbolic_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.symbolic_alpha_optim = Adam([self.symbolic_log_alpha], lr=args.lr)
            self.policy = SymbolicGaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        elif self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                if args.target_entropy is None:
                    self.target_entropy = -torch.prod(
                        torch.Tensor(action_space.shape).to(self.device)
                    ).item()
                else:
                    self.target_entropy = args.target_entropy
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], args.hidden_size, action_space
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            raise ValueError(f"Undefined policy type.")

    def select_action(self, state, batched=False, evaluate=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        else:
            state = state.to_float_tensor()
        if not batched:
            state = state.unsqueeze(0)
        state = state.to(self.device)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def update_parameters(
        self, memory, batch_size, updates, logger=None, reverse_mask=False
    ):
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
        ) = memory.sample(batch_size).astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            # policy sample, get
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            # print(f"SYMBOLIC CRITIC TARGET: {time.time() -start_update_time}")
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # print(f"NEXT Q: {time.time() - start_update_time}")
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        # print(f"GET 2 Q: {time.time() - start_update_time}")
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        # print(f"QF: {time.time() - start_update_time}")
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        # print(f"BACKWARD QFLOSS: {time.time() - start_update_time}")

        pi, self.log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * self.log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        return policy_loss
    
    def update_parameters_use_reward_bound(
        self, 
        memory, 
        batch_size, 
        updates,
        model_env=None,
        disturbance=None,
        keep_q_partition_per_dim=None,
        keep_q_sample_size=None,
        all_steps=None,
        lamb=0.5, 
        logger=None, 
        reverse_mask=False,
    ):
        # Sample a batch from memory
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            mask_batch,
        ) = memory.sample(batch_size).astuple()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        if reverse_mask:
            mask_batch = mask_batch.logical_not()

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch
            )
            # policy sample, get
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, self.log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (
            (self.alpha * self.log_pi) - min_qf_pi
        ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        info = None
        # use the updated reward representation
        if disturbance > 0.0:
            # add eps around the sampled states
            base_state = state_batch.repeat(keep_q_sample_size, 1)
            base_state_delta = torch.ones_like(base_state) * 0.0
            base_box_state_batch = Box(base_state, base_state_delta)
            
            state_delta = torch.ones_like(state_batch) * disturbance
            full_state_batch = Box(state_batch, state_delta)
            box_state_batch = partition_and_sample(full_state_batch, keep_q_partition_per_dim, keep_q_sample_size, all_steps)
            # get symbolic action
            symbolic_action, _, _, = self.policy.sample(box_state_batch)
            # keep the orginal state
            model_state = {"obs": base_box_state_batch, "propagation_indices": None}
            symbolic_next_observs, pred_rewards, pred_terminals, next_model_state = model_env.dynamics_model.sample(
                symbolic_action, 
                model_state,
                deterministic=False,
                rng=model_env._rng,
                symbolic=True,
            )
            rewards = pred_rewards
            # make the two rewards same with each other
            base_action, _, _, = self.policy.sample(base_state)
            base_model_state = {"obs": base_state, "propagation_indices": None}
            base_next_obs, base_pred_rewards, _, _ = model_env.dynamics_model.sample(
                base_action,
                base_model_state,
                rng=model_env._rng,
            )
            reward_batch_repeated = base_pred_rewards # use the updated reward rather than the one in the replay buffer (from an old policy & model env)
            keep_reward_loss = torch.abs((reward_batch_repeated.detach() - rewards.inf())).mean()
            info = {
                "state_delta_min": box_state_batch.delta.min().item(),
                "state_delta_max": box_state_batch.delta.max().item(),
                "action_delta_min": symbolic_action.delta.min().item(),
                "action_delta_max": symbolic_action.delta.max().item(),
                "next_state_delta_min": symbolic_next_observs.delta.min().item(),
                "next_state_delta_max": symbolic_next_observs.delta.max().item(),
                "reward_min": (rewards.inf()).min().item(),
                "reward_max": (rewards.sup()).max().item(),
                "constraint_loss": keep_reward_loss.item()}
        else:
            keep_reward_loss = policy_loss

        if disturbance > 0.0:
            combined_policy_loss = policy_loss + lamb * keep_reward_loss
        else:
            combined_policy_loss = policy_loss

        return combined_policy_loss, info

    def hard_update_symbolic_critic(self):
        hard_update(self.symbolic_critic, self.critic)
        
    def hard_update_symbolic(self):
        hard_update(self.symbolic_critic_target, self.critic_target)
        hard_update(self.symbolic_critic, self.critic)
        if self.automatic_entropy_tuning is True:
            self.symbolic_log_alpha = self.log_alpha
    
    def update_parameters_with_alpha_loss(self):
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (self.log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            # alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs
        # alpha_loss = torch.tensor(0.0).to(self.device)

        return alpha_loss.item()
    
    def update_parameters_with_alpha_loss_symbolic(self):
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.symbolic_log_alpha * (self.symbolic_log_pi + self.target_entropy).detach()
            ).mean()

            self.symbolic_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.symbolic_alpha_optim.step()

            self.symbolic_alpha = self.symbolic_log_alpha.exp()
            alpha_tlogs = self.symbolic_alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            alpha_tlogs = torch.tensor(self.symbolic_alpha)  # For TensorboardX logs

    def agent_soft_update(self, updates):
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
    
    def agent_soft_update_symbolic(self, updates):
        if updates % self.target_update_interval == 0:
            soft_update(self.symbolic_critic_target, self.symbolic_critic, self.tau)
    
    # Save model parameters
    def save_checkpoint(self, env_name=None, suffix="", ckpt_path=None):
        if ckpt_path is None:
            assert env_name is not None
            if not os.path.exists("checkpoints/"):
                os.makedirs("checkpoints/")
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print("Saving models to {}".format(ckpt_path))
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
                "symbolic_critic_state_dict": self.symbolic_critic.state_dict(),
                "symbolic_critic_target_state_dict": self.symbolic_critic_target.state_dict(),
                "symbolic_critic_optimizer_state_dict": self.symbolic_critic_optim.state_dict(),
            },
            ckpt_path,
        )

    # Load model parameters
    def load_checkpoint(self, work_dir, ckpt_path=None, evaluate=False):
        # select the model with the largest step size
        import os
        print(os.getcwd())
        # work_dir = '../../../../' + work_dir # TODO: set work_dir as global path, but this is for code submission
        if ckpt_path is None:
            step = max([int(f[:f.index('.')]) if f[f.index('.')-1] == '9' or f[f.index('.')-1] == '0' else -1 for f in os.listdir(work_dir)])
            ckpt_path = os.path.join(work_dir, f"{step}.pth")
        else:
            data = ckpt_path.split("/")
            step = int(data[-1].split(".")[0])
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
            self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])
            self.symbolic_critic.load_state_dict(checkpoint["symbolic_critic_state_dict"])
            self.symbolic_critic_target.load_state_dict(checkpoint["symbolic_critic_target_state_dict"])
            self.symbolic_critic_optim.load_state_dict(checkpoint["symbolic_critic_optimizer_state_dict"])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
                self.symbolic_critic.eval()
                self.symbolic_critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
                self.symbolic_critic.train()
                self.symbolic_critic_target.train()
        return step
    
