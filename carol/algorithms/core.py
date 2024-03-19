import os
from typing import Optional, Sequence, cast
import time
from xmlrpc.client import Boolean

import hydra.utils
import numpy as np
import torch
import itertools
from cox.store import Store

import carol.models
import carol.util
import carol.util.common
from carol.util import math
import carol.planning
import carol.base_algorithm.pytorch_sac_pranz24 as pytorch_sac_pranz24
from carol.planning.sac_wrapper import SACAgent

from carol.util.common import (
    get_last_logs,
)
from carol.util.scheduler import (
    ConstantScheduler,
    LinearScheduler,
    ExpScheduler,
    SmoothScheduler,
    RefSmoothedScheduler,
)
from carol.domain.box import Box

class CAROLalgorithm(object):
    def __init__(self, env, test_env, extra_env,\
        termination_fn, cfg, work_dir=None):
        self.env = env
        self.test_env = test_env
        self.extra_env = extra_env
        self.termination_fn = termination_fn
        self.cfg = cfg
        self.lamb = cfg.overrides.lamb
        self.work_dir = work_dir or os.getcwd()
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        self.resume = self.cfg.resume # resume from the previous training
        self.ini_step = 0
        self.use_original_loss = self.cfg.overrides.use_original_loss # use concrete policy loss
        self.keep_q_sample_size = self.cfg.overrides.keep_q_sample_size
        self.keep_q_partition_per_dim = self.cfg.overrides.keep_q_partition_per_dim
        self.use_reward_bound = self.cfg.overrides.use_reward_bound
        
        self.use_splitter = self.cfg.overrides.use_splitter # refine the replay buffer space
        self.splitter = self.cfg.overrides.splitter
        self.partition_refined = self.cfg.overrides.partition_refined
        self.sample_scale = self.cfg.overrides.sample_scale
        self.partition_per_dim = self.cfg.overrides.partition_per_dim
        # generate the partition combination to sample from
        self.all_steps = list(itertools.product([i for i in range(self.keep_q_partition_per_dim)], repeat=self.obs_shape[0]))

        # --------------------- Parameters for robust learning ---------------------
        if self.cfg.overrides.eschedule == "constant":
            self.escheduler = ConstantScheduler(
                target_eps=self.cfg.overrides.disturbance,
                start_steps=self.cfg.overrides.eschedule_start_steps,
                end_steps=self.cfg.overrides.eschedule_end_steps
            )
        elif self.cfg.overrides.eschedule == "linear":
            self.escheduler = LinearScheduler(
                target_eps=self.cfg.overrides.disturbance,
                start_steps=self.cfg.overrides.eschedule_start_steps,
                end_steps=self.cfg.overrides.eschedule_end_steps
            )
        elif self.cfg.overrides.eschedule == "exp":
            self.escheduler = ExpScheduler(
                target_eps=self.cfg.overrides.disturbance,
                start_steps=self.cfg.overrides.eschedule_start_steps,
                end_steps=self.cfg.overrides.eschedule_end_steps,
                exp_scale=self.cfg.overrides.exp_scale,
            )
        elif self.cfg.overrides.eschedule == "smooth":
            self.escheduler = SmoothScheduler(
                target_eps=self.cfg.overrides.disturbance,
                start_steps=self.cfg.overrides.eschedule_start_steps,
                end_steps=self.cfg.overrides.eschedule_end_steps,
                exp_scale=self.cfg.overrides.exp_scale,
            )
        elif self.cfg.overrides.eschedule == "ref":
            self.escheduler = RefSmoothedScheduler(
                target_eps=self.cfg.overrides.disturbance,
                start_steps=self.cfg.overrides.eschedule_start_steps,
                end_steps=self.cfg.overrides.eschedule_end_steps,
                exp_scale=self.cfg.overrides.exp_scale,
            )
        else:
            NotImplementedError(f"The Escheduler not defined.")
        self.disturbance = self.cfg.overrides.start_disturbance
        
        # --------------------- Update Agent Configuration ---------------------
        carol.planning.complete_agent_cfg(self.env, self.cfg.algorithm.agent)
        self.agent = SACAgent(
            cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(self.cfg.algorithm.agent))
        )
        
        # --------------------- Resume Training ---------------------
        self.training_log_path = os.path.join(self.work_dir, "training.txt")
        if self.resume:
            last_log_dict = get_last_logs(self.training_log_path)
            self.disturbance = last_log_dict['eps']
            self.ini_step = last_log_dict['Step']
            self.training_f = open(self.training_log_path, 'a')
            self.agent.sac_agent.load_checkpoint(self.work_dir)
        else:
            self.training_f = open(self.training_log_path, 'w')

        # --------------------- Parameters about Randomness ---------------------
        self.rng = np.random.default_rng(seed=self.cfg.seed)
        self.torch_generator = torch.Generator(device=self.cfg.device)
        if self.cfg.seed is not None:
            self.torch_generator.manual_seed(self.cfg.seed)

        # -------------- Create initial overrides. dataset --------------
        self.dynamics_model = carol.util.common.create_one_dim_tr_model(
            self.cfg, 
            self.obs_shape, 
            self.act_shape,
            model_dir=self.work_dir if self.resume else None,
            model_normalization=self.cfg.overrides.model_normalization,
            )
        self.dtype = np.double if cfg.algorithm.get("normalize_double_precision", False) else np.float32
        # replay_buffer: store the concrete trajectories for model training
        # sac_buffer: store the symbolic trajectories for sampling
        self.replay_buffer = carol.util.common.create_replay_buffer(
            self.cfg,
            self.obs_shape,
            self.act_shape,
            rng=self.rng,
            obs_type=self.dtype,
            action_type=self.dtype,
            reward_type=self.dtype,
            load_dir=self.work_dir if self.resume else None,
        )
        self.random_explore = self.cfg.algorithm.random_initial_explore
        carol.util.common.rollout_agent_trajectories(
            self.env,
            self.cfg.algorithm.initial_exploration_steps,
            carol.planning.RandomAgent(self.env) if self.random_explore else self.agent,
            {} if self.random_explore else {"sample": True, "batched": False},
            replay_buffer=self.replay_buffer,
        )
        
        # ---------------------------------------------------------
        # --------------------- Parameters for Training Loop ---------------------
        self.rollout_batch_size = (
            self.cfg.overrides.effective_model_rollouts_per_step * self.cfg.algorithm.freq_train_model
        )

        if self.cfg.overrides.freq_train_model == 0:
            self.trains_per_epoch = self.cfg.overrides.epoch_length
        else:
            self.trains_per_epoch = int(
                np.ceil(self.cfg.overrides.epoch_length / self.cfg.overrides.freq_train_model)
            )

        self.model_env = carol.models.ModelEnv(
            self.env, 
            self.dynamics_model, 
            termination_fn=self.termination_fn, 
            reward_fn=None,  
            generator=self.torch_generator
        )
        self.model_trainer = carol.models.ModelTrainer(
            self.dynamics_model,
            optim_lr=self.cfg.overrides.model_lr,
            weight_decay=self.cfg.overrides.model_wd,
        )

    def rollout_model_and_populate_sac_buffer(
        self,
        sac_buffer: carol.util.SymbolicReplayBuffer,
        rollout_horizon: int,
        disturbance: Optional[float] = None,
    ):
        sample_scale = 1
        rollout_batch_size = self.rollout_batch_size
        if disturbance is not None and disturbance > 0.0:
            if self.use_splitter:
                if self.splitter == "refined": 
                    # randomly sample from the obs space and add the refined eps
                    rollout_batch_size = int(self.rollout_batch_size/self.sample_scale)
                    if self.cfg.overrides.sample_original_rollout_batch:
                        sample_scale = self.sample_scale
                if self.splitter == "partition": 
                    # split the obs space into sub-spaces, and sample the split space 
                    # cover all the space
                    sample_scale = self.partition_per_dim ** self.obs_shape[0]
                    rollout_batch_size = int(self.rollout_batch_size/sample_scale)

        for _ in range(sample_scale):
            batch = self.replay_buffer.sample(rollout_batch_size)
            initial_obs, *_ = cast(carol.util.types.TransitionBatch, batch).astuple()
            '''
            Add symbolic coverage to the buffer
            '''
            if disturbance is not None: # if disturbance is not None, create the ball covering the initial obs
                disturbance_array = np.full_like(initial_obs, disturbance)
                initial_obs = Box(initial_obs, disturbance_array)

            model_state = self.model_env.reset(
                initial_obs_batch=cast(np.ndarray, initial_obs) if not isinstance(initial_obs, carol.domain.box.Box) else initial_obs,
                return_as_np=True,
            )
            accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
            obs = initial_obs
            for i in range(rollout_horizon): # rollout_horizon >=0
                if disturbance is not None:
                    # split obs into disturbance range ball
                    if self.use_splitter:
                        # This is the default parameter setting: 'original'
                        if self.splitter == 'original' or disturbance == 0.0:
                            # Implementation I: 
                            # Randomly sample from the obs space as center, use the given disturbance as the delta
                            random_base = np.random.rand(*obs.shape)
                            l, r = obs.c - obs.delta, obs.c + obs.delta
                            obs_center = obs.inf() + obs.width()*random_base
                            disturbance_array = np.full_like(obs.delta, disturbance)
                            obs = Box(obs_center, disturbance_array)
                            # clamp the refined obs by original obs box
                            obs = obs.clamp(min=l, max=r)
                        elif self.splitter == 'keep_center':
                            # Implementation IV:
                            # keep the center and add the disturbance
                            l, r = obs.c - obs.delta, obs.c + obs.delta
                            refined_disturbance = disturbance / self.partition_refined
                            obs_center = obs.c # no randomness, just use the center
                            disturbance_array = np.full_like(obs.delta, refined_disturbance)
                            obs = Box(obs_center, disturbance_array)
                            obs = obs.clamp(min=l, max=r)
                        elif self.splitter == 'refined':
                            # Implementation II: 
                            # Randomly sample from the obs space as center, use the refined version of the given disturbance as the delta
                            # And sample more of these
                            l, r = obs.c - obs.delta, obs.c + obs.delta
                            l_total, r_total = None, None
                            obs_center, disturbance_array = None, None
                            refined_disturbance = disturbance/self.partition_refined
                            for _ in range(self.sample_scale):
                                random_base = np.random.rand(*obs.shape)
                                if obs_center is None:
                                    obs_center = obs.inf() + obs.width()*random_base
                                    disturbance_array = np.full_like(obs.delta, refined_disturbance)
                                    l_total, r_total = l, r
                                else:
                                    obs_center = np.concatenate((obs_center, obs.inf() + obs.width()*random_base), 0, dtype=float)
                                    disturbance_array = np.concatenate((disturbance_array, np.full_like(obs.delta, refined_disturbance, dtype=float)), 0)
                                    l_total = np.concatenate((l_total, l), 0)
                                    r_total = np.concatenate((r_total, r), 0)
                            obs = Box(obs_center, disturbance_array)
                            # clamp the refined obs by original obs box
                            obs = obs.clamp(min=l_total, max=r_total)
                        elif self.splitter == "partition":
                            # Implementation III:
                            # partition the obs space based on partition_per_dim
                            # cover all the partitions
                            full_box = Box(obs.c, np.full_like())
                            num_disjunctions = self.partition_per_dim
                            refined_disturbance = disturbance/num_disjunctions
                            data_center_bias_list = [(-num_disjunctions+1+2*i)*refined_disturbance for i in range(num_disjunctions)]
                            pass
                        elif self.splitter == "center":
                             pass
                        else:
                            NotImplementedError(f"Splitter Not Implented.")
                    # clamp the obs based on obs space
                    obs = obs.clamp(min=self.model_env.observation_space.low, max=self.model_env.observation_space.high)
                    model_state["obs"] = obs.astype(np.float32)
                    # update the accum_dones index
                    accum_dones = np.zeros(obs.shape[0], dtype=bool)
                
                action = self.agent.act(obs, sample=self.cfg.algorithm.sac_samples_action, batched=True)
                pred_next_obs, pred_rewards, pred_dones, model_state = self.model_env.step(
                    action, model_state, sample=True
                )
                sac_buffer.add_batch(
                    obs[~accum_dones],
                    action[~accum_dones],
                    pred_next_obs[~accum_dones],
                    pred_rewards[~accum_dones, 0],
                    pred_dones[~accum_dones, 0],
                )
                obs = pred_next_obs
                accum_dones |= pred_dones.squeeze()

    def maybe_replace_sac_buffer(
        self,
        sac_buffer: Optional[carol.util.SymbolicReplayBuffer],
        new_capacity: int,
        symbolic: bool,
    ) -> carol.util.ReplayBuffer:
        if sac_buffer is None or new_capacity != sac_buffer.capacity:
            if sac_buffer is None:
                rng = np.random.default_rng(seed=self.cfg.seed)
            else:
                rng = sac_buffer.rng
            if symbolic:
                new_buffer = carol.util.SymbolicReplayBuffer(
                    new_capacity, 
                    self.obs_shape, 
                    self.act_shape, 
                    rng=rng
                )
            else:
                new_buffer = carol.util.ReplayBuffer(
                    new_capacity, 
                    self.obs_shape, 
                    self.act_shape, 
                    rng=rng
                )
            if sac_buffer is None:
                return new_buffer
            obs, action, next_obs, reward, done = sac_buffer.get_all().astuple()
            new_buffer.add_batch(obs, action, next_obs, reward, done)
            return new_buffer
        return sac_buffer

    def reward_evaluate(
        self,
    ) -> float:
        avg_episode_reward = 0
        with torch.no_grad():
            for episode in range(self.cfg.algorithm.num_eval_episodes):
                obs = self.test_env.reset()
                done = False
                episode_reward = 0
                steps = 0
                while not done:
                    action = self.agent.act(obs)
                    obs, reward, done, _ = self.test_env.step(action)
                    episode_reward += reward
                    steps += 1
                avg_episode_reward += episode_reward
        
        return avg_episode_reward / self.cfg.algorithm.num_eval_episodes, None

    def reward_evaluate_model(
        self,
        ):
        avg_episode_reward = 0
        
        with torch.no_grad():
            for episode in range(self.cfg.algorithm.num_eval_episodes):
                obs = self.test_env.reset()
                model_state = self.model_env.reset(obs[None, :])
                done = False
                episode_reward = 0
                steps = 0
                while not done:
                    try:
                        action = self.agent.act(obs)
                    except:
                        obs = self.test_env.reset()
                        model_state = self.model_env.reset(obs[None, :])
                        action = self.agent.act(obs)
                    # If obs/model_state is out of scope, reset
                    # In the cases where the model is not well-learnt, the output model_state/output might be out of scope.
                    try:
                        obs, reward, done, model_state = self.model_env.step(action, model_state)
                    except ValueError:
                        obs = self.test_env.reset()
                        model_state = self.model_env.reset(obs[None, :])
                    episode_reward += reward.item()
                    obs = obs[0]
                    done = done.item()
                    steps += 1
                    if steps >= self.cfg.overrides.trial_length:
                        done = True
                avg_episode_reward += episode_reward
        return avg_episode_reward / self.cfg.algorithm.num_eval_episodes, None

    def train(self): # mbpo for L_{normal}(\theta) + \lambda * L_{symbolic}(\theta)
        updates_made = 0
        env_steps = self.ini_step if self.ini_step > 0 else 0
        epoch = 0
        if self.use_original_loss:
            sac_buffer = None

        # initialize best metrics
        best_eval_reward = -np.inf
        
        print(f"{'-'*10}Training Starts{'-'*10}")
        training_start_time = time.time()
        ori_policy_loss, policy_loss = None, None

        while env_steps < self.cfg.overrides.num_steps:
            rollout_length = int(
                carol.util.math.truncated_linear(
                    *(self.cfg.overrides.rollout_schedule + [epoch + 1])
                )
            )
            sac_buffer_capacity = rollout_length * self.rollout_batch_size * self.trains_per_epoch
            sac_buffer_capacity *= self.cfg.overrides.num_epochs_to_retain_sac_buffer
            if self.use_original_loss:
                sac_buffer = self.maybe_replace_sac_buffer(
                    sac_buffer, 
                    sac_buffer_capacity,
                    symbolic=False,
                )
            
            obs, done = None, False
            start_epoch_time = time.time()
            for steps_epoch in range(self.cfg.overrides.epoch_length):
                if steps_epoch == 0 or done:
                    obs, done = self.env.reset(), False
                # --- Doing env step and adding to model dataset ---
                next_obs, reward, done, _ = carol.util.common.step_env_and_add_to_buffer(
                    self.env, obs, self.agent, {}, self.replay_buffer
                )
                # --------------- Dynamics Model Training -----------------
                if self.cfg.overrides.freq_train_model == 0:
                    pass
                elif (env_steps + 1) % self.cfg.overrides.freq_train_model == 0: 
                # train the environment model
                    self.dynamics_model.model.zero_grad()
                    carol.util.common.train_model_and_save_model_and_data(
                        self.dynamics_model,
                        self.model_trainer,
                        self.cfg.overrides,
                        self.replay_buffer, # replay_buffer store the no-disturbance state-action pairs
                        work_dir=self.work_dir, # store the model and the replay buffer
                    )
                    # print(f"-------- Rollout model and populate sac buffer {time.time() - start_epoch_time}s -------")
                    if self.use_original_loss:
                        self.rollout_model_and_populate_sac_buffer(
                            sac_buffer,
                            rollout_length,
                        )

                # print(f"-----Start Agent Training Time: {time.time() - start_epoch_time}")
                # torch.autograd.set_detect_anomaly(True)
                for _ in range(self.cfg.overrides.num_sac_updates_per_step): # 20 for cartpole
                    use_real_data = self.rng.random() < self.cfg.algorithm.real_data_ratio
                    if self.use_original_loss:
                        which_buffer = self.replay_buffer if use_real_data else sac_buffer 
                        if (env_steps + 1) % self.cfg.overrides.sac_updates_every_steps != 0 or len(
                            which_buffer
                        ) < self.cfg.overrides.sac_batch_size:
                            break  # only update every once in a while or when the buffer size > sac batch size
                    
                    indeed_use_symbolic_loss = False
                    info = None
                    if self.use_original_loss:
                        if self.use_reward_bound:
                            ori_policy_loss, info = self.agent.sac_agent.update_parameters_use_reward_bound(
                                which_buffer,
                                self.cfg.overrides.sac_batch_size,
                                updates_made,
                                model_env=self.model_env,
                                disturbance=self.disturbance,
                                keep_q_partition_per_dim=self.keep_q_partition_per_dim,
                                keep_q_sample_size=self.keep_q_sample_size,
                                all_steps=self.all_steps,
                                lamb=self.lamb,
                                reverse_mask=True,
                            )
                        else:
                            ori_policy_loss = self.agent.sac_agent.update_parameters(
                                which_buffer,
                                self.cfg.overrides.sac_batch_size,
                                updates_made,
                                reverse_mask=True,
                            )
                    self.agent.sac_agent.policy_optim.zero_grad()
                    ori_policy_loss.backward()
                    report_loss = ori_policy_loss.item()
                    
                    self.agent.sac_agent.policy_optim.step()

                    if self.use_original_loss:
                        self.agent.sac_agent.update_parameters_with_alpha_loss()
                        self.agent.sac_agent.agent_soft_update(updates=updates_made)

                    updates_made += 1

                # ------ Epoch ended (evaluate and save model) ------
                if (env_steps + 1) % self.cfg.overrides.epoch_length == 0:
                    avg_reward, _ = self.reward_evaluate() # reward and safety reward from sampled points through the ground truth environment
                    # print(f"--------- Test Model Training Reward and Concrete Safety --------")
                    model_avg_reward = None

                    self.agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(self.work_dir, f"{env_steps}.pth") 
                        # ckpt_path=os.path.join(self.work_dir, f"0.pth") # replace the previous pth to save space
                    )

                    epoch += 1
                    print(f"Step: {env_steps}; Loss: {report_loss}; Reward: {avg_reward}; eps: {self.disturbance};")
                    if info is not None:
                        print(f"state_delta_min: {info['state_delta_min']}; state_delta_max: {info['state_delta_max']}; action_delta_min: {info['action_delta_min']}; action_delta_max: {info['action_delta_max']}; next_state_delta_min: {info['next_state_delta_min']}; next_state_delta_max: {info['next_state_delta_max']}; reward_min: {info['reward_min']}; reward_max: {info['reward_max']}; constraint_loss: {info['constraint_loss']}")
                    self.training_f.write(f"Step: {env_steps}; Loss: {report_loss}; Reward: {avg_reward}; eps: {self.disturbance};")
                    if info is not None:
                        self.training_f.write(f" state_delta_min: {info['state_delta_min']}; state_delta_max: {info['state_delta_max']}; action_delta_min: {info['action_delta_min']}; action_delta_max: {info['action_delta_max']}; next_state_delta_min: {info['next_state_delta_min']}; next_state_delta_max: {info['next_state_delta_max']}; reward_min: {info['reward_min']}; reward_max: {info['reward_max']}; constraint_loss: {info['constraint_loss']}")
                    self.training_f.write("\n")
                    self.training_f.flush()

                    # Schedule the disturbance
                    self.disturbance = self.escheduler.step_eps(env_steps, self.disturbance)
                            
                env_steps += 1
                obs = next_obs
            
            print(f"Epoch: {epoch} Step: {env_steps} Time: {time.time() - start_epoch_time}")
        
        return np.float32(best_eval_reward) 
                


        






