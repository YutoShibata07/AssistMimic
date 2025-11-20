"""
Knowledge Distillation Agent for SimpleLiftUp with DAgger + PPO
Based on InterMimic's intermimic_agent_distill.py, adapted for 2000 iterations curriculum.

Curriculum schedule (2k iterations):
- Iterations 0-500: Expert loss only (pure imitation)
- Iterations 500-600: Expert loss + Critic loss (start value learning)
- Iterations 600-700: Monitor critic EV (explained variance)
- Iterations 700+: Add Actor loss when EV >= 0.6 for 3 consecutive checks
"""

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from isaacgym.torch_utils import *

import numpy as np
import torch
from torch import nn

import phc.learning.im_amp as im_amp


class IMAmpDistill(im_amp.IMAmpAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.expert_loss_coef = config.get('expert_loss_coef', 1.0)
        self.entropy_coef = config.get('entropy_coef', 0.0)
        self.ev_ma = 0.0  # Running average explained variance
        self.critic_win_streak = 0  # Consecutive windows where EV >= threshold
        self.actor_update_num = 0  # Number of actor updates performed

        # Load curriculum configuration
        curriculum_config = config.get('curriculum', {})
        self.dagger_beta_start = curriculum_config.get('dagger_beta_start_epoch', 200)
        self.dagger_beta_end = curriculum_config.get('dagger_beta_end_epoch', 2000)

        self.stage1_end = curriculum_config.get('stage1_end_epoch', 500)
        self.stage2_start = curriculum_config.get('stage2_start_epoch', 500)
        self.stage2_end = curriculum_config.get('stage2_end_epoch', 600)
        self.stage3_start = curriculum_config.get('stage3_start_epoch', 600)
        self.stage3_end = curriculum_config.get('stage3_end_epoch', 700)
        self.stage4_start = curriculum_config.get('stage4_start_epoch', 700)

        self.ev_threshold = curriculum_config.get('ev_threshold', 0.6)
        self.ev_win_streak_required = curriculum_config.get('ev_win_streak', 3)

        self.actor_rampup_steps = curriculum_config.get('actor_rampup_steps', 1000)
        self.expert_decay_steps = curriculum_config.get('expert_decay_steps', 1000)
        self.expert_min_weight = curriculum_config.get('expert_min_weight', 0.1)

        print("=" * 80)
        print("Knowledge Distillation Agent Initialized")
        print(f"  Expert loss coefficient: {self.expert_loss_coef}")
        print(f"  Entropy coefficient: {self.entropy_coef}")
        print(f"")
        print("  Curriculum schedule:")
        print(f"    DAgger beta decay: [{self.dagger_beta_start}, {self.dagger_beta_end})")
        print(f"    Stage 1 (Pure Expert): [0, {self.stage1_end})")
        print(f"    Stage 2 (Expert + Critic): [{self.stage2_start}, {self.stage2_end})")
        print(f"    Stage 3 (Monitor EV): [{self.stage3_start}, {self.stage3_end})")
        print(f"      - EV threshold: {self.ev_threshold}")
        print(f"      - Win streak required: {self.ev_win_streak_required}")
        print(f"    Stage 4 (Full RL): [{self.stage4_start}+)")
        print(f"      - Actor rampup steps: {self.actor_rampup_steps}")
        print(f"      - Expert decay steps: {self.expert_decay_steps}")
        print(f"      - Expert min weight: {self.expert_min_weight}")
        print("=" * 80)
        return

    def init_tensors(self):
        super().init_tensors()
        batch_shape = self.experience_buffer.obs_base_shape

        # Add expert tensors to experience buffer
        self.experience_buffer.tensor_dict['expert_mask'] = torch.zeros(
            batch_shape, dtype=torch.float32, device=self.ppo_device
        )
        self.experience_buffer.tensor_dict['expert'] = torch.zeros(
            (*batch_shape, 153), dtype=torch.float32, device=self.ppo_device
        )
        self.experience_buffer.tensor_dict['rand_action_mask'] = torch.zeros(
            (*batch_shape, 1), dtype=torch.float32, device=self.ppo_device
        )

        # Update tensor list for batch collection
        self.tensor_list += ['amp_obs', 'rand_action_mask', 'expert', 'expert_mask']
        return

    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list
        done_indices = []

        # Track termination flags and raw rewards (for logging)
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)

        # DAgger beta coefficient: probability of using expert action
        # Schedule: linearly decay from 1.0 to 0.0 between configured epochs
        beta_decay_range = self.dagger_beta_end - self.dagger_beta_start
        if beta_decay_range > 0:
            beta_t = max(1.0 - max((self.epoch_num - self.dagger_beta_start) / beta_decay_range, 0), 0)
        else:
            beta_t = 0.0  # No DAgger if decay range is invalid

        for n in range(self.horizon_length):
            # Reset environments and get expert actions
            self.obs, self.expert = self.env_reset(done_indices)

            # CRITICAL: For DAgger, we ALWAYS need expert actions for ALL envs
            # env_reset only returns expert for reset envs, so we must compute for all
            task = None
            if hasattr(self.vec_env, 'task'):
                task = self.vec_env.task
            elif hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task'):
                task = self.vec_env.env.task

            obs_tensor = self.obs['obs'] if isinstance(self.obs, dict) else self.obs
            # Always get expert actions for ALL environments (pass None for env_ids)
            expert_mus, expert_actions_tensor = task._get_expert_actions(None, obs_tensor)
            self.expert = {"mus": expert_mus, "actions": expert_actions_tensor}
           
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            # Get expert actions for DAgger
            expert_actions = None
            if self.expert is not None and 'actions' in self.expert:
                expert_actions = self.expert['actions'].to(self.ppo_device)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                # Get actions with DAgger mixing
                res_dict = self.get_action_values(
                    self.obs,
                    None,  # rand_action_probs (not used in distillation)
                    beta_t,
                    expert_actions
                )

            # Store expert actions (mu) for supervised loss
            if self.expert is not None and 'mus' in self.expert and len(self.expert['mus']) > 0:
                self.experience_buffer.update_data('expert', n, self.expert['mus'].to(self.ppo_device))
            # else:
            #     error_msg = "Expert actions not found in expert dictionary"
            #     print(error_msg)
            #     raise ValueError(error_msg)

            # Store other data
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            # Environment step with expert return
            self.obs, rewards, self.dones, infos, self.expert = self.env_step(res_dict['actions'])

            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            # Track terminations and raw rewards
            terminated = infos['terminate'].float()
            terminated_flags += terminated

            reward_raw_mean = infos['reward_raw'].mean(dim=0)
            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean

            # Compute next values for GAE
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            # Update episode tracking
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if self.vec_env.env.task.viewer:
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]

        # Compute advantages and returns
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float().to(self.ppo_device)
        mb_values = self.experience_buffer.tensor_dict['values'].to(self.ppo_device)
        mb_next_values = self.experience_buffer.tensor_dict['next_values'].to(self.ppo_device)
        mb_rewards = self.experience_buffer.tensor_dict['rewards'].to(self.ppo_device)

        # Calculate AMP rewards and combine with task rewards
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs'].to(self.ppo_device)
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_advs = mb_advs.to(self.ppo_device)
        mb_values = mb_values.to(self.ppo_device)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(
            a2c_common.swap_and_flatten01, self.tensor_list
        )
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] = reward_raw / self.horizon_length
        batch_dict['played_frames'] = self.batch_size

        # Add AMP reward components to batch_dict
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)
        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)

        return batch_dict

    def get_action_values(self, obs_dict, rand_action_probs=None, use_experts=0.0, expert=None):
        """
        Get action values with DAgger mixing.

        Args:
            obs_dict: Observation dictionary
            rand_action_probs: Random action probabilities (not used in distillation)
            use_experts: Beta coefficient for DAgger (probability of expert action)
            expert: Expert actions from teacher policy

        Returns:
            Dictionary with actions, values, etc.
        """
        res_dict = super().get_action_values(obs_dict)

        num_envs = self.vec_env.env.task.num_envs

        # Sample expert action probability for each environment
        expert_action_probs = torch.full(
            (num_envs,),
            use_experts,
            dtype=torch.float32,
            device=self.ppo_device
        )
        expert_action_probs = torch.bernoulli(expert_action_probs)

        # Replace student actions with expert actions based on mask
        det_action_mask = expert_action_probs == 1.0
        if expert is not None and len(expert) > 0 and expert.shape[0] == num_envs:
            res_dict['actions'][det_action_mask] = expert[det_action_mask]

        res_dict['expert_mask'] = expert_action_probs

        # rand_action_mask: 1.0 where student policy was used (not expert)
        # This is used to mask losses to only update on student-generated actions
        res_dict['rand_action_mask'] = (1.0 - expert_action_probs).unsqueeze(-1)

        return res_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)

        # Add expert data to dataset
        expert = batch_dict['expert']
        expert_mask = batch_dict['expert_mask']
        rand_action_mask = batch_dict['rand_action_mask']

        self.dataset.values_dict['expert'] = expert
        self.dataset.values_dict['expert_mask'] = expert_mask
        self.dataset.values_dict['rand_action_mask'] = rand_action_mask

        return

    # def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
    #     """
    #     Override actor loss to add numerical stability.
    #     Prevents overflow in exp() by clamping log ratio.
    #     """
    #     # Clamp log probability difference to prevent overflow
    #     # exp(20) ≈ 5e8, exp(-20) ≈ 2e-9 - safe range for float32
    #     log_ratio = old_action_log_probs_batch - action_log_probs
    #     log_ratio = torch.clamp(log_ratio, -20.0, 20.0)

    #     ratio = torch.exp(log_ratio)
    #     surr1 = advantage * ratio
    #     surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
    #     a_loss = torch.max(-surr1, -surr2)

    #     clipped = torch.abs(ratio - 1.0) > curr_e_clip
    #     clipped = clipped.detach()

    #     info = {'actor_loss': a_loss, 'actor_clipped': clipped.detach()}
    #     return info

    def _supervise_loss(self, student, teacher):
        """Compute supervised loss between student and teacher actions"""
        e_loss = (student - teacher) ** 2

        info = {
            'expert_loss': e_loss.sum(dim=-1)
        }
        return info

    def env_step(self, actions):
        """Environment step that returns expert actions"""
        actions = self.preprocess_actions(actions)
        step_out = self.vec_env.step(actions)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            obs, rewards, dones, infos, expert = step_out
        else:
            obs, rewards, dones, infos = step_out
            # Compute expert from task if not provided by vec_env
            expert = None
            task = getattr(self.vec_env, 'task', None)
            if task is not None and hasattr(task, '_get_expert_actions'):
                try:
                    expert_mus, expert_actions = task._get_expert_actions(None, obs)
                    expert = {"mus": expert_mus, "actions": expert_actions}
                except Exception as e:
                    print(f"Warning: Failed to compute expert in env_step: {e}")

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return (
                self.obs_to_tensors(obs),
                rewards.to(self.ppo_device),
                dones.to(self.ppo_device),
                infos,
                expert
            )
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return (
                self.obs_to_tensors(obs),
                torch.from_numpy(rewards).to(self.ppo_device).float(),
                torch.from_numpy(dones).to(self.ppo_device),
                infos,
                expert
            )

    def env_reset(self, env_ids=None):
        """Environment reset that returns expert actions"""
        reset_out = self.vec_env.reset(env_ids)
        expert = None

        # Handle tuple return (obs, expert) from vec_env
        if isinstance(reset_out, (tuple, list)) and len(reset_out) == 2:
            obs, expert = reset_out
        else:
            obs = reset_out

        # Convert obs to tensors first
        obs = self.obs_to_tensors(obs)

        # If expert not provided by vec_env, compute it from task
        if expert is None:
            # Try different ways to access the task
            task = None
            if hasattr(self.vec_env, 'task'):
                task = self.vec_env.task
            elif hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task'):
                task = self.vec_env.env.task

            if task is not None and hasattr(task, '_get_expert_actions'):
                try:
                    # Get obs tensor for expert computation
                    if isinstance(obs, dict) and 'obs' in obs:
                        obs_tensor = obs['obs']
                    else:
                        obs_tensor = obs

                    expert_mus, expert_actions = task._get_expert_actions(env_ids, obs_tensor)
                    expert = {"mus": expert_mus, "actions": expert_actions}
                except Exception as e:
                    print(f"Warning: Failed to compute expert in env_reset (env_ids={env_ids}): {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Debug: print task type
                if task is not None:
                    print(f"Warning: Task type {type(task).__name__} does not have _get_expert_actions method")
                else:
                    print(f"Warning: No task found - vec_env type: {type(self.vec_env).__name__}")

        return obs, expert

    def calc_gradients(self, input_dict):
        """
        Calculate gradients with curriculum learning schedule.

        Curriculum (2k iterations):
        - [0, 500): Expert loss only
        - [500, 600): Expert loss + Critic loss (gradually add critic)
        - [600, 700): Monitor critic EV
        - [700+, EV >= 0.6, streak >= 3): Add Actor loss (gradually transition to RL)
        """
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        expert_mus = input_dict['expert']
        obs_batch = self._preproc_obs(obs_batch)

        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)

        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        # Prepare AMP observations for discriminator
        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)

        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
            'amp_obs': amp_obs,
            'amp_obs_replay': amp_obs_replay,
            'amp_obs_demo': amp_obs_demo
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # Initialize loss components
            a_loss = torch.tensor(0.0, device=self.ppo_device)
            c_loss = torch.tensor(0.0, device=self.ppo_device)
            b_loss = torch.tensor(0.0, device=self.ppo_device)
            a_clip_frac = torch.tensor(0.0, device=self.ppo_device)
            a_info = {}

            # Always compute expert loss
            e_info = self._supervise_loss(mu, expert_mus)
            e_loss = torch.mean(e_info['expert_loss'])

            # Curriculum stage determination
            if rand_action_sum > 0:
                # We have collected some experience with student policy

                # Compute individual loss components ONLY on student-generated samples
                # to avoid evaluating ratio on expert-replaced entries
                student_mask = (rand_action_mask.squeeze(-1) > 0.5)
                if torch.any(student_mask):
                    a_info_student = self._actor_loss(
                        old_action_log_probs_batch[student_mask],
                        action_log_probs[student_mask],
                        advantage[student_mask],
                        curr_e_clip
                    )
                    # Scatter back into full tensors (zeros elsewhere)
                    a_loss_raw = torch.zeros_like(advantage)
                    a_loss_raw[student_mask] = a_info_student['actor_loss']

                    a_clipped_full = torch.zeros_like(advantage, dtype=torch.bool)
                    a_clipped_full[student_mask] = a_info_student['actor_clipped']
                    a_clipped = a_clipped_full.float()
                else:
                    # No student samples in this minibatch
                    a_loss_raw = torch.zeros_like(advantage)
                    a_clipped = torch.zeros_like(advantage)

                c_info = self._critic_loss(
                    value_preds_batch,
                    values,
                    curr_e_clip,
                    return_batch,
                    self.clip_value
                )
                c_loss_raw = c_info['critic_loss']

                # Stage 3: Monitor critic explained variance
                if self.epoch_num >= self.stage3_start:
                    returns_var = return_batch.var(unbiased=False) + 1e-8
                    errors_var = (return_batch - values).var(unbiased=False)
                    ev = 1.0 - errors_var / returns_var
                    self.ev_ma = 0.99 * self.ev_ma + 0.01 * ev.item()

                    if self.ev_ma >= self.ev_threshold:
                        self.critic_win_streak += 1
                    else:
                        self.critic_win_streak = 0

                b_loss_raw = self.bound_loss(mu)

                # Average losses over valid samples
                c_loss = torch.mean(c_loss_raw)
                a_loss = torch.sum(rand_action_mask.squeeze(-1) * a_loss_raw) / rand_action_sum
                entropy_avg = torch.sum(rand_action_mask.squeeze(-1) * entropy) / rand_action_sum
                b_loss = torch.sum(rand_action_mask.squeeze(-1) * b_loss_raw) / rand_action_sum
                a_clip_frac = torch.sum(rand_action_mask.squeeze(-1) * a_clipped) / rand_action_sum

                # Curriculum-based loss composition
                # Check if we should enable actor updates
                # If ev_win_streak_required=0, skip the EV check (always allow actor)
                actor_enabled = (self.epoch_num >= self.stage4_start and
                                (self.ev_win_streak_required == 0 or
                                 self.critic_win_streak >= self.ev_win_streak_required))

                if actor_enabled:
                    # Stage 4: Full RL with decreasing expert loss
                    actor_weight = min((self.actor_update_num / self.actor_rampup_steps), 1.0)
                    expert_weight = max(1.0 - (self.actor_update_num / self.expert_decay_steps), self.expert_min_weight)
                    loss = (
                        actor_weight * a_loss +
                        self.critic_coef * c_loss +
                        self.bounds_loss_coef * b_loss +
                        self.expert_loss_coef * expert_weight * e_loss
                    )
                    self.actor_update_num += 1

                elif self.epoch_num >= self.stage2_start:
                    # Stage 2: Expert + Critic
                    # Gradually ramp up critic between stage2_start and stage2_end
                    critic_rampup_range = self.stage2_end - self.stage2_start
                    if critic_rampup_range > 0:
                        critic_weight = min(((self.epoch_num - self.stage2_start) / critic_rampup_range), 1.0)
                    else:
                        critic_weight = 1.0
                    loss = (
                        critic_weight * self.critic_coef * c_loss +
                        self.expert_loss_coef * e_loss
                    )

                else:
                    # Stage 1: Expert only
                    loss = self.expert_loss_coef * e_loss

            else:
                # # No student experience yet (all expert actions)
                # # Still need to train critic to estimate returns properly!
                # # Compute critic loss for all samples (not masked by rand_action_mask)
                c_info = self._critic_loss(
                    value_preds_batch,
                    values,
                    curr_e_clip,
                    return_batch,
                    self.clip_value
                )
                c_loss = torch.mean(c_info['critic_loss'])

                a_info = {'actor_loss': a_loss, 'actor_clipped': a_clip_frac}

                # Include critic loss even when all actions are from expert
                loss = self.expert_loss_coef * e_loss #+ self.critic_coef * c_loss

            # Prepare info dict
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            # Zero gradients
            if self.distributed_wrapper.is_distributed:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        # Backward pass
        self.scaler.scale(loss).backward()

        # Gradient clipping and optimizer step
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # Compute KL divergence
        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(
                mu.detach(),
                sigma.detach(),
                old_mu_batch,
                old_sigma_batch,
                reduce_kl
            )
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        # Store training results
        self.train_result = {
            'entropy': entropy_avg if rand_action_sum > 0 else torch.tensor(0.0),
            'kl': kl_dist,
            'last_lr': self.last_lr,
            'lr_mul': lr_mul,
            'b_loss': b_loss,
            'ev_ma': self.ev_ma,
            'critic_win_streak': self.critic_win_streak,
            'actor_update_num': self.actor_update_num,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(e_info)

        return

    def _assemble_train_info(self, train_info, frame):
        # Call parent to get base train info
        train_info_dict = super()._assemble_train_info(train_info, frame)

        # Add distillation-specific metrics to train_info_dict for WandB
        if 'expert_loss' in train_info:
            train_info_dict['expert_loss'] = torch_ext.mean_list(train_info['expert_loss']).item()

        # Add curriculum metrics (convert to Python scalars)
        # These values are stored per mini-batch, so we need to take the last value
        if 'ev_ma' in train_info:
            ev_ma_val = train_info['ev_ma']
            # If it's a list, take the last value; otherwise use as-is
            train_info_dict['ev_ma'] = float(ev_ma_val[-1] if isinstance(ev_ma_val, list) else ev_ma_val)
        if 'critic_win_streak' in train_info:
            cws_val = train_info['critic_win_streak']
            train_info_dict['critic_win_streak'] = int(cws_val[-1] if isinstance(cws_val, list) else cws_val)
        if 'actor_update_num' in train_info:
            aun_val = train_info['actor_update_num']
            train_info_dict['actor_update_num'] = int(aun_val[-1] if isinstance(aun_val, list) else aun_val)

        return train_info_dict

    def env_eval_step(self, env, actions):
        """
        Override env_eval_step to handle 5-value return from distillation environment.
        During evaluation, we don't need expert actions.
        """
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        step_result = env.step(actions)

        # Handle 5-value return (obs, rewards, dones, infos, expert)
        if len(step_result) == 5:
            obs, rewards, dones, infos, expert = step_result
        else:
            obs, rewards, dones, infos = step_result

        if hasattr(obs, "dtype") and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def eval(self):
        """
        Override eval to handle tuple return from env_reset.
        During evaluation, we don't need expert actions.
        """
        # Store original env_reset
        original_env_reset = self.env_reset

        # Create wrapper that only returns obs
        def eval_env_reset(env_ids=None):
            result = original_env_reset(env_ids)
            # If tuple, return only obs
            if isinstance(result, tuple):
                return result[0]
            return result

        # Temporarily replace env_reset
        self.env_reset = eval_env_reset

        try:
            # Call parent eval
            eval_result = super().eval()
        finally:
            # Restore original env_reset
            self.env_reset = original_env_reset

        return eval_result

    def _log_train_info(self, train_info_dict, frame):
        # Note: train_info_dict is the processed dict from _assemble_train_info
        super()._log_train_info(train_info_dict, frame)

        # Log distillation-specific metrics to TensorBoard
        # These are already processed to scalars in _assemble_train_info
        if 'expert_loss' in train_info_dict:
            self.writer.add_scalar(
                'losses/e_loss',
                train_info_dict['expert_loss'],
                frame
            )

        # Log curriculum metrics to TensorBoard
        if 'ev_ma' in train_info_dict:
            self.writer.add_scalar('distill/ev_ma', train_info_dict['ev_ma'], frame)
        if 'critic_win_streak' in train_info_dict:
            self.writer.add_scalar('distill/critic_win_streak', train_info_dict['critic_win_streak'], frame)
        if 'actor_update_num' in train_info_dict:
            self.writer.add_scalar('distill/actor_update_num', train_info_dict['actor_update_num'], frame)

        return