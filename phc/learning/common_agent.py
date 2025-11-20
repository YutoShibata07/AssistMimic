import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml
import glob
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from rl_games.algos_torch import a2c_continuous, a2c_discrete
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim
import torch.nn as nn
from gym import spaces
import gc

import learning.amp_datasets as amp_datasets
from learning.distributed_wrapper import DistributedWrapper

from tensorboardX import SummaryWriter
import wandb


class CommonAgent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, config):
        # Handle distributed training setup before calling parent init
        self.distributed_wrapper = DistributedWrapper()
        # Set alias for compatibility with MultiPulse code
        self.hvd = self.distributed_wrapper
        
        # Update config for distributed training if needed
        if self.distributed_wrapper.is_distributed:
            config = self.distributed_wrapper.update_algo_config(config)
            # Disable rl_games multi_gpu to avoid horovod import - handle different config structures
            if 'params' in config and 'config' in config['params'] and 'multi_gpu' in config['params']['config']:
                config['params']['config']['multi_gpu'] = False
            elif 'config' in config and 'multi_gpu' in config['config']:
                config['config']['multi_gpu'] = False
            elif 'multi_gpu' in config:
                config['multi_gpu'] = False
        
        a2c_common.A2CBase.__init__(self, base_name, config)
        self.cfg = config
        self.exp_name = self.cfg['train_dir'].split('/')[-1]

        self._load_config_params(config)
        
        # Ensure multi_gpu is False for distributed training to avoid horovod conflicts
        if self.distributed_wrapper.is_distributed:
            self.multi_gpu = False
            print(f"Distributed training detected: Setting multi_gpu=False, using PyTorch DDP instead")

        self.is_discrete = False
        self._setup_action_space()
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        
        self.clip_actions = config.get('clip_actions', True)
        self._save_intermediate = config.get('save_intermediate', False)

        net_config = self._build_net_config()
        
        if self.normalize_input:
            if "vec_env" in self.__dict__:
                obs_shape = torch_ext.shape_whc_to_cwh(self.vec_env.env.task.get_running_mean_size())
            else:
                obs_shape = self.obs_shape
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)
            
        net_config['mean_std'] = self.running_mean_std
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)

        # Store weight_share configuration for checkpoint saving
        if hasattr(self.model, 'a2c_network') and hasattr(self.model.a2c_network, 'weight_share'):
            self.weight_share = self.model.a2c_network.weight_share
            print(f"DEBUG: Retrieved weight_share from model.a2c_network: {self.weight_share}")
        else:
            self.weight_share = True  # Default to True for backward compatibility
            print(f"DEBUG: Using default weight_share: {self.weight_share}")

        # Set environment reference for asymmetric critic if supported
        if hasattr(self.model, 'a2c_network') and hasattr(self.model.a2c_network, 'set_env_ref'):
            try:
                self.model.a2c_network.set_env_ref(self.vec_env.env.task)
                print("✓ Environment reference set for asymmetric critic")
            except Exception as e:
                print(f"✗ Failed to set environment reference: {e}")
        
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        # Initialize default optimizer first
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.use_role_based_optimizers = False

        if self.has_central_value:
            cv_config = {
                'state_shape': torch_ext.shape_whc_to_cwh(self.state_shape),
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'model': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'multi_gpu': self.distributed_wrapper.is_distributed
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)

        return

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])

        self.tensor_list += ['next_obses']
        return

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = osp.join(self.network_path, self.config['name'])

        if self.distributed_wrapper.is_distributed:
            self.distributed_wrapper.setup_algo(self)

        self._init_train()

        while True:
            epoch_start = time.time()

            epoch_num = self.update_epoch()

            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame
            if self.distributed_wrapper.is_distributed:
                self.distributed_wrapper.sync_stats(self)

            if self.rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time

                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                train_info_dict = self._assemble_train_info(train_info, frame)
                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                if self.save_freq > 0:
                    
                    if epoch_num % min(50, self.save_best_after) == 0:
                        self.save(model_output_file)
                    
                    if (self._save_intermediate) and (epoch_num % (self.save_freq) == 0):
                        # Save intermediate model every save_freq  epoches
                        int_model_output_file = model_output_file + '_' + str(epoch_num).zfill(8)
                        self.save(int_model_output_file)
                        
                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()

                    for i in range(self.value_size):
                        self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)

                    if (self._save_intermediate) and (epoch_num % (self.save_freq) == 0):
                        eval_info = self.eval()
                        train_info_dict.update(eval_info)
                        self.model.eval()
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    train_info_dict.update({"episode_lengths": mean_lengths, "mean_rewards": np.mean(mean_rewards)})
                    self._log_train_info(train_info_dict, frame)

                    epoch_end = time.time()
                    
                    # Get hand contact reward info if available
                    hand_contact_info = ""
                    if hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task') and hasattr(self.vec_env.env.task, 'hand_contact_reward'):
                        hand_reward = self.vec_env.env.task.hand_contact_reward
                        hand_mean = hand_reward.mean().item()
                        hand_active = (hand_reward > 0).sum().item()
                        hand_contact_info = f"\thand_contact: {hand_mean:.4f} ({hand_active}/{len(hand_reward)} active)"
                        
                        # Add hand contact stats to train_info_dict for wandb logging
                        train_info_dict.update({
                            "hand_contact_reward_mean": hand_mean,
                            "hand_contact_active_envs": hand_active,
                            "hand_contact_active_ratio": hand_active / len(hand_reward)
                        })
                    
                    # Get recipient max root height info if available
                    recipient_height_info = ""
                    if hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task') and hasattr(self.vec_env.env.task, 'episode_max_recipient_heights'):
                        max_heights = self.vec_env.env.task.episode_max_recipient_heights
                        if max_heights is not None and len(max_heights) > 0:
                            # Only consider recipient environments (odd env_ids)
                            recipient_indices = torch.arange(1, len(max_heights), 2, device=max_heights.device)
                            if len(recipient_indices) > 0:
                                recipient_max_heights = max_heights[recipient_indices].clone()
                                
                                # Zero out heights for environments with insufficient hand contacts
                                task = self.vec_env.env.task
                                if (hasattr(task, 'zero_reward_on_poor_contact') and task.zero_reward_on_poor_contact and
                                    hasattr(task, 'episode_hand_contact_count') and hasattr(task, 'min_required_hand_contacts')):
                                    
                                    contact_counts = task.episode_hand_contact_count
                                    min_contacts = task.min_required_hand_contacts
                                    
                                    # Check which recipient environments have insufficient contacts
                                    for i, recipient_env_id in enumerate(recipient_indices):
                                        if recipient_env_id < len(contact_counts):
                                            contact_count = contact_counts[recipient_env_id].item()
                                            if contact_count < min_contacts:
                                                recipient_max_heights[i] = 0.0
                                
                                height_mean = recipient_max_heights.mean().item()
                                recipient_height_info = f"\trecipient_max_height: {height_mean:.4f}"
                                
                                # Add recipient height stats to train_info_dict for wandb logging
                                train_info_dict.update({
                                    "recipient_max_height_mean": height_mean,
                                    "recipient_max_height_max": recipient_max_heights.max().item(),
                                    "recipient_max_height_min": recipient_max_heights.min().item()
                                })
                    
                    log_str = f"{self.exp_name}-Ep: {self.epoch_num}\trwd: {np.mean(mean_rewards):.1f}\tfps_step: {fps_step:.1f}\tfps_total: {fps_total:.1f}\tep_time:{epoch_end - epoch_start:.1f}\tframe: {self.frame}\teps_len: {mean_lengths:.1f}{hand_contact_info}{recipient_height_info}"
                    
                    print(log_str)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if epoch_num > self.max_epochs:
                    self.save(model_output_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num

                update_time = 0
        return

    def eval(self):
        print("evaluation routine not implemented")
        return {}

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.distributed_wrapper.is_distributed:
                        pass  # DDP automatically averages gradients
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.distributed_wrapper.is_distributed:
                    pass  # DDP automatically averages gradients
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.distributed_wrapper.is_distributed:
                pass  # DDP automatically averages gradients
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)
        return train_info
    
    def get_action_values(self, obs):
        obs_orig = obs['obs']
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            "obs_orig": obs_orig,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    #'actions' : res_dict['action'],
                    #'rnn_states' : self.rnn_states
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def play_steps(self):
        self.set_eval()

        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)

            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            # Store successful trajectories before resetting rewards
            if len(done_indices) > 0:
                episode_rewards = self.current_rewards[done_indices].squeeze(-1)  # Remove extra dimension
                if hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task') and hasattr(self.vec_env.env.task, 'store_successful_trajectories'):
                    self.vec_env.env.task.store_successful_trajectories(done_indices.squeeze(-1), episode_rewards)

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = self._calc_advs(batch_dict)

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        
        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)
            
        self.dataset.update_values_dict(dataset_dict)
        return dataset_dict

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {'is_train': True, 'prev_actions': actions_batch, 'obs': obs_batch}

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

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)
            
            # gotta average
            a_loss = torch.mean(a_loss)
            c_loss = torch.mean(c_loss)
            b_loss = torch.mean(b_loss)
            entropy = torch.mean(entropy)

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss

            a_clip_frac = torch.mean(a_info['actor_clipped'].float())

            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac

            if self.distributed_wrapper.is_distributed:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None
                    
            self.scaler.scale(loss).backward()
        
        #TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.distributed_wrapper.is_distributed:
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

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.train_result = {'entropy': entropy, 'kl': kl_dist, 'last_lr': self.last_lr, 'lr_mul': lr_mul, 'b_loss': b_loss}
        self.train_result.update(a_info)
        self.train_result.update(c_info)

        return

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _get_mean_rewards(self):
        return self.game_rewards.get_mean()

    def _load_config_params(self, config):
        self.last_lr = config['learning_rate']
        # Load role-based learning rates if specified
        self.caregiver_learning_rate = config.get('caregiver_learning_rate', self.last_lr)
        self.recipient_learning_rate = config.get('recipient_learning_rate', self.last_lr)
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
        }
        return config

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        return

    def _init_train(self):
        return

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs_dict['obs'] = self._preproc_obs(obs_dict['obs'])
        if self.model.is_rnn():
            value, state = self.model.a2c_network.eval_critic(obs_dict)
        else:
            value = self.model.a2c_network.eval_critic(obs_dict)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()

        info = {'actor_loss': a_loss, 'actor_clipped': clipped.detach()}
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        if clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        info = {'critic_loss': c_loss}
        return info

    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _record_train_batch_info(self, batch_dict, train_info):
        return
    
    def _assemble_train_info(self, train_info, frame):
        train_info_dict = {
            "update_time": train_info['update_time'],
            "play_time": train_info['play_time'],
            "last_lr": train_info['last_lr'][-1] * train_info['lr_mul'][-1],
            "lr_mul": train_info['lr_mul'][-1],
            "e_clip": self.e_clip * train_info['lr_mul'][-1],
        }
        
        if "actor_loss" in train_info:
            train_info_dict.update(
                {
                    "a_loss": torch_ext.mean_list(train_info['actor_loss']).item(),
                    "c_loss": torch_ext.mean_list(train_info['critic_loss']).item(),
                    "bounds_loss": torch_ext.mean_list(train_info['b_loss']).item(),
                    "entropy": torch_ext.mean_list(train_info['entropy']).item(),
                    "clip_frac": torch_ext.mean_list(train_info['actor_clip_frac']).item(),
                    "kl": torch_ext.mean_list(train_info['kl']).item(),
                }
            )
        
        # Add contact statistics from environment if available
        contact_stats = self._get_contact_stats()
        if contact_stats:
            train_info_dict.update(contact_stats)
            # Debug: Print contact stats keys periodically
            if frame % 10000 == 0:  # Every 10k frames
                print(f"DEBUG: contact_stats keys sent to wandb: {list(contact_stats.keys())}")
        
        # Add episode max recipient heights if available
        episode_max_heights = self._get_episode_max_recipient_heights()
        if episode_max_heights is not None:
            train_info_dict.update({"episode_max_recipient_heights": episode_max_heights})
        
        return train_info_dict

    def _get_contact_stats(self):
        """Get contact statistics from the environment task"""
        # Try different possible paths to access the environment
        if hasattr(self, 'vec_env') and hasattr(self.vec_env, 'task') and hasattr(self.vec_env.task, '_contact_stats'):
            return self.vec_env.task._contact_stats
        
        if hasattr(self, 'vec_env') and hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task') and hasattr(self.vec_env.env.task, '_contact_stats'):
            return self.vec_env.env.task._contact_stats
        
        if hasattr(self, 'vec_env') and hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, '_contact_stats'):
            return self.vec_env.env._contact_stats
        
        # # Debug: Print available attributes to understand the structure
        # if hasattr(self, 'vec_env'):
        #     print(f"vec_env attributes: {dir(self.vec_env)}")
        #     if hasattr(self.vec_env, 'task'):
        #         print(f"vec_env.task attributes: {dir(self.vec_env.task)}")
        #     if hasattr(self.vec_env, 'env'):
        #         print(f"vec_env.env attributes: {dir(self.vec_env.env)}")
        #         if hasattr(self.vec_env.env, 'task'):
        #             print(f"vec_env.env.task attributes: {dir(self.vec_env.env.task)}")
        
        return {}

    def _get_episode_max_recipient_heights(self):
        """Get episode max recipient heights from the environment task"""
        # Try different possible paths to access the environment
        if hasattr(self, 'vec_env') and hasattr(self.vec_env, 'task') and hasattr(self.vec_env.task, '_latest_episode_max_heights_mean'):
            return self.vec_env.task._latest_episode_max_heights_mean
        
        if hasattr(self, 'vec_env') and hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task') and hasattr(self.vec_env.env.task, '_latest_episode_max_heights_mean'):
            return self.vec_env.env.task._latest_episode_max_heights_mean
        
        if hasattr(self, 'vec_env') and hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, '_latest_episode_max_heights_mean'):
            return self.vec_env.env._latest_episode_max_heights_mean
        
        return None

    def _log_train_info(self, train_info, frame):
        
        for k, v in train_info.items():
            self.writer.add_scalar(k, v, self.epoch_num)
        
        if not wandb.run is None:
            wandb.log(train_info, step=self.epoch_num)
       
        return 

    def post_epoch(self, epoch_num):
        pass



class CommonDiscreteAgent(a2c_discrete.DiscreteA2CAgent):

    def __init__(self, base_name, config):
        a2c_common.DiscreteA2CBase.__init__(self, base_name, config)
        self.cfg = config
        self.exp_name = self.cfg['train_dir'].split('/')[-1]

        self._load_config_params(config)

        self._setup_action_space()
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        
        self.clip_actions = config.get('clip_actions', True)
        self._save_intermediate = config.get('save_intermediate', False)

        net_config = self._build_net_config()
        if self.normalize_input:
            obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)
        net_config['mean_std'] = self.running_mean_std
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)

        # Store weight_share configuration for checkpoint saving
        if hasattr(self.model, 'a2c_network') and hasattr(self.model.a2c_network, 'weight_share'):
            self.weight_share = self.model.a2c_network.weight_share
            print(f"DEBUG: Retrieved weight_share from model.a2c_network: {self.weight_share}")
        else:
            self.weight_share = True  # Default to True for backward compatibility
            print(f"DEBUG: Using default weight_share: {self.weight_share}")

        # Set environment reference for asymmetric critic if supported
        if hasattr(self.model, 'a2c_network') and hasattr(self.model.a2c_network, 'set_env_ref'):
            try:
                self.model.a2c_network.set_env_ref(self.vec_env.env.task)
                print("✓ Environment reference set for asymmetric critic")
            except Exception as e:
                print(f"✗ Failed to set environment reference: {e}")
        
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        # Initialize default optimizer first
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.use_role_based_optimizers = False

        if self.has_central_value:
            cv_config = {
                'state_shape': torch_ext.shape_whc_to_cwh(self.state_shape),
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'model': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'multi_gpu': self.distributed_wrapper.is_distributed
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)

        return

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])

        self.tensor_list += ['next_obses']
        return

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = osp.join(self.network_path, self.config['name'])

        if self.distributed_wrapper.is_distributed:
            self.distributed_wrapper.setup_algo(self)

        self._init_train()

        while True:
            epoch_start = time.time()

            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame
            if self.distributed_wrapper.is_distributed:
                self.distributed_wrapper.sync_stats(self)

            if self.rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time

                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                train_info_dict = self._assemble_train_info(train_info, frame)
                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                if self.save_freq > 0:
                    
                    if epoch_num % min(50, self.save_best_after) == 0:
                        self.save(model_output_file)
                    
                    if (self._save_intermediate) and (epoch_num % (self.save_freq) == 0):
                        # Save intermediate model every save_freq  epoches
                        int_model_output_file = model_output_file + '_' + str(epoch_num).zfill(8)
                        self.save(int_model_output_file)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()

                    for i in range(self.value_size):
                        self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)

                    if (self._save_intermediate) and (epoch_num % (self.save_freq) == 0):
                        eval_info = self.eval()
                        train_info_dict.update(eval_info)
                        self.model.eval()
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    train_info_dict.update({"episode_lengths": mean_lengths, "mean_rewards": np.mean(mean_rewards)})
                    self._log_train_info(train_info_dict, frame)

                    epoch_end = time.time()
                    
                    # Get hand contact reward info if available
                    hand_contact_info = ""
                    if hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task') and hasattr(self.vec_env.env.task, 'hand_contact_reward'):
                        hand_reward = self.vec_env.env.task.hand_contact_reward
                        hand_mean = hand_reward.mean().item()
                        hand_active = (hand_reward > 0).sum().item()
                        hand_contact_info = f"\thand_contact: {hand_mean:.4f} ({hand_active}/{len(hand_reward)} active)"
                        
                        # Add hand contact stats to train_info_dict for wandb logging
                        train_info_dict.update({
                            "hand_contact_reward_mean": hand_mean,
                            "hand_contact_active_envs": hand_active,
                            "hand_contact_active_ratio": hand_active / len(hand_reward)
                        })
                    
                    # Get recipient max root height info if available
                    recipient_height_info = ""
                    if hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task') and hasattr(self.vec_env.env.task, 'episode_max_recipient_heights'):
                        max_heights = self.vec_env.env.task.episode_max_recipient_heights
                        if max_heights is not None and len(max_heights) > 0:
                            # Only consider recipient environments (odd env_ids)
                            recipient_indices = torch.arange(1, len(max_heights), 2, device=max_heights.device)
                            if len(recipient_indices) > 0:
                                recipient_max_heights = max_heights[recipient_indices].clone()
                                
                                # Zero out heights for environments with insufficient hand contacts
                                task = self.vec_env.env.task
                                if (hasattr(task, 'zero_reward_on_poor_contact') and task.zero_reward_on_poor_contact and
                                    hasattr(task, 'episode_hand_contact_count') and hasattr(task, 'min_required_hand_contacts')):
                                    
                                    contact_counts = task.episode_hand_contact_count
                                    min_contacts = task.min_required_hand_contacts
                                    
                                    # Check which recipient environments have insufficient contacts
                                    for i, recipient_env_id in enumerate(recipient_indices):
                                        if recipient_env_id < len(contact_counts):
                                            contact_count = contact_counts[recipient_env_id].item()
                                            if contact_count < min_contacts:
                                                recipient_max_heights[i] = 0.0
                                
                                height_mean = recipient_max_heights.mean().item()
                                recipient_height_info = f"\trecipient_max_height: {height_mean:.4f}"
                                
                                # Add recipient height stats to train_info_dict for wandb logging
                                train_info_dict.update({
                                    "recipient_max_height_mean": height_mean,
                                    "recipient_max_height_max": recipient_max_heights.max().item(),
                                    "recipient_max_height_min": recipient_max_heights.min().item()
                                })
                    
                    log_str = f"{self.exp_name}-Ep: {self.epoch_num}\trwd: {np.mean(mean_rewards):.1f}\tfps_step: {fps_step:.1f}\tfps_total: {fps_total:.1f}\tep_time:{epoch_end - epoch_start:.1f}\tframe: {self.frame}\teps_len: {mean_lengths:.1f}{hand_contact_info}{recipient_height_info}"
                    print(log_str)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if epoch_num > self.max_epochs:
                    self.save(model_output_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num

                update_time = 0
        return

    def eval(self):
        print("evaluation routine not implemented")
        return {}

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.distributed_wrapper.is_distributed:
                        pass  # DDP automatically averages gradients
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.distributed_wrapper.is_distributed:
                    pass  # DDP automatically averages gradients
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.distributed_wrapper.is_distributed:
                pass  # DDP automatically averages gradients
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)
        return train_info

    def play_steps(self):
        self.set_eval()

        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)

            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            # Store successful trajectories before resetting rewards
            if len(done_indices) > 0:
                episode_rewards = self.current_rewards[done_indices].squeeze(-1)  # Remove extra dimension
                if hasattr(self.vec_env, 'env') and hasattr(self.vec_env.env, 'task') and hasattr(self.vec_env.env.task, 'store_successful_trajectories'):
                    self.vec_env.env.task.store_successful_trajectories(done_indices.squeeze(-1), episode_rewards)

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = self._calc_advs(batch_dict)

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        
        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)
            
        self.dataset.update_values_dict(dataset_dict)
        return dataset_dict


    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def _get_mean_rewards(self):
        return self.game_rewards.get_mean()

    def _load_config_params(self, config):
        self.last_lr = config['learning_rate']
        # Load role-based learning rates if specified
        self.caregiver_learning_rate = config.get('caregiver_learning_rate', self.last_lr)
        self.recipient_learning_rate = config.get('recipient_learning_rate', self.last_lr)
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
        }
        return config

    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape
        
        batch_size = self.num_agents * self.num_actors
        if type(action_space) is spaces.Discrete:
            self.actions_shape = (self.horizon_length, batch_size)
            self.actions_num = action_space.n
            self.is_multi_discrete = False
        if type(action_space) is spaces.Tuple:
            self.actions_shape = (self.horizon_length, batch_size, len(action_space)) 
            self.actions_num = [action.n for action in action_space]
            self.is_multi_discrete = True
        return

    def _init_train(self):
        return

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs_dict['obs'] = self._preproc_obs(obs_dict['obs'])
        if self.model.is_rnn():
            value, state = self.model.a2c_network.eval_critic(obs_dict)
        else:
            value = self.model.a2c_network.eval_critic(obs_dict)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()

        info = {'actor_loss': a_loss, 'actor_clipped': clipped.detach()}
        return info

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        if clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        info = {'critic_loss': c_loss}
        return info

    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _record_train_batch_info(self, batch_dict, train_info):
        return
    
    def _assemble_train_info(self, train_info, frame):
        train_info_dict = {
            "update_time": train_info['update_time'],
            "play_time": train_info['play_time'],
            "a_loss": torch_ext.mean_list(train_info['actor_loss']).item(),
            "c_loss": torch_ext.mean_list(train_info['critic_loss']).item(),
            "entropy": torch_ext.mean_list(train_info['entropy']).item(),
            "last_lr": train_info['last_lr'][-1] * train_info['lr_mul'][-1],
            "lr_mul": train_info['lr_mul'][-1],
            "e_clip": self.e_clip * train_info['lr_mul'][-1],
            "clip_frac": torch_ext.mean_list(train_info['actor_clip_frac']).item(),
            "kl": torch_ext.mean_list(train_info['kl']).item(),
        }
        
        # Add contact statistics from environment if available
        contact_stats = self._get_contact_stats()
        if contact_stats:
            train_info_dict.update(contact_stats)
            # Debug: Print contact stats keys periodically
            if frame % 10000 == 0:  # Every 10k frames
                print(f"DEBUG: contact_stats keys sent to wandb: {list(contact_stats.keys())}")
        
        # Add episode max recipient heights if available
        episode_max_heights = self._get_episode_max_recipient_heights()
        if episode_max_heights is not None:
            train_info_dict.update({"episode_max_recipient_heights": episode_max_heights})
        
        return train_info_dict

    def _log_train_info(self, train_info, frame):
        
        for k, v in train_info.items():
            self.writer.add_scalar(k, v, self.epoch_num)
        
        if not wandb.run is None:
            wandb.log(train_info, step=self.epoch_num)
       
        return 

    def post_epoch(self, epoch_num):
        pass
    
    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.vec_env.env.task.set_char_color(rand_col, env_ids)
        return


    def _setup_role_based_optimizers(self):
        """Setup separate optimizers for caregiver and recipient networks when weight_share=False"""
        print(f"Setting up role-based optimizers: caregiver_lr={self.caregiver_learning_rate}, recipient_lr={self.recipient_learning_rate}")

        # Collect parameters for each role
        caregiver_params = []
        recipient_params = []
        shared_params = []

        for name, param in self.model.named_parameters():
            if 'caregiver_aux_mlp' in name or 'caregiver_final_fc' in name:
                caregiver_params.append(param)
            elif 'recipient_aux_mlp' in name or 'recipient_final_fc' in name:
                recipient_params.append(param)
            else:
                # Shared parameters (PNN, sigma, etc.) - add to both optimizers
                shared_params.append(param)

        # Create separate optimizers with different learning rates
        caregiver_all_params = caregiver_params + shared_params
        recipient_all_params = recipient_params + shared_params

        self.caregiver_optimizer = optim.Adam(caregiver_all_params, lr=float(self.caregiver_learning_rate),
                                            eps=1e-08, weight_decay=self.weight_decay)
        self.recipient_optimizer = optim.Adam(recipient_all_params, lr=float(self.recipient_learning_rate),
                                            eps=1e-08, weight_decay=self.weight_decay)

        # Set primary optimizer to caregiver for compatibility
        self.optimizer = self.caregiver_optimizer
        self.use_role_based_optimizers = True

        print(f"Created caregiver optimizer with {len(caregiver_all_params)} parameters")
        print(f"Created recipient optimizer with {len(recipient_all_params)} parameters")

    def setup_role_based_optimizers_if_needed(self):
        """Setup role-based optimizers after full initialization if conditions are met"""
        if hasattr(self, 'weight_share') and not self.weight_share and (float(self.caregiver_learning_rate) != float(self.recipient_learning_rate)):
            print(f"Setting up role-based optimizers after initialization")
            self._setup_role_based_optimizers()
            return True
        return False

    def train_epoch(self):
        """Override train_epoch to handle role-based optimizers"""
        if hasattr(self, 'use_role_based_optimizers') and self.use_role_based_optimizers:
            return self._train_epoch_role_based()
        else:
            return super().train_epoch()

    def _train_epoch_role_based(self):
        """Train epoch with separate caregiver and recipient optimizers"""
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()

        self.set_train()

        train_info = None

        if self.has_central_value:
            train_info = self.train_central_value(batch_dict)

        losses = []
        for _ in range(0, self.mini_epochs):
            ep_kls = []
            for i in range(len(batch_dict['obs'])):
                curr_train_info = self._train_role_based_critic(batch_dict, i)

                if train_info is None:
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

                ep_kls.append(curr_train_info['kl'])

            av_kls = torch_ext.mean_list(ep_kls)

            if self.has_central_value:
                train_info['central_value_loss'] = self.central_value_net.update()

        if self.multi_gpu:
            self.hvd.average_value(train_info['kl'], 'kl')
        if train_info:
            for k, v in train_info.items():
                train_info[k] = torch_ext.mean_list(v)

        if av_kls.item() > self.e_clip and self.lr_schedule != 'legacy':
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)
        return train_info

    def _train_role_based_critic(self, input_dict, batch_num):
        """Train critic with role-based optimizers"""
        self.calc_gradients(input_dict)
        return_info = {}

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(input_dict['mu'], input_dict['sigma'], input_dict['old_mu'], input_dict['old_sigma'], reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist).mean()

        self.trancate_gradients_and_step()

        with torch.no_grad():
            return_info = {
                'kl': kl_dist
            }
            if self.bounds_loss_coef is not None:
                return_info['bounds_loss'] = input_dict['bounds_loss']

        return return_info

    def trancate_gradients_and_step(self):
        """Override gradient truncation and step for role-based optimizers"""
        if hasattr(self, 'use_role_based_optimizers') and self.use_role_based_optimizers:
            if self.truncate_grads:
                if self.multi_gpu:
                    self.hvd.synchronize()
                self.scaler.unscale_(self.caregiver_optimizer)
                self.scaler.unscale_(self.recipient_optimizer)
                torch.nn.utils.clip_grad_norm_(self.caregiver_optimizer.param_groups[0]['params'], self.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.recipient_optimizer.param_groups[0]['params'], self.grad_norm)

            self.scaler.step(self.caregiver_optimizer)
            self.scaler.step(self.recipient_optimizer)
            self.scaler.update()

            self.caregiver_optimizer.zero_grad()
            self.recipient_optimizer.zero_grad()
        else:
            return super().trancate_gradients_and_step()

    def update_lr(self, lr):
        """Update learning rate for role-based optimizers"""
        if hasattr(self, 'use_role_based_optimizers') and self.use_role_based_optimizers:
            # Update both optimizers proportionally
            caregiver_ratio = float(self.caregiver_learning_rate) / float(self.last_lr)
            recipient_ratio = float(self.recipient_learning_rate) / float(self.last_lr)

            new_caregiver_lr = lr * caregiver_ratio
            new_recipient_lr = lr * recipient_ratio

            for param_group in self.caregiver_optimizer.param_groups:
                param_group['lr'] = new_caregiver_lr
            for param_group in self.recipient_optimizer.param_groups:
                param_group['lr'] = new_recipient_lr

            print(f"Updated learning rates - Caregiver: {new_caregiver_lr}, Recipient: {new_recipient_lr}")
        else:
            return super().update_lr(lr)
