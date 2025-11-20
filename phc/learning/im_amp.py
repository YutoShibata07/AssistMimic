

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from phc.utils.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch
from torch import nn
from phc.env.tasks.humanoid_amp_task import HumanoidAMPTask

import learning.replay_buffer as replay_buffer
import phc.learning.amp_agent as amp_agent
from phc.utils.flags import flags
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch.players import rescale_actions

from tensorboardX import SummaryWriter
import joblib
import gc
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from tqdm import tqdm


class IMAmpAgent(amp_agent.AMPAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        

    def get_action(self, obs_dict, is_determenistic=False):
        obs = obs_dict["obs"]

        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        
        if self.clip_actions:
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def env_eval_step(self, env, actions):

        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, "dtype") and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return (
                self.obs_to_torch(obs),
                torch.from_numpy(rewards),
                torch.from_numpy(dones),
                infos,
            )

    def restore(self, fn):
        super().restore(fn)
        
        all_fails = glob.glob(osp.join(self.network_path, f"failed_*"))
        if len(all_fails) > 0:
            print("------------------------------------------------------ Restoring Termination History ------------------------------------------------------")
            failed_pth = sorted(all_fails, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"loading: {failed_pth}")
            termination_history = joblib.load(failed_pth)['termination_history']
            humanoid_env = self.vec_env.env.task
            res = humanoid_env._motion_lib.update_sampling_prob(termination_history)
            if res:
                print("Successfully restored termination history")
            else:
                print("Termination history length does not match")
            
        return
    
    def init_rnn(self):
        if self.is_rnn:
            rnn_states = self.model.get_default_rnn_state()
            self.states = [torch.zeros((s.size()[0], self.vec_env.env.task.num_envs, s.size(
            )[2]), dtype=torch.float32).to(self.device) for s in rnn_states]
            
            
    def update_training_data(self, failed_keys):
        humanoid_env = self.vec_env.env.task
        if humanoid_env.auto_pmcp:
            humanoid_env._motion_lib.update_hard_sampling_weight(failed_keys)
        elif humanoid_env.auto_pmcp_soft:
            humanoid_env._motion_lib.update_soft_sampling_weight(failed_keys)
        elif hasattr(humanoid_env, 'failed_motion_weight') and humanoid_env.failed_motion_weight:
            # Use new failed motion weighted sampling
            failed_weight_multiplier = getattr(humanoid_env, 'failed_weight_multiplier', 2.0)
            humanoid_env._motion_lib.update_failed_motion_weight(failed_keys, failed_weight_multiplier)
        joblib.dump({"failed_keys": failed_keys, "termination_history": humanoid_env._motion_lib._termination_history}, osp.join(self.network_path, f"failed_{self.epoch_num:010d}.pkl"))
        
        
        
    def eval(self):
        print("############################ Evaluation ############################")
        if not flags.has_eval:
            return {}

        # Clear CUDA cache before evaluation to start with clean slate
        torch.cuda.empty_cache()
        gc.collect()

        self.set_eval()

        self.terminate_state = torch.zeros(
            self.vec_env.env.task.num_envs, device=self.device
        )
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        humanoid_env = self.vec_env.env.task
        self.success_rate = 0
        self.pbar = tqdm(
            range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs)
        )
        self.pbar.set_description("")

        ################## Save results first; ZL: Ugllllllllly code, refractor asap ##################
        termination_distances, cycle_motion, zero_out_far, reset_ids = (
            humanoid_env._termination_distances.clone(),
            humanoid_env.cycle_motion,
            humanoid_env.zero_out_far,
            humanoid_env._reset_bodies_id,
        )

        if "_recovery_episode_prob" in humanoid_env.__dict__:
            recovery_episode_prob, fall_init_prob = (
                humanoid_env._recovery_episode_prob,
                humanoid_env._fall_init_prob,
            )
            humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0
        # envにhhi_assist_bed_data_pathがあれば0.3, そうでなければ0.5
        if hasattr(humanoid_env, 'hhi_assist_bed_data_path'):
            humanoid_env._termination_distances[:] = 0.5
            print("Set termination distances to 0.3m for HHI-Assist Bed")
        elif hasattr(humanoid_env, 'interx_data_path'):
            humanoid_env._termination_distances[:] = 0.5
            print("Set termination distances to 0.5m for Inter-X HelpUp")
        else:
            error_msg = "No hhi_assist_bed_data_path or interx_data_path found in env"
            raise RuntimeError(error_msg)
        
        humanoid_env.cycle_motion = False
        humanoid_env.zero_out_far = False
        flags.test, flags.im_eval = (True, True,)  # need to be test to have: motion_times[:] = 0
        humanoid_env._motion_lib = humanoid_env._motion_eval_lib
        humanoid_env.begin_seq_motion_samples()
        # if len(humanoid_env._reset_bodies_id) > 15:
        #         humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
        ################## Save results first; ZL: Ugllllllllly code, refractor asap ##################

        self.print_stats = False
        self.has_batch_dimension = True

        need_init_rnn = self.is_rnn
        obs_dict = self.env_reset()
        batch_size = humanoid_env.num_envs

        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

        cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        
        # Initialize tracking for recipient head max heights during contact
        self.eval_recipient_max_heights = []
        self.eval_episode_recipient_heights = torch.zeros(batch_size, device=self.device)
        self.eval_has_contact_episodes = []

        done_indices = []

        with torch.no_grad():
            while True:
                obs_dict = self.env_reset(done_indices)

                action = self.get_action(obs_dict, is_determenistic=True)
                obs_dict, r, done, info = self.env_eval_step(self.vec_env.env, action)
                cr += r
                steps += 1

                # Track recipient head heights during contact for evaluation
                self._track_eval_recipient_heights()

                done, info = self._post_step_eval(info, done.clone())

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents]
                done_count = len(done_indices)
                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                    done_indices = done_indices[:, 0]

                if info['end']:
                    # Clean up loop variables before breaking
                    del obs_dict, action, r, done, all_done_indices
                    break

        # Clean up temporary evaluation loop variables
        del cr, steps, done_indices

        ################## Save results first; ZL: Ugllllllllly code, refractor asap ##################
        humanoid_env._termination_distances[:] = termination_distances
        humanoid_env.cycle_motion = cycle_motion
        humanoid_env.zero_out_far = zero_out_far
        flags.test, flags.im_eval = False, False
        humanoid_env._motion_eval_lib.clear_cache() # add this
        humanoid_env._motion_lib = humanoid_env._motion_train_lib
        if "_recovery_episode_prob" in humanoid_env.__dict__:
            humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = (
                recovery_episode_prob,
                fall_init_prob,
            )
        humanoid_env._reset_bodies_id = reset_ids

        ################## Save results first; ZL: Ugllllllllly code, refractor asap ##################
        # Extract eval_info before cleaning up to avoid accessing deleted variables
        eval_info_result = info["eval_info"]

        # Comprehensive memory cleanup to prevent memory leaks
        self.update_training_data(info['failed_keys'])

        # Delete all evaluation-related tensors and lists
        del self.terminate_state, self.terminate_memory, self.mpjpe, self.mpjpe_all
        del self.gt_pos, self.gt_pos_all, self.pred_pos, self.pred_pos_all

        # Close and delete progress bar
        if hasattr(self, 'pbar') and self.pbar is not None:
            self.pbar.close()
            del self.pbar

        # Delete evaluation tracking variables
        if hasattr(self, 'eval_recipient_max_heights'):
            del self.eval_recipient_max_heights
        if hasattr(self, 'eval_episode_recipient_heights'):
            del self.eval_episode_recipient_heights
        if hasattr(self, 'eval_has_contact_episodes'):
            del self.eval_has_contact_episodes

        # Delete RNN states if they exist (evaluation may have modified them)
        if hasattr(self, 'states') and self.states is not None:
            # Clear states but don't delete - will be reinitialized by init_rnn if needed
            if isinstance(self.states, list):
                for state in self.states:
                    if state is not None:
                        state.zero_()
            elif torch.is_tensor(self.states):
                self.states.zero_()

        # Delete saved configuration variables to free memory
        del termination_distances, cycle_motion, zero_out_far, reset_ids
        if "_recovery_episode_prob" in humanoid_env.__dict__:
            del recovery_episode_prob, fall_init_prob

        # Force garbage collection and clear CUDA cache BEFORE env_reset to prevent memory buildup
        torch.cuda.empty_cache()
        gc.collect()

        # Now reset ALL environments back to training mode
        # This must come after cleanup to avoid accumulating observation buffers
        self.env_reset()

        # Final cleanup after reset to ensure all temporary buffers are freed
        torch.cuda.empty_cache()
        gc.collect()

        return eval_info_result

    def _post_step_eval(self, info, done):
        end = False
        eval_info = {}
        # modify done such that games will exit and reset.
        humanoid_env = self.vec_env.env.task
        termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
        # termination_state = info["terminate"]
        self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
        if (~self.terminate_state).sum() > 0:
            max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
            curr_ids = humanoid_env._motion_lib._curr_motion_ids
            if (max_possible_id == curr_ids).sum() > 0:
                bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                if (~self.terminate_state[:bound]).sum() > 0:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][
                        ~self.terminate_state[:bound]
                    ].max()
                else:
                    curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()
                
            if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
        else:
            curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

        # Detach tensors from computational graph and move to CPU to prevent GPU memory leaks
        # Move to CPU immediately to reduce GPU memory pressure
        self.mpjpe.append(info["mpjpe"].detach().cpu())
        self.gt_pos.append(info["body_pos_gt"])  # Already numpy array
        self.pred_pos.append(info["body_pos"])   # Already numpy array
        self.curr_stpes += 1

        if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
            self.curr_stpes = 0
            self.terminate_memory.append(self.terminate_state.cpu().numpy())
            self.success_rate = (1- np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

            # MPJPE
            all_mpjpe = torch.stack(self.mpjpe)
            assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
            all_mpjpe_list = [all_mpjpe[:(i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
            # Delete large tensor immediately after converting to list
            del all_mpjpe

            all_body_pos_pred = np.stack(self.pred_pos)
            all_body_pos_pred_list = [all_body_pos_pred[:(i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
            # Delete large numpy array immediately after converting to list
            del all_body_pos_pred

            all_body_pos_gt = np.stack(self.gt_pos)
            all_body_pos_gt_list = [all_body_pos_gt[:(i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
            # Delete large numpy array immediately after converting to list
            del all_body_pos_gt

            self.mpjpe_all.append(all_mpjpe_list)
            self.pred_pos_all += all_body_pos_pred_list
            self.gt_pos_all += all_body_pos_gt_list

            # Delete temporary lists to free memory
            del all_mpjpe_list, all_body_pos_pred_list, all_body_pos_gt_list
            

            if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                self.pbar.clear()
                terminate_hist = np.concatenate(self.terminate_memory)
                succ_idxes = np.flatnonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]).tolist()

                pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
                gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]


                # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                # humanoid_env._motion_lib.get_motion_num_steps().sum()

                failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])

                metrics_all = compute_metrics_lite(pred_pos_all, gt_pos_all)
                metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                metrics_all_print = {m: np.mean(v) for m, v in metrics_all.items()}
                metrics_succ_print = {m: np.mean(v) for m, v in metrics_succ.items()}
                
                if len(metrics_succ_print) == 0:
                    print("No success!!!")
                    metrics_succ_print = metrics_all_print
                
                # Calculate eval recipient head max height statistics
                eval_recipient_head_max_height = 0.0
                if self.eval_recipient_max_heights:
                    # Filter episodes with contact (similar to training logic)
                    valid_heights = [h for h, has_contact in zip(self.eval_recipient_max_heights, self.eval_has_contact_episodes) if has_contact and h > 0]
                    if valid_heights:
                        eval_recipient_head_max_height = np.mean(valid_heights)
                
                # Calculate recipient average torques
                avg_recipient_torque_all = 0.0
                avg_recipient_torque_succ = 0.0
                per_joint_torque_all = {}
                per_joint_torque_succ = {}

                if hasattr(humanoid_env, 'recipient_torque_buffer') and humanoid_env.recipient_torque_buffer:
                    # Calculate torque magnitudes for all episodes
                    all_torque_magnitudes = []
                    succ_torque_magnitudes = []

                    # Organize torques by motion_id for per-joint statistics
                    all_motion_torques = {}  # motion_id -> list of torque arrays
                    succ_motion_torques = {}  # motion_id -> list of torque arrays

                    # Get successful motion indices from terminate_hist
                    success_motion_set = set(succ_idxes)

                    for torque_data in humanoid_env.recipient_torque_buffer:
                        torque_magnitude = np.linalg.norm(torque_data['torques'])
                        all_torque_magnitudes.append(torque_magnitude)

                        motion_id = torque_data.get('motion_id')

                        # Collect torques organized by motion_id
                        if motion_id is not None:
                            if motion_id not in all_motion_torques:
                                all_motion_torques[motion_id] = []
                            all_motion_torques[motion_id].append(torque_data['torques'])

                            # Check if this episode was successful based on motion_id
                            if motion_id in success_motion_set:
                                succ_torque_magnitudes.append(torque_magnitude)
                                if motion_id not in succ_motion_torques:
                                    succ_motion_torques[motion_id] = []
                                succ_motion_torques[motion_id].append(torque_data['torques'])

                    if all_torque_magnitudes:
                        avg_recipient_torque_all = np.mean(all_torque_magnitudes)
                    if succ_torque_magnitudes:
                        avg_recipient_torque_succ = np.mean(succ_torque_magnitudes)

                    # Calculate per-joint torque statistics
                    # For each motion, compute time-averaged absolute torque per joint, then average across motions
                    def compute_per_joint_stats(motion_torques_dict):
                        """Compute per-joint torque statistics from motion_id -> [timesteps] dictionary"""
                        if not motion_torques_dict:
                            return {}

                        # Get number of joints from first torque array
                        first_torques = next(iter(motion_torques_dict.values()))[0]
                        num_joints = len(first_torques)

                        # Compute per-motion time-averaged absolute torques per joint
                        motion_joint_avgs = []  # List of [num_joints] arrays

                        for motion_id, torque_list in motion_torques_dict.items():
                            # torque_list: list of [num_joints] arrays across timesteps
                            torque_array = np.array(torque_list)  # shape: [timesteps, num_joints]

                            # Time-average of absolute torques per joint
                            time_avg_abs_torques = np.mean(np.abs(torque_array), axis=0)  # shape: [num_joints]
                            motion_joint_avgs.append(time_avg_abs_torques)

                        # Stack and compute statistics across motions
                        motion_joint_avgs = np.array(motion_joint_avgs)  # shape: [num_motions, num_joints]

                        stats = {
                            'mean_abs': np.mean(motion_joint_avgs, axis=0).tolist(),  # Mean across motions
                            'std': np.std(motion_joint_avgs, axis=0).tolist(),         # Std across motions
                            'percentile_95': np.percentile(motion_joint_avgs, 95, axis=0).tolist(),
                            'percentile_5': np.percentile(motion_joint_avgs, 5, axis=0).tolist(),
                        }

                        return stats

                    per_joint_torque_all = compute_per_joint_stats(all_motion_torques)
                    per_joint_torque_succ = compute_per_joint_stats(succ_motion_torques)

                    # Add joint names to the statistics
                    if per_joint_torque_succ and hasattr(humanoid_env, '_dof_names'):
                        joint_names = humanoid_env._dof_names
                        # For SMPL/SMPLX with 3 DOF per joint, create full DOF names
                        if hasattr(humanoid_env, '_dof_offsets'):
                            full_dof_names = []
                            for joint_name in joint_names:
                                full_dof_names.extend([
                                    f"{joint_name}_X",
                                    f"{joint_name}_Y",
                                    f"{joint_name}_Z"
                                ])
                            per_joint_torque_all['joint_names'] = full_dof_names
                            per_joint_torque_succ['joint_names'] = full_dof_names
                        else:
                            per_joint_torque_all['joint_names'] = joint_names
                            per_joint_torque_succ['joint_names'] = joint_names

                print("------------------------------------------")
                print(f"Success Rate: {self.success_rate:.10f}")
                print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]) + f"\tavg_torque: {avg_recipient_torque_all:.3f}")
                print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]) + f"\tavg_torque: {avg_recipient_torque_succ:.3f}")

                # Print top 5 joints with highest average torque (if available)
                if per_joint_torque_succ and 'mean_abs' in per_joint_torque_succ:
                    mean_torques = np.array(per_joint_torque_succ['mean_abs'])
                    top5_indices = np.argsort(mean_torques)[-5:][::-1]
                    joint_names = per_joint_torque_succ.get('joint_names', [f'Joint_{i}' for i in range(len(mean_torques))])
                    print("\nTop 5 joints by average torque (successful episodes):")
                    for idx in top5_indices:
                        print(f"  {joint_names[idx]}: {mean_torques[idx]:.2f} ± {per_joint_torque_succ['std'][idx]:.2f}")

                print("Failed keys: ", len(failed_keys), failed_keys)

                end = True

                eval_info = {
                    "eval_success_rate": self.success_rate,
                    "eval_recipient_head_max_height": eval_recipient_head_max_height,
                    "eval_mpjpe_all": metrics_all_print['mpjpe_g'],
                    "eval_mpjpe_succ": metrics_succ_print['mpjpe_g'],
                    "accel_dist": metrics_succ_print['accel_dist'],
                    "vel_dist": metrics_succ_print['vel_dist'],
                    "mpjpel_all": metrics_all_print['mpjpe_l'],
                    "mpjpel_succ": metrics_succ_print['mpjpe_l'],
                    "mpjpe_pa": metrics_succ_print['mpjpe_pa'],
                    "avg_recipient_torque_all": avg_recipient_torque_all,
                    "avg_recipient_torque_succ": avg_recipient_torque_succ,
                    # "per_joint_torque_all": per_joint_torque_all,  # Commented out: dict not compatible with TensorBoard add_scalar
                    # "per_joint_torque_succ": per_joint_torque_succ,  # Commented out: dict not compatible with TensorBoard add_scalar
                }
                
                # failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[:humanoid_env._motion_lib._num_unique_motions]]
                # success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[:humanoid_env._motion_lib._num_unique_motions]]
                # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                # joblib.dump(failed_keys, "output/dgx/smpl_im_shape_long_1/failed_1.pkl")
                # joblib.dump(success_keys, "output/dgx/smpl_im_fit_3_1/long_succ.pkl")
                # print("....")

                # Cleanup large intermediate variables to prevent memory leaks
                result = {"end": end, "eval_info": eval_info, "failed_keys": failed_keys,  "success_keys": success_keys}

                # Explicitly delete large intermediate tensors/arrays
                del pred_pos_all_succ, gt_pos_all_succ, pred_pos_all, gt_pos_all
                del metrics_all, metrics_succ, terminate_hist

                return done, result

            done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

            # Store episode max heights and contact info for this batch
            # for env_idx in range(humanoid_env.num_envs):
            #     max_height = self.eval_episode_recipient_heights[env_idx].item()
            #     # Check if this environment had contact during the episode
            #     has_contact = max_height > 0  # If max height is > 0, contact occurred
            #     self.eval_recipient_max_heights.append(max_height)
            #     self.eval_has_contact_episodes.append(has_contact)
            
            # Reset episode tracking for next batch
            self.eval_episode_recipient_heights.zero_()

            humanoid_env.forward_motion_samples()
            self.terminate_state = torch.zeros(self.vec_env.env.task.num_envs, device=self.device)

            self.pbar.update(1)
            self.pbar.refresh()
            # Clear lists to prevent memory accumulation
            self.mpjpe.clear()
            self.gt_pos.clear()
            self.pred_pos.clear()

            # Force memory cleanup after each batch to prevent gradual accumulation
            torch.cuda.empty_cache()


        update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
        self.pbar.set_description(update_str)

        return done, {"end": end, "eval_info": eval_info, "failed_keys": [],  "success_keys": []}

    def _track_eval_recipient_heights(self):
        """Track recipient head heights during evaluation, only when contact occurs"""
        humanoid_env = self.vec_env.env.task
        
        # Only track if this is SimpleLiftUp mode
        if not hasattr(humanoid_env, 'simple_lift_up_mode') or not humanoid_env.simple_lift_up_mode:
            return
        
        for env_idx in range(humanoid_env.num_envs):
            if env_idx % 2 == 1:  # Recipient environments (odd env_ids)
                # Get current head height
                head_height = humanoid_env._get_head_height(env_idx)
                
                # Check if there is contact between caregiver and recipient
                caregiver_env_idx = env_idx - 1  # Caregiver is env_idx - 1
                # has_contact = humanoid_env._check_caregiver_recipient_hand_contact(caregiver_env_idx, env_idx)
                
                # # Only update max height when contact is present (same logic as training)
                # if has_contact and head_height > 0:
                #     self.eval_episode_recipient_heights[env_idx] = max(
                #         self.eval_episode_recipient_heights[env_idx], head_height
                #     )
