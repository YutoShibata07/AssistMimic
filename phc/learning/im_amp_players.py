

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import torch
import yaml
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
from phc.utils.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import learning.amp_players as amp_players
from tqdm import tqdm
import joblib
import time
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from rl_games.common.tr_helpers import unsqueeze_obs

COLLECT_Z = False

class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)

        self.terminate_state = torch.zeros(self.env.task.num_envs, device=self.device)
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        if COLLECT_Z:
            self.zs, self.zs_all = [], []

        humanoid_env = self.env.task
        humanoid_env._termination_distances[:] = humanoid_env.cfg["env"]["terminationDistance"]  # Use env config termination distance
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0

        if flags.im_eval:
            self.success_rate = 0
            self.pbar = tqdm(range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs))
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False
            
            # if len(humanoid_env._reset_bodies_id) > 15:
            #     humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
            
            humanoid_env.cycle_motion = False
            self.print_stats = False
        
        # joblib.dump({"mlp": self.model.a2c_network.actor_mlp, "mu": self.model.a2c_network.mu}, "single_model.pkl") # ZL: for saving part of the model.
        return

    def _post_step(self, info, done):
        super()._post_step(info)
        
        
        # modify done such that games will exit and reset.
        if flags.im_eval:

            humanoid_env = self.env.task
            
            termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            # termination_state = info["terminate"]
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            if (~self.terminate_state).sum() > 0:
                max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
                curr_ids = humanoid_env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                else:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            if COLLECT_Z: self.zs.append(info["z"])
            self.curr_stpes += 1

            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                self.success_rate = (1 - np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)
                try:
                    assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
                except:
                    import ipdb; ipdb.set_trace()
                    print('??')

                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                if COLLECT_Z:
                    all_zs = torch.stack(self.zs)
                    all_zs = [all_zs[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.zs_all += all_zs


                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                

                if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                    terminate_hist = np.concatenate(self.terminate_memory)
                    succ_idxes = np.nonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions])[0].tolist()

                    pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                    pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
                    gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]

                    # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                    # humanoid_env._motion_lib.get_motion_num_steps().sum()

                    failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                    if flags.real_traj:
                        pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
                        gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]
                        pred_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all_succ]
                        gt_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all_succ]
                        
                        
                        
                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    # Calculate recipient-specific metrics (odd environment indices)
                    # In interaction mode, odd env_ids (1, 3, 5, ...) are recipients
                    recipient_indices = [i for i in range(len(pred_pos_all)) if i % 2 == 1]
                    recipient_succ_indices = [i for i in succ_idxes if i % 2 == 1]

                    if len(recipient_indices) > 0:
                        pred_pos_recipient = [pred_pos_all[i] for i in recipient_indices]
                        gt_pos_recipient = [gt_pos_all[i] for i in recipient_indices]
                        metrics_recipient = compute_metrics_lite(pred_pos_recipient, gt_pos_recipient)
                        metrics_recipient_print = {m: np.mean(v) for m, v in metrics_recipient.items()}
                    else:
                        metrics_recipient_print = {}

                    if len(recipient_succ_indices) > 0:
                        pred_pos_recipient_succ = [pred_pos_all[i] for i in recipient_succ_indices]
                        gt_pos_recipient_succ = [gt_pos_all[i] for i in recipient_succ_indices]
                        metrics_recipient_succ = compute_metrics_lite(pred_pos_recipient_succ, gt_pos_recipient_succ)
                        metrics_recipient_succ_print = {m: np.mean(v) for m, v in metrics_recipient_succ.items()}
                    else:
                        metrics_recipient_succ_print = {}

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
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]) + f"\tavg_torque: {avg_recipient_torque_all:.3f}")
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]) + f"\tavg_torque: {avg_recipient_torque_succ:.3f}")
                    if metrics_recipient_print:
                        print("Recipient All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_recipient_print.items()]))
                    if metrics_recipient_succ_print:
                        print("Recipient Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_recipient_succ_print.items()]))

                    # Print top 5 joints with highest average torque (if available)
                    if per_joint_torque_succ and 'mean_abs' in per_joint_torque_succ:
                        mean_torques = np.array(per_joint_torque_succ['mean_abs'])
                        top5_indices = np.argsort(mean_torques)[-5:][::-1]
                        joint_names = per_joint_torque_succ.get('joint_names', [f'Joint_{i}' for i in range(len(mean_torques))])
                        print("\nTop 5 joints by average torque (successful episodes):")
                        for idx in top5_indices:
                            print(f"  {joint_names[idx]}: {mean_torques[idx]:.2f} ± {per_joint_torque_succ['std'][idx]:.2f}")

                    # print(1 - self.terminate_state.sum() / self.terminate_state.shape[0])
                    print(self.config['network_path'])

                    # Save evaluation results to output directory
                    import os
                    import json
                    from datetime import datetime

                    # Get experiment name from config
                    exp_name = None
                    if 'exp_name' in self.config:
                        exp_name = self.config['exp_name']
                    elif hasattr(self.env, 'task') and hasattr(self.env.task, 'cfg') and 'exp_name' in self.env.task.cfg:
                        exp_name = self.env.task.cfg['exp_name']
                    else:
                        # Try to extract from network path
                        network_path = self.config.get('network_path', '')
                        if network_path:
                            exp_name = os.path.basename(os.path.dirname(network_path))

                    if exp_name:
                        output_dir = f"output/HumanoidIm/{exp_name}"
                    else:
                        output_dir = "output/HumanoidIm/default_exp"

                    # Add eval_subdir if specified
                    eval_subdir = None
                    if hasattr(self.env, 'task') and hasattr(self.env.task, 'cfg'):
                        eval_subdir = self.env.task.cfg.get('eval_subdir', '')
                    if eval_subdir:
                        output_dir = os.path.join(output_dir, "evaluation", eval_subdir)

                    os.makedirs(output_dir, exist_ok=True)

                    # Save evaluation configuration
                    if eval_subdir and hasattr(self.env, 'task') and hasattr(self.env.task, 'cfg'):
                        task_cfg = self.env.task.cfg
                        env_cfg = task_cfg.get('env', {})
                        eval_config = {
                            "eval_subdir": eval_subdir,
                            "recipient_mass_scale": env_cfg.get('recipient_mass_scale', 0.7),
                            "recipient_weakness_scale": env_cfg.get('recipient_weakness_scale', 1.0),
                            "kp_scale": env_cfg.get('kp_scale', 1.0),
                            "kd_scale": env_cfg.get('kd_scale', 1.0),
                            "num_envs": env_cfg.get('num_envs', 1024),
                            "interx_data_path": env_cfg.get('interx_data_path', ''),
                            "exp_name": exp_name,
                            "timestamp": datetime.now().isoformat()
                        }
                        eval_config_path = os.path.join(output_dir, "eval_config.yaml")
                        with open(eval_config_path, 'w') as f:
                            yaml.dump(eval_config, f, default_flow_style=False)

                    # Create comprehensive results dictionary
                    results = {
                        "timestamp": datetime.now().isoformat(),
                        "success_rate": float(self.success_rate),
                        "metrics_all": {k: float(v) for k, v in metrics_all_print.items()},
                        "metrics_succ": {k: float(v) for k, v in metrics_print.items()},
                        "metrics_recipient_all": {k: float(v) for k, v in metrics_recipient_print.items()},
                        "metrics_recipient_succ": {k: float(v) for k, v in metrics_recipient_succ_print.items()},
                        "avg_recipient_torque_all": float(avg_recipient_torque_all),
                        "avg_recipient_torque_succ": float(avg_recipient_torque_succ),
                        "per_joint_torque_all": per_joint_torque_all,
                        "per_joint_torque_succ": per_joint_torque_succ,
                        "eval_config": {
                            "eval_subdir": eval_subdir or "default",
                            "recipient_mass_scale": self.env.task.recipient_mass_scale if hasattr(self.env.task, 'recipient_mass_scale') else 0.7,
                            "recipient_weakness_scale": self.env.task.recipient_weakness_scale if hasattr(self.env.task, 'recipient_weakness_scale') else 1.0,
                            "kp_scale": self.env.task._kp_scale if hasattr(self.env.task, '_kp_scale') else 1.0,
                            "kd_scale": self.env.task._kd_scale if hasattr(self.env.task, '_kd_scale') else 1.0,
                        },
                        "failed_keys_count": len(failed_keys),
                        "failed_keys": failed_keys.tolist() if hasattr(failed_keys, 'tolist') else list(failed_keys),
                        "network_path": self.config.get('network_path', '')
                    }

                    # Add individual recipient metrics with _recipient suffix for easier access
                    for metric_name, metric_value in metrics_recipient_print.items():
                        results[f"{metric_name}_recipient"] = float(metric_value)
                    for metric_name, metric_value in metrics_recipient_succ_print.items():
                        results[f"{metric_name}_recipient_succ"] = float(metric_value)
                    
                    # Save as JSON
                    json_path = os.path.join(output_dir, "evaluation_results.json")
                    with open(json_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    
                    # Save as text (human readable)
                    txt_path = os.path.join(output_dir, "evaluation_results.txt")
                    with open(txt_path, 'w') as f:
                        f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(f"Success Rate: {self.success_rate:.10f}\n")
                        f.write(f"All:  " + " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]) + f"\tavg_torque: {avg_recipient_torque_all:.3f}\n")
                        f.write(f"Succ: " + " \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]) + f"\tavg_torque: {avg_recipient_torque_succ:.3f}\n")
                        if metrics_recipient_print:
                            f.write(f"Recipient All:  " + " \t".join([f"{k}: {v:.3f}" for k, v in metrics_recipient_print.items()]) + "\n")
                        if metrics_recipient_succ_print:
                            f.write(f"Recipient Succ: " + " \t".join([f"{k}: {v:.3f}" for k, v in metrics_recipient_succ_print.items()]) + "\n")
                        f.write(f"Failed Keys Count: {len(failed_keys)}\n")
                        f.write(f"Network Path: {self.config.get('network_path', '')}\n")
                        f.write(f"Output Directory: {output_dir}\n")
                    
                    print(f"Results saved to: {output_dir}")
                    print(f"  - JSON: {json_path}")
                    print(f"  - TXT:  {txt_path}")
                    if COLLECT_Z:
                        zs_all = self.zs_all[:humanoid_env._motion_lib._num_unique_motions]
                        zs_dump = {k: zs_all[idx].cpu().numpy() for idx, k in enumerate(humanoid_env._motion_lib._motion_data_keys)}
                        joblib.dump(zs_dump, osp.join(self.config['network_path'], "zs_run.pkl"))
                    
                    # import ipdb; ipdb.set_trace()  # Disabled for normal test execution

                    # joblib.dump(np.concatenate(self.zs_all[: humanoid_env._motion_lib._num_unique_motions]), osp.join(self.config['network_path'], "zs.pkl"))

                    joblib.dump(failed_keys, osp.join(self.config['network_path'], "failed.pkl"))
                    joblib.dump(success_keys, osp.join(self.config['network_path'], "long_succ.pkl"))
                    print("....")

                    # Exit after all motions have been evaluated once
                    print("Evaluation complete. Exiting.")
                    sys.exit(0)

                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

                humanoid_env.forward_motion_samples()
                self.terminate_state = torch.zeros(
                    self.env.task.num_envs, device=self.device
                )

                self.pbar.update(1)
                self.pbar.refresh()
                self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                if COLLECT_Z: self.zs = []
                self.curr_stpes = 0


            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)

        return done
    
    def get_z(self, obs_dict):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            z = self.model.a2c_network.eval_z(input_dict)
            return z

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)


                    if COLLECT_Z: z = self.get_z(obs_dict)
                        

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    if COLLECT_Z: info['z'] = z
                    done = self._post_step(info, done.clone())

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)
                        
                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = (
                                    s[:, all_done_indices, :] * 0.0
                                )

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count, "w:", game_res,)
                            else:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count,)

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

        return
