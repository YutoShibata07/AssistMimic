import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgym.torch_utils import quat_rotate_inverse, quat_conjugate, quat_mul, quat_apply, to_torch
import joblib
from phc.utils.flags import flags
from phc.utils import torch_utils
from phc.utils.torch_utils import quat_to_tan_norm
import random
from gym import spaces
import math

# Import the parent class
from phc.env.tasks.humanoid_im import HumanoidIm, compute_point_goal_reward

# Import RSI module
from phc.env.tasks.RSI import RSIMixin

# Import for motion library configuration
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict

# Import common utilities
from phc.env.tasks.commons import (
    normalize_quaternion_batch,
    get_partner_env_ids,
    get_motion_filename_from_data
)


class HumanoidImInterxHelpUp(HumanoidIm, RSIMixin):
    """
    Humanoid Interaction class for single-humanoid per environment interaction tasks.
    Following the instruction: 1 environment = 1 humanoid, with collision groups for pair interaction.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Store interaction-specific config
        self.interx_data_path = cfg["env"].get("interx_data_path", 
                                               "../PHC/sample_data/interx_processed_fixed_v4.pkl")
        print(f"interx_data_path: {self.interx_data_path}")
        self.humanoid_number = 2  # As per instruction: pair them with collision groups
        
        
        # SimpleLiftUp mode settings
        self.simple_lift_up_mode = cfg["env"].get("simple_lift_up_mode", False)
        self.task_reward_only = cfg["env"].get("task_reward_only", False)
        self.dense_height_reward = cfg["env"].get("dense_height_reward", False)
        self.recipient_mass_scale = cfg["env"].get("recipient_mass_scale", 0.7)

        # Recipient kinematic replay mode
        self.recipient_kinematic_replay = cfg["env"].get("recipient_kinematic_replay", False)

        # Failed motion weighted sampling settings
        self.failed_motion_weight = cfg["env"].get("failed_motion_weight", False)
        self.failed_weight_multiplier = cfg["env"].get("failed_weight_multiplier", 2.0)

        # Power coefficient for caregiver and recipient
        self.caregiver_power_coefficient = cfg["env"].get("caregiver_power_coefficient", 0.001)
        self.recipient_power_coefficient = cfg["env"].get("recipient_power_coefficient", 0.05)

        # Caregiver touch head reward settings
        self.caregiver_touch_head_reward = cfg["env"].get("caregiver_touch_head_reward", False)
        self.caregiver_touch_head_coefficient = cfg["env"].get("caregiver_touch_head_coefficient", 0.001)
        self.caregiver_touch_threshold = cfg["env"].get("caregiver_touch_threshold", 0.1)

        # Caregiver recipient torque reduction reward settings
        self.caregiver_torque_reduction_reward = cfg["env"].get("caregiver_torque_reduction_reward", True)
        self.caregiver_torque_reduction_coefficient = cfg["env"].get("caregiver_torque_reduction_coefficient", 0.5)
        self.caregiver_torque_reduction_scale = cfg["env"].get("caregiver_torque_reduction_scale", 150.0)  # Scale for exponential decay (typical torque: 100-200)
        self.future_size = 1

        print(f"SimpleLiftUp mode: {self.simple_lift_up_mode}")
        if self.simple_lift_up_mode:
            print(f"  - Task reward only: {self.task_reward_only}")
            print(f"  - Dense height reward: {self.dense_height_reward}")
            print(f"  - Recipient mass scale: {self.recipient_mass_scale}")

        print(f"Failed motion weighted sampling: {self.failed_motion_weight}")
        if self.failed_motion_weight:
            print(f"  - Failed weight multiplier: {self.failed_weight_multiplier}")

        # Recipient assist force settings (PD control)
        self.recipient_assist_enabled = cfg["env"].get("recipient_assist_enabled", False)
        self.caregiver_assist_enabled = cfg["env"].get("caregiver_assist_enabled", False)
        self.recipient_assist_kp_pos = cfg["env"].get("recipient_assist_kp_pos", 1000.0)
        self.recipient_assist_kd_pos = cfg["env"].get("recipient_assist_kd_pos", 100.0)

        # Curriculum learning for recipient assist
        self.recipient_assist_kp_pos_initial = self.recipient_assist_kp_pos  # Store initial Kp value
        self.recipient_assist_kd_pos_initial = self.recipient_assist_kd_pos  # Store initial Kd value
        self.curriculum_start_epoch = cfg["env"].get("curriculum_start_epoch", 200)
        self.curriculum_duration_epochs = cfg["env"].get("curriculum_duration_epochs", 1000)
        self.curriculum_kp_pos_final = cfg["env"].get("curriculum_kp_pos_final", 0.0)
        self.curriculum_kd_pos_final = cfg["env"].get("curriculum_kd_pos_final", 0.0)

        print(f"Recipient assist: {self.recipient_assist_enabled}")
        if self.recipient_assist_enabled:
            print(f"  - Kp_pos: {self.recipient_assist_kp_pos}, Kd_pos: {self.recipient_assist_kd_pos}")
            print(f"  - Curriculum learning: start_epoch={self.curriculum_start_epoch}, duration={self.curriculum_duration_epochs}")
            print(f"    - Kp: {self.recipient_assist_kp_pos_initial} -> {self.curriculum_kp_pos_final}")
            print(f"    - Kd: {self.recipient_assist_kd_pos_initial} -> {self.curriculum_kd_pos_final}")

        # Add properties for automatic PMCP detection (needed by learning agent)
        self.auto_pmcp = cfg["env"].get("auto_pmcp", False)
        self.auto_pmcp_soft = cfg["env"].get("auto_pmcp_soft", False)
        
        # Initialize basic assignments early to avoid initialization issues
        self.env_motion_assignments = {}
        self.env_role_assignments = {}
        
        # Evaluation mode flag for controlling trajectory buffer usage
        self.evaluation_mode = False
        
        # Reference to training agent for accessing epoch_num
        self._training_agent = None
        self._save_frequency = cfg["env"].get("save_frequency", 50)  # Default from config
        
        # Do not force numObservations here; it will be derived dynamically by parent init via get_obs_size()
        
        # Initialize parent class
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, 
                        device_type=device_type, device_id=device_id, headless=headless)
        
        # Setup body name to ID mapping for contact processing
        self._setup_body_name_to_id_mapping()
        
        # Update the observation buffer size after initialization
        self._update_obs_buf_size()
        
        # SimpleLiftUp tracking (after parent initialization when device is available)
        if self.simple_lift_up_mode:
            self.recipient_max_heights = torch.zeros(self.num_envs, device=self.device)
            self.episode_max_recipient_heights = torch.zeros(self.num_envs, device=self.device)
            # Statistics for wandb logging (similar to episode_lengths)
            self.episode_max_heights_buffer = []  # Buffer to store completed episode max heights
            self._latest_episode_max_heights_mean = 0.0  # Latest mean for logging
            
            # Initialize Reference State Initialization
            self._init_rsi(cfg)
        
        # Hand contact tracking for reward validation
        self.episode_hand_contact_count = torch.zeros(self.num_envs, device=self.device)
    
        # Initialize camera after parent initialization
        if self.viewer != None or flags.server_mode:
            self._init_camera()
        
        # Hand distance reward configuration
        self.hand_target_flg = self.cfg["env"].get("hand_target_flg", False)  # False: nearest upper body joint, True: hand-to-hand
        
        # Static termination tracking (30 frames with joint movement < 1cm)
        self.static_frames_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.prev_joint_positions = None  # Will be initialized in first step
        self.STATIC_FRAME_THRESHOLD = 30  # 30 frames
        self.JOINT_MOVEMENT_THRESHOLD = self.cfg["env"].get("joint_movement_threshold", 0.01)  # default 1cm
        
        # Initialize recipient assist force tensors
        if self.recipient_assist_enabled:
            self.recipient_assist_rb_forces = torch.zeros((self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)
        
        # Initialize torque tracking for recipient joints
        self.recipient_torque_buffer = []  # List to store torque data for each step
        self.current_motion_filename = None  # Current motion filename for saving
        if self.caregiver_assist_enabled:
            self.caregiver_assist_rb_forces = torch.zeros((self._rigid_body_state.shape[0], 3), device=self.device, dtype=torch.float)

        # Previous actions for aux_features (initialized after parent class sets num_actions)
        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # Kinematic replay: cache recipient env/actor IDs
        if self.recipient_kinematic_replay:
            self._kin_replay_recipient_env_ids = torch.arange(self.num_envs, device=self.device)[
                torch.arange(self.num_envs, device=self.device) % 2 == 1
            ]
            self._kin_replay_recipient_actor_ids = self._humanoid_actor_ids[self._kin_replay_recipient_env_ids]
            print("[KinReplay] Recipient kinematic replay mode ENABLED")

        return
    
    def _physics_step(self):
        if not self.recipient_kinematic_replay:
            return super()._physics_step()

        for i in range(self.control_freq_inv):
            self.control_i = i
            self.render()
            if not self.paused and self.enable_viewer_sync:
                if self.control_mode == "pd":
                    self.torques = self._compute_torques(self.pd_tar)
                    self.gym.set_dof_actuation_force_tensor(
                        self.sim, gymtorch.unwrap_tensor(self.torques))
                    self.gym.simulate(self.sim)
                    if self.device == 'cpu':
                        self.gym.fetch_results(self.sim, True)
                    self.gym.refresh_dof_state_tensor(self.sim)
                else:
                    self.gym.simulate(self.sim)

                # Force-set recipient states after each simulate()
                self._set_recipient_kinematic_state()
        return

    def _set_recipient_kinematic_state(self):
        """Force recipient state to reference motion each physics substep (kinematic replay)."""
        env_ids = self._kin_replay_recipient_env_ids

        # Compute motion time (same as _compute_assist_forces)
        motion_times = (self.progress_buf[env_ids].float() * self.dt
                        + self._motion_start_times[env_ids]
                        + self._motion_start_times_offset[env_ids])

        # Get reference motion state (apply pair offset like _get_state_from_motionlib_cache)
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        offset = self._global_offset[env_ids].clone()
        offset[:, 0] -= pair_offset_val  # All env_ids here are recipients
        motion_res = self._motion_lib.get_motion_state(
            self._sampled_motion_ids[env_ids],
            motion_times,
            offset=offset
        )

        # Overwrite root state
        self._humanoid_root_states[env_ids, 0:3] = motion_res["root_pos"]
        self._humanoid_root_states[env_ids, 3:7] = motion_res["root_rot"]
        self._humanoid_root_states[env_ids, 7:10] = motion_res["root_vel"]
        self._humanoid_root_states[env_ids, 10:13] = motion_res["root_ang_vel"]

        # Overwrite DOF state
        self._dof_pos[env_ids] = motion_res["dof_pos"]
        self._dof_vel[env_ids] = motion_res["dof_vel"]

        # Apply to IsaacGym simulation
        actor_ids = self._kin_replay_recipient_actor_ids
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids)
        )

    def _build_termination_heights(self):
        """Override to set different termination distances for caregiver and recipient"""
        super()._build_termination_heights()

        # Read termination distances from config
        termination_distance_caregiver = self.cfg["env"].get("terminationDistance", 0.5)
        termination_distance_recipient = self.cfg["env"].get("terminationDistanceRecipient", 0.3)

        # Create termination distance array for each environment
        termination_distances = []
        for env_id in range(self.num_envs):
            if env_id % self.humanoid_number == 0:
                # Caregiver (even env_id)
                termination_distances.append(termination_distance_caregiver)
            else:
                # Recipient (odd env_id)
                termination_distances.append(termination_distance_recipient)

        # Create tensor with shape (num_envs, num_bodies)
        termination_distances_array = np.array([termination_distances]).T  # Shape: (num_envs, 1)
        termination_distances_array = np.repeat(termination_distances_array, self.num_bodies, axis=1)  # Shape: (num_envs, num_bodies)
        self._termination_distances = to_torch(termination_distances_array, device=self.device)

        print(f"Set termination distances: caregiver={termination_distance_caregiver}, recipient={termination_distance_recipient}")
        return

    def pre_physics_step(self, actions):
        """Override to compute and apply recipient assist forces before physics step"""
        # Compute and apply recipient assist forces BEFORE parent pre_physics_step
        # testでは適用しない
        if flags.test == False:
            if self.recipient_assist_enabled:
                self._compute_assist_forces(target="recipient")
            if self.caregiver_assist_enabled:
                self._compute_assist_forces(target="caregiver")

            if self.recipient_assist_enabled and self.caregiver_assist_enabled:
                self._apply_assist_forces(target="both")
            elif self.recipient_assist_enabled:
                self._apply_assist_forces(target="recipient")
            elif self.caregiver_assist_enabled:
                self._apply_assist_forces(target="caregiver")

        # Call parent pre_physics_step
        super().pre_physics_step(actions)

        # Store current actions as previous actions for next step's aux_features
        self.prev_actions.copy_(actions)

    def _apply_assist_forces(self, target="recipient"):
        """Apply computed assist forces to the simulation"""
        # Apply rigid body forces (position-based assistance only)
        if target == "both":
            both_forces = self.recipient_assist_rb_forces + self.caregiver_assist_rb_forces
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(both_forces),
                None,  # No torques
                gymapi.ENV_SPACE
            )
        elif target == "recipient":
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.recipient_assist_rb_forces),
                None,  # No torques
                gymapi.ENV_SPACE
            )
        elif target == "caregiver":
            self.gym.apply_rigid_body_force_tensors(
                self.sim,
                gymtorch.unwrap_tensor(self.recipient_assist_rb_forces),
                    None,  # No torques
                    gymapi.ENV_SPACE
                )
        return


    def _load_motion(self, motion_train_file, motion_test_file=[]):
        """Override to setup interaction motion data and use full_fix for height adjustment"""
        # Use the interaction data path if no specific motion file is provided
        if not motion_train_file or motion_train_file == "":
            motion_train_file = self.interx_data_path

        # Override parent's implementation to use full_fix instead of pene_fix
        assert (self._dof_offsets[-1] == self.num_dof)

        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            motion_lib_cfg = EasyDict({
                "motion_file": motion_train_file,
                "device": torch.device("cpu"),
                "fix_height": FixHeightMode.full_fix,  # Changed from pene_fix to full_fix
                "min_length": self._min_motion_len,
                "max_length": -1,
                "im_eval": flags.im_eval,
                "multi_thread": True ,
                "smpl_type": self.humanoid_type,
                "randomrize_heading": True,
                "device": self.device,
            })
            motion_eval_file = motion_train_file
            self._motion_train_lib = MotionLibSMPL(motion_lib_cfg)
            motion_lib_cfg.im_eval = True
            self._motion_eval_lib = MotionLibSMPL(motion_lib_cfg)

            self._motion_lib = self._motion_train_lib
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=(not flags.test) and (not self.seq_motions), max_len=-1 if flags.test else self.max_len)
        else:
            # Fallback to parent implementation for non-SMPL humanoids
            from phc.utils.motion_lib import MotionLib
            self._motion_lib = MotionLib(motion_file=motion_train_file, dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets, device=self.device)

        # Setup interaction motion data after motion has been loaded (with safety checks)
        self._setup_interaction_motion_data()

        return

    def _setup_interaction_motion_data(self):
        """Setup motion data for interaction tasks"""
        print(f"Loading interaction data from: {self.interx_data_path}")
        
        # Safety check for motion data file
        try:
            # Load interaction motion data
            self.interaction_data = joblib.load(self.interx_data_path)
            print(f"Loaded interaction data with {len(self.interaction_data)} keys")
        except Exception as e:
            print(f"Error loading interaction data: {e}")
            print("Using fallback: disabling interaction motion data")
            self.env_motion_assignments = {}
            self.env_role_assignments = {}
            self.interaction_data = {}
            return
        
        # Extract sequence names
        self.interaction_sequences = {}
        sequence_names = set()
        
        for key in self.interaction_data.keys():
            if key.endswith('_caregiver') or key.endswith('_recipient'):
                sequence_name = key.rsplit('_', 1)[0]
                sequence_names.add(sequence_name)
        
        self.interaction_sequences = sorted(list(sequence_names))
        print(f"Found {len(self.interaction_sequences)} interaction sequences")
        
        # Safety check for empty sequences - THIS IS THE CRITICAL FIX
        if len(self.interaction_sequences) == 0:
            print("ERROR: No interaction sequences found! This would cause division by zero.")
            print("Available keys in data:", list(self.interaction_data.keys())[:10])  # Show first 10 keys
            # Use fallback to prevent FPE
            self.env_motion_assignments = {}
            self.env_role_assignments = {}
            return
        
        # Create motion assignments for environments
        # Following instruction: env_id even = caregiver, env_id odd = recipient
        self.env_motion_assignments = {}
        self.env_role_assignments = {}
        
        for env_id in range(self.num_envs):
            pair_id = env_id // self.humanoid_number
            sequence_id = pair_id % len(self.interaction_sequences)  # Now safe from division by zero
            sequence_name = self.interaction_sequences[sequence_id]
            
            # Assign role based on env_id: even = caregiver, odd = recipient
            if env_id % self.humanoid_number == 0:
                role = "caregiver"
            else:
                role = "recipient"
            
            motion_key = f"{sequence_name}_{role}"
            self.env_motion_assignments[env_id] = motion_key
            self.env_role_assignments[env_id] = role
            if env_id < 10:
                print(f"Assigned {motion_key} to env {env_id} (role: {role})")
        
        print("Created motion assignments for {} environments".format(self.num_envs))
        
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Override to create single-humanoid environments with collision groups for interaction"""
        
        # Call parent method to create basic environments
        super()._create_envs(num_envs, spacing, num_per_row)
        
        # Apply recipient weakness modification after all environments are created
        self._apply_recipient_weakness()
        # self._setup_collision_groups_and_positioning()
        
        return
    
    

    def get_num_actors_per_env(self):
        # Calculate dynamically from actual tensor size like parent class
        if hasattr(self, '_root_states') and self._root_states is not None:
            num_actors = self._root_states.shape[0] // self.num_envs
            return num_actors
        else:
            # Fallback during initialization before tensors are setup
            return super().get_num_actors_per_env()
        
    def get_virtual_bodies_per_env(self):
        num_actors = self.get_num_actors_per_env()
        if num_actors == 1:
            return 52
        elif num_actors == 2: # humanoid root (1) + bed (1)
            return 53
        elif num_actors == 53:  # humanoid root (1) + markers (52)
            return 104
        elif num_actors == 54:  # humanoid root (1) + bed (1) + markers (52)
            return 105 # 52 markers + 1 bed + 1 humanoid root
    
    def _set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rigid_body_pos=None,
        rigid_body_rot=None,
        rigid_body_vel=None,
        rigid_body_ang_vel=None,
    ):
        """Override to apply pair positioning for interaction"""
        # Apply pair positioning adjustment for interaction
        adjusted_root_pos = root_pos.clone()  # Need clone since we modify positions below
        adjusted_rigid_body_pos = rigid_body_pos.clone() if rigid_body_pos is not None else None  # Need clone since we modify positions below
        
        for i, env_id in enumerate(env_ids):
            env_id_val = env_id.item() if torch.is_tensor(env_id) else env_id
            
            motion_key = self.env_motion_assignments.get(env_id_val, "unknown")
            # print(f"_set_env_state: env_id={env_id_val}, motion_key={motion_key}, original_pos={root_pos[i]}")
            
            # If this is an odd env_id (recipient), adjust position to match pair
            if env_id_val % 2 == 1:
                # Get the base position from motion data and adjust it to pair position
                # The pair should be positioned close together for interaction
                spacing_val = self.cfg["env"].get('env_spacing', 5.0)
                if spacing_val == 0:  # Prevent potential issues with zero spacing
                    spacing_val = 5.0
                offset = spacing_val * 2
                pair_offset = torch.tensor([0, 0.0, 0.0], device=self.device, dtype=root_pos.dtype)
                adjusted_root_pos[i] = root_pos[i] + pair_offset
                
                # Also adjust rigid body positions if provided
                if adjusted_rigid_body_pos is not None:
                    adjusted_rigid_body_pos[i] = rigid_body_pos[i] + pair_offset.unsqueeze(0)
                
                # print(f"  -> Adjusted recipient pos: {adjusted_root_pos[i]}")
                # self._marker_pos[env_id_val] += pair_offset.unsqueeze(0)
            else:
                # Even env_id (caregiver) - use motion position as is
                adjusted_root_pos[i] = root_pos[i]
                # print(f"  -> Caregiver pos: {adjusted_root_pos[i]}")
        
        # Call parent with adjusted positions
        super()._set_env_state(
            env_ids=env_ids,
            root_pos=adjusted_root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            rigid_body_pos=adjusted_rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
        )


        
        return

    def _update_marker(self):
        """Override to apply pair positioning for markers"""
        # Call parent method first
        super()._update_marker()
        
        # Apply the same offset to recipient markers
        if hasattr(self, '_marker_pos') and hasattr(self, 'cfg'):
            offset = self.cfg["env"].get('env_spacing', 5.0) * 2
            pair_offset = torch.tensor([-offset, 0.0, 0.0], device=self.device, dtype=self._marker_pos.dtype)
            
            # Apply offset to recipient environments (odd env_ids)
            recipient_envs = [env_id for env_id in range(self.num_envs) if env_id % 2 == 1]
            self._marker_pos[recipient_envs] += pair_offset.unsqueeze(0)
        
        return


    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        """Override to apply position offsets to reference motion for recipients"""
        # Create offset array if not provided
        if offset is None:
            offset = torch.zeros((len(motion_ids), 3), device=self.device, dtype=torch.float32)
        else:
            offset = offset.clone()
        
        # Apply position offset to recipients (odd env_ids)
        # Since motion_ids map directly to env_ids, we can check motion_ids directly
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        
        for i, motion_id in enumerate(motion_ids):
            # motion_id corresponds directly to env_id
            env_id = motion_id.item() if torch.is_tensor(motion_id) else motion_id
            if env_id % 2 == 1:  # Recipient (odd env_id)
                offset[i, 0] -= pair_offset_val  # Move in negative x direction
        
        # Call parent method with adjusted offset
        return super()._get_state_from_motionlib_cache(motion_ids, motion_times, offset=offset)
    
    def _compute_reset(self):
        """Override to implement pair-based termination logic"""
        # Call parent method to compute individual resets
        super()._compute_reset()
        
        # Apply pair-based termination: if either member of a pair fails, both should reset
        # Safety check to prevent division by zero
        if self.num_envs < 2:
            return  # Not enough environments for pairs
        
        for pair_id in range(self.num_envs // 2):
            caregiver_env = pair_id * 2
            recipient_env = pair_id * 2 + 1
            
            # If either environment in the pair needs reset, reset both
            if caregiver_env < self.num_envs and recipient_env < self.num_envs:
                # if flags.test:
                #     pair_reset = self.reset_buf[recipient_env]
                #     pair_terminate = self._terminate_buf[recipient_env]
                # else:
                #     pair_reset = self.reset_buf[caregiver_env] or self.reset_buf[recipient_env]
                #     pair_terminate = self._terminate_buf[recipient_env] or self._terminate_buf[caregiver_env] 
                pair_reset = self.reset_buf[caregiver_env] or self.reset_buf[recipient_env]
                pair_terminate = self._terminate_buf[recipient_env] or self._terminate_buf[caregiver_env] 
                
                # Apply pair-based reset to both environments
                self.reset_buf[caregiver_env] = pair_reset
                self.reset_buf[recipient_env] = pair_reset
                self._terminate_buf[caregiver_env] = pair_terminate
                self._terminate_buf[recipient_env] = pair_terminate
                
                # if pair_reset:
                #     print(f"Pair reset triggered: env {caregiver_env} (caregiver) and env {recipient_env} (recipient)")
        
        return
    
    def get_motion_file(self):
        """Override to use interaction motion file"""
        return self.interx_data_path

    

    def _compute_observations(self, env_ids=None):
        """Override to add partner relative position/velocity to observations"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        # Get the base self observations (proprioception)
        self_obs = self._compute_humanoid_obs(env_ids)
        
        # Check if self_obs_buf size needs to be updated
        if self.self_obs_buf.shape[1] != self_obs.shape[1]:
            print(f"Resizing self_obs_buf from {self.self_obs_buf.shape} to match new self observation size {self_obs.shape[1]}")
            self.self_obs_buf = torch.zeros((self.num_envs, self_obs.shape[1]), 
                                          device=self.device, dtype=self_obs.dtype)
        
        self.self_obs_buf[env_ids] = self_obs

        # Get task observations (reference differences) with role-specific handling
        if self._enable_task_obs:
            task_obs = self._compute_task_obs_with_role_handling(env_ids)
        else:
            task_obs = torch.zeros(len(env_ids), 0, device=self.device)

        # Compute partner relative position/velocity observations with safety
        try:
            partner_obs = self._compute_partner_obs(env_ids)
            # Verify the result is valid
            if torch.isnan(partner_obs).any() or torch.isinf(partner_obs).any():
                print("Warning: NaN/Inf in partner_obs, using zeros")
                partner_obs = torch.zeros(len(env_ids), self._get_partner_obs_size(), device=self.device)
        except Exception as e:
            msg = f"Error computing partner obs: {e}"
            raise Exception(msg)
        
        # Combine all observations in the correct order for the network
        # Network expects: [base_self_obs, task_obs, aux_features]
        # aux_features = [self_contact, partner_obs, partner_contact]
        
        # Extract base self observation (without contact info)
        if self.cfg["env"].get("self_obs_v", 1) == 4:
            # self_obs contains [base_proprioception, self_contact]
            contact_size = self._get_hand_contact_obs_size()
            base_self_obs = self_obs[:, :-contact_size]  # Remove contact info
            self_contact_obs = self_obs[:, -contact_size:]  # Extract contact info
        elif self.cfg["env"].get("self_obs_v", 1) == 5:
            # self_obs contains [base_proprioception, self_contact, self_force]
            contact_size = self._get_hand_contact_obs_size()
            force_size = self._get_hand_force_obs_size()
            total_aux_size = contact_size + force_size
            base_self_obs = self_obs[:, :-total_aux_size]  # Remove contact and force info
            self_contact_obs = self_obs[:, -total_aux_size:-force_size]  # Extract contact info
            self_force_obs = self_obs[:, -force_size:]  # Extract force info
        else:
            base_self_obs = self_obs
            self_contact_obs = torch.zeros(len(env_ids), 0, device=self.device)
            self_force_obs = torch.zeros(len(env_ids), 0, device=self.device)
        
        # Extract partner base and contact observations
        if self.cfg["env"].get("partner_obs_v", 1) == 3:
            # partner_obs contains [base_partner_obs, partner_contact]
            partner_contact_size = self._get_hand_contact_obs_size()
            base_partner_obs = partner_obs[:, :-partner_contact_size]
            partner_contact_obs = partner_obs[:, -partner_contact_size:]
        else:
            base_partner_obs = partner_obs
            partner_contact_obs = torch.zeros(len(env_ids), 0, device=self.device)
        
        # MultiPulse compatibility: NO reference_contact_diff, NO role_flags
        # aux_features = [self_contact, base_partner_obs, partner_contact] only
        
        # Extract future trajectory features and normalize them relative to current frame
        # Extract self future trajectories (30 frames - 1 second)
        self_future_trajectories = self._extract_future_trajectories(env_ids, future_frames=self.future_size)

        # Extract partner future trajectories (30 frames - 1 second)
        partner_future_trajectories = self._extract_partner_future_trajectories(env_ids, future_frames=self.future_size)

        # Normalize trajectories relative to current self root position and orientation
        self_future_trajectories_normalized = self._normalize_trajectories_to_current_frame(
            self_future_trajectories, env_ids, target_env_ids=env_ids)
        partner_future_trajectories_normalized = self._normalize_trajectories_to_current_frame(
            partner_future_trajectories, env_ids, target_env_ids=env_ids)

        # Flatten: [batch, 30, feature_dim] -> [batch, 30*feature_dim]
        self_future_trajectory_flat = self_future_trajectories_normalized.view(len(env_ids), -1)
        partner_future_trajectory_flat = partner_future_trajectories_normalized.view(len(env_ids), -1)
        # Add role label (env_id % 2) for recipient/caregiver determination
        role_labels = (env_ids % 2).float().unsqueeze(1)  # 0 for even (recipient), 1 for odd (caregiver)

        # Get previous actions for aux_features
        prev_actions_batch = self.prev_actions[env_ids]

        # Create aux_features list with future trajectories and previous actions (role_label at the end)
        aux_features_parts = [self_contact_obs]
        if self.cfg["env"].get("self_obs_v", 1) == 5:
            aux_features_parts.append(self_force_obs)
        aux_features_parts.extend([base_partner_obs, partner_contact_obs, self_future_trajectory_flat, partner_future_trajectory_flat, prev_actions_batch, role_labels])
    

        if self.simple_lift_up_mode:
            # SimpleLiftUp mode: concatenate aux_features to base_self_obs for MLP processing
            aux_features = torch.cat(aux_features_parts, dim=-1)
            obs = torch.cat([base_self_obs, task_obs, aux_features], dim=-1)

        else:
            # Normal interaction mode: combine aux_features (MultiPulse compatible)
            aux_features = torch.cat(aux_features_parts, dim=-1)
            # Final observation order: [base_self_obs, task_obs, aux_features]
            obs = torch.cat([base_self_obs, task_obs, aux_features], dim=-1)
        
        if self.add_obs_noise and not flags.test:
            obs = obs + torch.randn_like(obs) * 0.1
            
        if self.obs_v == 4:
            # Double sub will return a copy.
            B, N = obs.shape
            
            # Check if obs_buf needs to be resized due to partner observations
            if self.obs_buf.shape[1] != N * self.past_track_steps:
                print(f"Resizing obs_buf from {self.obs_buf.shape} to match new observation size {N}")
                # Reinitialize obs_buf with correct size
                self.obs_buf = torch.zeros((self.num_envs, N * self.past_track_steps), 
                                         device=self.device, dtype=obs.dtype)
            
            sums = self.obs_buf[env_ids, 0:self.past_track_steps].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            
            if zeros.any():
                obs_slice[zeros] = torch.tile(obs[zeros], (1, self.past_track_steps))
            if nonzero.any() and obs_slice.shape[1] >= N:
                obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
            
            self.obs_buf[env_ids] = obs_slice
        else:
            # Check if obs_buf size needs to be updated
            if self.obs_buf.shape[1] != obs.shape[1]:
                print(f"Resizing obs_buf from {self.obs_buf.shape} to match new observation size {obs.shape[1]}")
                self.obs_buf = torch.zeros((self.num_envs, obs.shape[1]), 
                                         device=self.device, dtype=obs.dtype)
            self.obs_buf[env_ids] = obs
        
        # Detach ALL recipient observations to prevent gradient flow (Option 1 implementation)
        # for i, env_id in enumerate(env_ids):
        #     if env_id % 2 == 1:  # Recipient environments (odd env_ids)
        #         obs[i] = obs[i].detach()

        return obs

    def _compute_partner_obs(self, env_ids):
        """Compute partner position and velocity observations"""
        # Get partner observation version from config
        partner_obs_v = self.cfg["env"].get("partner_obs_v", 1)
        
        if partner_obs_v == 3:
            return self._compute_partner_obs_v3(env_ids)
        else:
            return self._compute_partner_obs_v1_v2(env_ids)

    def _compute_partner_obs_v1_v2(self, env_ids):
        """Compute partner observations v1 and v2 (vectorized implementation for speed)"""
        # Get partner observation version from config
        partner_obs_v = self.cfg["env"].get("partner_obs_v", 1)

        # Convert env_ids to tensor if needed
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device)

        batch_size = len(env_ids)

        # Vectorized partner ID computation
        partner_env_ids = get_partner_env_ids(env_ids)

        # Ensure all partner IDs are valid
        valid_mask = partner_env_ids < self.num_envs
        if not valid_mask.all():
            invalid_partners = partner_env_ids[~valid_mask]
            raise Exception(f"Partner env_ids {invalid_partners.tolist()} out of bounds")

        # Batch fetch all current and partner states
        current_root_pos = self._rigid_body_pos[env_ids, 0, :3].clone()  # [batch_size, 3]
        current_root_rot = self._rigid_body_rot[env_ids, 0, :]    # [batch_size, 4]
        current_body_pos = self._rigid_body_pos[env_ids, :, :3].clone()  # [batch_size, num_bodies, 3]
        current_body_vel = self._rigid_body_vel[env_ids, :, :3]  # [batch_size, num_bodies, 3]

        partner_root_pos = self._rigid_body_pos[partner_env_ids, 0, :3].clone()  # [batch_size, 3]
        partner_body_pos = self._rigid_body_pos[partner_env_ids, :, :3].clone()  # [batch_size, num_bodies, 3]
        partner_body_vel = self._rigid_body_vel[partner_env_ids, :, :3]  # [batch_size, num_bodies, 3]
        partner_body_rot = self._rigid_body_rot[partner_env_ids, :]  # [batch_size, num_bodies, 4]

        # Vectorized motion cache offset compensation
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        motion_offset = torch.tensor([pair_offset_val, 0.0, 0.0], device=self.device, dtype=partner_root_pos.dtype)

        # Apply offset to partner states where partner is recipient (partner_env_id % 2 == 1)
        partner_recipient_mask = (partner_env_ids % 2 == 1)
        if partner_recipient_mask.any():
            partner_root_pos[partner_recipient_mask] += motion_offset
            partner_body_pos[partner_recipient_mask] += motion_offset.unsqueeze(0)

        # Apply offset to current states where current is recipient (env_id % 2 == 1)
        current_recipient_mask = (env_ids % 2 == 1)
        if current_recipient_mask.any():
            current_root_pos[current_recipient_mask] += motion_offset
            current_body_pos[current_recipient_mask] += motion_offset.unsqueeze(0)

        # Vectorized quaternion normalization
        quat_norms = torch.norm(current_root_rot, dim=1, keepdim=True)
        zero_mask = quat_norms.squeeze(1) < 1e-6
        current_root_rot[zero_mask] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        current_root_rot[~zero_mask] = current_root_rot[~zero_mask] / torch.clamp(quat_norms[~zero_mask], min=1e-8)

        if partner_obs_v == 1:
            not_implemented_error = "Partner obs v1 is not implemented"
            raise Exception(not_implemented_error)

        elif (partner_obs_v == 2) or (partner_obs_v == 3):
            # V2: Absolute partner positions/velocities/orientations in self coordinate frame (vectorized)
            # 1. All partner joint positions in current humanoid's frame
            partner_pos_relative = partner_body_pos - current_root_pos.unsqueeze(1)  # [batch_size, num_bodies, 3]
            batch_size, num_bodies, _ = partner_pos_relative.shape

            # Expand quaternions for batch processing
            current_root_rot_expanded = current_root_rot.unsqueeze(1).expand(-1, num_bodies, -1)  # [batch_size, num_bodies, 4]
            partner_pos_relative_flat = partner_pos_relative.reshape(-1, 3)  # [batch_size * num_bodies, 3]
            current_root_rot_flat = current_root_rot_expanded.reshape(-1, 4)  # [batch_size * num_bodies, 4]

            partner_pos_local_flat = quat_rotate_inverse(current_root_rot_flat, partner_pos_relative_flat)
            partner_pos_local = partner_pos_local_flat.reshape(batch_size, -1)  # [batch_size, num_bodies * 3]

            # 2. All partner joint velocities in current humanoid's frame
            partner_body_vel_flat = partner_body_vel.reshape(-1, 3)  # [batch_size * num_bodies, 3]
            partner_vel_local_flat = quat_rotate_inverse(current_root_rot_flat, partner_body_vel_flat)
            partner_vel_local = partner_vel_local_flat.reshape(batch_size, -1)  # [batch_size, num_bodies * 3]

            # 3. All partner joint orientations in current humanoid's frame (6D representation)
            # Transform partner body rotations to current humanoid's frame
            partner_body_rot_flat = partner_body_rot.reshape(-1, 4)  # [batch_size * num_bodies, 4]
            # Compute relative rotation: q_rel = q_current_inv * q_partner
            partner_rot_local_flat = quat_mul(quat_conjugate(current_root_rot_flat), partner_body_rot_flat)  # [batch_size * num_bodies, 4]

            # Convert quaternions to 6D representation (first two columns of rotation matrix)
            partner_rot_local_6d_flat = quat_to_tan_norm(partner_rot_local_flat)  # [batch_size * num_bodies, 6]
            partner_rot_local_6d = partner_rot_local_6d_flat.reshape(batch_size, -1)  # [batch_size, num_bodies * 6]

            # 4. Partner root orientation with respect to gravity (heading-invariant, for fall detection)
            # Get partner root rotation (index 0) and remove heading component
            partner_root_rot = partner_body_rot[:, 0, :]  # [batch_size, 4]
            partner_root_heading_rot_inv = torch_utils.calc_heading_quat_inv(partner_root_rot)  # [batch_size, 4]
            partner_root_rot_gravity_frame = quat_mul(partner_root_heading_rot_inv, partner_root_rot)  # [batch_size, 4]
            partner_root_rot_gravity_6d = quat_to_tan_norm(partner_root_rot_gravity_frame)  # [batch_size, 6]

            # Combine partner observations
            partner_obs = torch.cat([
                partner_pos_local,           # [batch_size, num_bodies * 3]
                partner_vel_local,           # [batch_size, num_bodies * 3]
                partner_rot_local_6d,        # [batch_size, num_bodies * 6]
                partner_root_rot_gravity_6d  # [batch_size, 6] - for fall detection
            ], dim=1)
        else:
            error_msg = f"Invalid partner_obs_v: {partner_obs_v}"
            raise Exception(error_msg)

        return partner_obs

    def _update_obs_buf_size(self):
        """Update observation buffer size after initialization"""
        # Calculate the correct observation sizes
        correct_obs_size = self.get_obs_size()
        correct_self_obs_size = self.get_self_obs_size()
        
        print(f"[DEBUG] Observation size calculation:")
        print(f"[DEBUG] - correct_obs_size: {correct_obs_size}")
        print(f"[DEBUG] - correct_self_obs_size: {correct_self_obs_size}")
        print(f"[DEBUG] - current obs_buf shape: {self.obs_buf.shape}")
        print(f"[DEBUG] - current self_obs_buf shape: {self.self_obs_buf.shape}")
        
        # Update observation buffer if size doesn't match
        if self.obs_buf.shape[1] != correct_obs_size:
            print(f"Resizing obs_buf from {self.obs_buf.shape} to {(self.num_envs, correct_obs_size)}")
            self.obs_buf = torch.zeros((self.num_envs, correct_obs_size), 
                                     device=self.device, dtype=self.obs_buf.dtype)
        
        # Update self observation buffer if size doesn't match
        if self.self_obs_buf.shape[1] != correct_self_obs_size:
            print(f"Resizing self_obs_buf from {self.self_obs_buf.shape} to {(self.num_envs, correct_self_obs_size)}")
            self.self_obs_buf = torch.zeros((self.num_envs, correct_self_obs_size), 
                                          device=self.device, dtype=self.self_obs_buf.dtype)
                                          
        # Update config to reflect correct observation size
        self.cfg["env"]["numObservations"] = correct_obs_size
        print(f"[DEBUG] Final obs_buf shape: {self.obs_buf.shape}")
        print(f"[DEBUG] Final self_obs_buf shape: {self.self_obs_buf.shape}")
        print(f"[DEBUG] Updated cfg numObservations to: {correct_obs_size}")

    def _compute_partner_contact_obs(self, env_ids):
        """Compute partner contact observations for given environment IDs"""
        # Vectorized computation - handle all environments at once
        device = self.device

        # Convert env_ids to tensor if not already
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=device)

        # Find partner environment IDs (vectorized pair logic: even/odd pairing)
        partner_env_ids = get_partner_env_ids(env_ids)

        # Check if all partners exist
        valid_partners = partner_env_ids < self.num_envs
        if not torch.all(valid_partners):
            invalid_partners = partner_env_ids[~valid_partners]
            raise Exception(f"Partner env_ids out of bounds: {invalid_partners.tolist()}")

        # Get partner's hand contact observations (batch)
        partner_hand_contact = self._compute_hand_contact_obs(partner_env_ids)

        return partner_hand_contact

    def _compute_partner_obs_v3(self, env_ids):
        """Compute partner observations v3 with contact information and optional wrist-relative partner positions"""
        # Get base partner observations (v2)
        base_partner_obs = self._compute_partner_obs_v2(env_ids)
        
        # Check if wrist-relative features are enabled (default True for new feature)
        enable_wrist_relative = self.cfg["env"].get("enable_wrist_relative_obs", True)
        
        obs_parts = [base_partner_obs]
        
        # Add wrist-relative partner observations if enabled
        if enable_wrist_relative:
            wrist_relative_partner_obs = self._compute_wrist_relative_partner_obs(env_ids)
            obs_parts.append(wrist_relative_partner_obs)
        
        # Get partner contact observations
        partner_contact_obs = self._compute_partner_contact_obs(env_ids)
        if partner_contact_obs.shape[1] > 0:
            obs_parts.append(partner_contact_obs)
        
        full_partner_obs = torch.cat(obs_parts, dim=-1)
        
        return full_partner_obs

    def _compute_wrist_relative_partner_obs(self, env_ids):
        """Compute partner joint positions relative to self's L_Wrist and R_Wrist positions"""
        # Setup body name to ID mapping if not already done
        self._setup_body_name_to_id_mapping()

        # Get wrist body IDs
        l_wrist_id = self._body_name_to_id.get('L_Wrist')
        r_wrist_id = self._body_name_to_id.get('R_Wrist')

        if l_wrist_id is None or r_wrist_id is None:
            error_msg = f"Wrist IDs not found in body name to ID mapping: L_Wrist={l_wrist_id}, R_Wrist={r_wrist_id}"
            raise ValueError(error_msg)

        # Vectorized computation - handle all environments at once
        num_envs = len(env_ids)
        device = self.device

        # Convert env_ids to tensor if not already
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=device)

        # Find partner environment IDs (vectorized pair logic: even/odd pairing)
        partner_env_ids = get_partner_env_ids(env_ids)

        # Check if all partners exist
        valid_partners = partner_env_ids < self.num_envs
        if not torch.all(valid_partners):
            invalid_partners = partner_env_ids[~valid_partners]
            raise ValueError(f"Partner environment IDs not found: {invalid_partners.tolist()}")

        # Get current humanoid's wrist positions and orientations (batch)
        current_l_wrist_pos = self._rigid_body_pos[env_ids, l_wrist_id, :3].clone()  # [num_envs, 3]
        current_r_wrist_pos = self._rigid_body_pos[env_ids, r_wrist_id, :3].clone()  # [num_envs, 3]
        current_l_wrist_rot = self._rigid_body_rot[env_ids, l_wrist_id, :]    # [num_envs, 4]
        current_r_wrist_rot = self._rigid_body_rot[env_ids, r_wrist_id, :]    # [num_envs, 4]

        # Get partner body positions (batch)
        partner_body_pos = self._rigid_body_pos[partner_env_ids, :, :3].clone()  # [num_envs, num_bodies, 3]

        # Apply motion cache offset compensation (vectorized)
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], device=device, dtype=partner_body_pos.dtype)

        # Partner offset (recipients get offset)
        partner_is_recipient = (partner_env_ids % 2 == 1)
        partner_body_pos[partner_is_recipient] = partner_body_pos[partner_is_recipient] + motion_cache_offset.unsqueeze(0).unsqueeze(0)

        # Current environment offset (recipients get offset)
        current_is_recipient = (env_ids % 2 == 1)
        current_l_wrist_pos[current_is_recipient] = current_l_wrist_pos[current_is_recipient] + motion_cache_offset
        current_r_wrist_pos[current_is_recipient] = current_r_wrist_pos[current_is_recipient] + motion_cache_offset

        # Normalize wrist rotations (vectorized)
        current_l_wrist_rot = normalize_quaternion_batch(current_l_wrist_rot)
        current_r_wrist_rot = normalize_quaternion_batch(current_r_wrist_rot)

        # Compute partner joint positions relative to L_Wrist (vectorized)
        # partner_body_pos: [num_envs, num_bodies, 3]
        # current_l_wrist_pos: [num_envs, 3]
        partner_pos_rel_l_wrist = partner_body_pos - current_l_wrist_pos.unsqueeze(1)  # [num_envs, num_bodies, 3]
        num_bodies = partner_pos_rel_l_wrist.shape[1]

        # Expand wrist rotations for all bodies
        l_wrist_rot_batch = current_l_wrist_rot.unsqueeze(1).expand(-1, num_bodies, -1)  # [num_envs, num_bodies, 4]
        partner_pos_l_wrist_local = quat_rotate_inverse(l_wrist_rot_batch.reshape(-1, 4),
                                                       partner_pos_rel_l_wrist.reshape(-1, 3))
        partner_pos_l_wrist_local = partner_pos_l_wrist_local.reshape(num_envs, -1)  # [num_envs, num_bodies * 3]

        # Compute partner joint positions relative to R_Wrist (vectorized)
        partner_pos_rel_r_wrist = partner_body_pos - current_r_wrist_pos.unsqueeze(1)  # [num_envs, num_bodies, 3]
        r_wrist_rot_batch = current_r_wrist_rot.unsqueeze(1).expand(-1, num_bodies, -1)  # [num_envs, num_bodies, 4]
        partner_pos_r_wrist_local = quat_rotate_inverse(r_wrist_rot_batch.reshape(-1, 4),
                                                       partner_pos_rel_r_wrist.reshape(-1, 3))
        partner_pos_r_wrist_local = partner_pos_r_wrist_local.reshape(num_envs, -1)  # [num_envs, num_bodies * 3]

        # Combine L_Wrist and R_Wrist relative observations
        wrist_relative_obs = torch.cat([
            partner_pos_l_wrist_local,  # [num_envs, num_bodies * 3] - partner positions relative to L_Wrist
            partner_pos_r_wrist_local   # [num_envs, num_bodies * 3] - partner positions relative to R_Wrist
        ], dim=1)  # [num_envs, num_bodies * 6]

        return wrist_relative_obs

    def _compute_partner_obs_v2(self, env_ids):
        """Compute partner observations v2 (existing implementation)"""
        # This should call the original implementation with partner_obs_v=2
        return self._compute_partner_obs_v1_v2(env_ids)

    def _get_partner_obs_size(self):
        """Override to calculate partner observation size based on version"""
        # Handle case where cfg is not fully initialized
        if not hasattr(self, 'cfg') or self.cfg is None:
            return 0
        
        partner_obs_v = self.cfg["env"].get("partner_obs_v", 1)
        
        if partner_obs_v == 3:
            # V3: V2 + optional wrist-relative partner positions + partner contact information
            base_size = self._get_partner_obs_size_v2()
            contact_size = self._get_hand_contact_obs_size()
            
            # Check if wrist-relative features are enabled
            enable_wrist_relative = self.cfg["env"].get("enable_wrist_relative_obs", True)
            if enable_wrist_relative:
                wrist_relative_size = self._get_wrist_relative_partner_obs_size()
                return base_size + wrist_relative_size + contact_size
            else:
                return base_size + contact_size
        elif partner_obs_v == 2:
            return self._get_partner_obs_size_v2()
        else:
            return self._get_partner_obs_size_v2()

    def _get_partner_obs_size_v2(self):
        """Calculate partner observation size for v2"""
        # Based on current implementation: joint_pos + joint_vel + joint_rot_6d + root_rot_gravity_6d
        if hasattr(self, '_body_names') and self._body_names is not None:
            num_bodies = len(self._body_names)
        elif hasattr(self, 'num_bodies'):
            num_bodies = self.num_bodies  # Use actual body count from parent class
        else:
            # Fallback values during early initialization
            humanoid_type = getattr(self, 'humanoid_type', 'smpl')
            num_bodies = 23 if humanoid_type == 'smpl' else 52 if humanoid_type == 'smplx' else 23
        return num_bodies * 3 + num_bodies * 3 + num_bodies * 6 + 6  # joint_pos + joint_vel + joint_rot_6d + root_rot_gravity_6d
    
    def _get_wrist_relative_partner_obs_size(self):
        """Calculate wrist-relative partner observation size"""
        # Based on current implementation: 2 wrists * num_bodies * 3D coordinates
        if hasattr(self, '_body_names') and self._body_names is not None:
            num_bodies = len(self._body_names)
        elif hasattr(self, 'num_bodies'):
            num_bodies = self.num_bodies  # Use actual body count from parent class
        else:
            # Fallback values during early initialization
            humanoid_type = getattr(self, 'humanoid_type', 'smpl')
            num_bodies = 23 if humanoid_type == 'smpl' else 52 if humanoid_type == 'smplx' else 23
        return num_bodies * 3 * 2  # 2 wrists * num_bodies * 3D

    def _setup_body_name_to_id_mapping(self):
        """Setup mapping from body names to IDs for contact processing"""
        if hasattr(self, '_body_name_to_id'):
            return
            
        self._body_name_to_id = {}
        
        # Use the first environment to get body handles
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        
        # Get all body names from the environment
        if hasattr(self, '_body_names'):
            body_names = self._body_names
        else:
            # Fallback to common SMPL-X body names
            body_names = [
                'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
                'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
                'Torso', 'Spine', 'Chest', 'Neck', 'Head',
                'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
            ]
        
        # Map each body name to its ID
        for i, body_name in enumerate(body_names):
            try:
                body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
                if body_id != -1:
                    self._body_name_to_id[body_name] = body_id
            except:
                # Handle cases where body name doesn't exist
                continue
        
        print(f"Setup body name to ID mapping for {len(self._body_name_to_id)} bodies")

    def _get_hand_force_obs_size(self):
        """Calculate hand and elbow force observation size"""
        # Hand joints (including fingertips) and elbows force bodies:
        # ['L_Wrist', 'R_Wrist']
        # 2 bodies * 3D force vector = 6 values
        return 42

    def _get_hand_contact_obs_size(self):
        """Calculate hand contact observation size"""
        # Handle case where cfg is not fully initialized
        if not hasattr(self, 'cfg') or self.cfg is None:
            return 0
        
        # Hand contact bodies from config
        hand_contact_bodies = self.cfg["env"].get("hand_contact_bodies", [])
        
        # Each contact body contributes only the contact flag: 1 value
        return len(hand_contact_bodies)

    def _compute_hand_force_obs(self, env_ids):
        """Compute hand and elbow force observations for given environment IDs (vectorized)"""
        # Define hand joints (including fingertips) and elbows
        force_body_names = ['L_Wrist', 'L_Index3', 'L_Middle3', 'L_Ring3', 'L_Pinky3', 'L_Thumb3',
                           'R_Wrist', 'R_Index3', 'R_Middle3', 'R_Ring3', 'R_Pinky3', 'R_Thumb3',
                           'L_Elbow', 'R_Elbow']

        # Setup body name to ID mapping if not already done
        self._setup_body_name_to_id_mapping()

        # Get force body IDs
        force_body_ids = []
        for body_name in force_body_names:
            body_id = self._body_name_to_id.get(body_name)
            if body_id is not None:
                force_body_ids.append(body_id)

        if len(force_body_ids) != len(force_body_names):
            msg = f"Expected {len(force_body_names)} force bodies, got {len(force_body_ids)}"
            raise Exception(msg)

        # Convert env_ids to tensor if needed
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device)

        batch_size = len(env_ids)
        num_force_bodies = len(force_body_ids)

        # Batch fetch all root rotations
        current_root_rot = self._rigid_body_rot[env_ids, 0, :]  # [batch_size, 4]

        # Vectorized quaternion normalization
        quat_norms = torch.norm(current_root_rot, dim=1, keepdim=True)
        zero_mask = quat_norms.squeeze(1) < 1e-6
        current_root_rot[zero_mask] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        current_root_rot[~zero_mask] = current_root_rot[~zero_mask] / torch.clamp(quat_norms[~zero_mask], min=1e-8)

        # Batch fetch contact forces for all force bodies
        env_forces = self._contact_forces[env_ids][:, force_body_ids, :]  # [batch_size, num_force_bodies, 3]

        # Vectorized transformation to local frame
        # Expand quaternions for batch processing
        current_root_rot_expanded = current_root_rot.unsqueeze(1).expand(-1, num_force_bodies, -1)  # [batch_size, num_force_bodies, 4]
        env_forces_flat = env_forces.reshape(-1, 3)  # [batch_size * num_force_bodies, 3]
        current_root_rot_flat = current_root_rot_expanded.reshape(-1, 4)  # [batch_size * num_force_bodies, 4]

        # Transform all force vectors to local frame
        force_local_flat = quat_rotate_inverse(current_root_rot_flat, env_forces_flat)  # [batch_size * num_force_bodies, 3]
        force_local = force_local_flat.reshape(batch_size, num_force_bodies, 3)  # [batch_size, num_force_bodies, 3]

        # Flatten to get final observation format
        force_obs = force_local.reshape(batch_size, -1)  # [batch_size, num_force_bodies * 3]

        return force_obs

    def _compute_hand_contact_obs(self, env_ids):
        """Compute hand contact observations for given environment IDs - FORCE-ONLY FOR SPEED"""
        hand_contact_bodies = self.cfg["env"].get("hand_contact_bodies", [])
        if not hand_contact_bodies:
            return torch.zeros((len(env_ids), 0), device=self.device)
        
        hand_contact_threshold = self.cfg["env"].get("hand_contact_threshold", 0.1)
        
        # Setup body name to ID mapping if not already done
        self._setup_body_name_to_id_mapping()
        
        # Get hand contact body IDs (batch processing)
        hand_contact_body_ids = [self._body_name_to_id[body_name] for body_name in hand_contact_bodies if body_name in self._body_name_to_id]
        
        if not hand_contact_body_ids:
            msg = f"No hand contact body IDs found"
            raise Exception(msg)
        
        # Extract contact forces for hand bodies
        hand_contact_forces = self._contact_forces[env_ids][:, hand_contact_body_ids, :]  # [len(env_ids), n_hand_bodies, 3]
        
        # Compute contact magnitudes
        contact_magnitudes = torch.norm(hand_contact_forces, dim=-1)  # [len(env_ids), n_hand_bodies]
        
        force_contact = contact_magnitudes > hand_contact_threshold
        # # DISTANCE-BASED CONTACT CHECK COMMENTED OUT FOR SPEED:
        distance_threshold = self.cfg["env"].get("hand_contact_proximity_threshold", 0.2)  # default 20cm
        
        # Specific hand joints for distance check: Wrist
        specific_hand_joints = ['L_Wrist', 'R_Wrist']
        
        # Get specific hand joint IDs that are also in hand_contact_bodies
        specific_joint_ids = []
        specific_joint_indices = []
        for body_name in hand_contact_bodies:
            if body_name in specific_hand_joints:
                body_id = self._body_name_to_id.get(body_name)
                if body_id is not None:
                    specific_joint_ids.append(body_id)
                    # Find the index in hand_contact_body_ids
                    try:
                        idx = hand_contact_body_ids.index(body_id)
                        specific_joint_indices.append(idx)
                    except ValueError:
                        pass
        
        # Initialize distance contact flags
        distance_contact = torch.zeros_like(force_contact)  # [len(env_ids), n_hand_bodies]
        
        if specific_joint_ids:
            # Batch processing for distance condition check
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device)

            # Get partner env_ids (batch)
            partner_env_ids = get_partner_env_ids(env_ids_tensor)

            # Check bounds (batch)
            valid_current = env_ids_tensor < self._rigid_body_pos.shape[0]
            valid_partner = partner_env_ids < self._rigid_body_pos.shape[0]
            valid_mask = valid_current & valid_partner
            
            assert torch.all(valid_mask), "Invalid mask"

            if torch.any(valid_mask):
                valid_indices = torch.where(valid_mask)[0]
                valid_env_ids = env_ids_tensor[valid_indices]
                valid_partner_ids = partner_env_ids[valid_indices]

                # Get wrist body IDs (batch processing) + upper body joint IDs
                wrist_body_names = ['L_Wrist', 'R_Wrist']
                partner_wrist_ids = [self._body_name_to_id[name] for name in wrist_body_names if name in self._body_name_to_id]

                if partner_wrist_ids:
                    # Get current environments' specific hand joint positions (batch)
                    current_hand_joint_pos = self._rigid_body_pos[valid_env_ids][:, specific_joint_ids, :3]  # [n_valid, n_specific_joints, 3]

                    # Get partner wrist positions (batch)
                    partner_body_pos = self._rigid_body_pos[valid_partner_ids][:, partner_wrist_ids, :3]  # [n_valid, n_wrists, 3]

                    # Apply motion cache offset compensation (batch)
                    pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
                    motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], device=self.device, dtype=partner_body_pos.dtype)

                    # For caregivers (even env_ids), adjust partner positions
                    caregiver_mask = valid_env_ids % 2 == 0
                    if torch.any(caregiver_mask):
                        partner_body_pos[caregiver_mask] = partner_body_pos[caregiver_mask] + motion_cache_offset.unsqueeze(0).unsqueeze(0)

                    # For recipients (odd env_ids), adjust current positions
                    recipient_mask = valid_env_ids % 2 == 1
                    if torch.any(recipient_mask):
                        current_hand_joint_pos[recipient_mask] = current_hand_joint_pos[recipient_mask] + motion_cache_offset.unsqueeze(0).unsqueeze(0)

                    # Compute distances (batch)
                    # current_hand_joint_pos: [n_valid, n_specific_joints, 3]
                    # partner_body_pos: [n_valid, n_wrists, 3]
                    distances = torch.cdist(current_hand_joint_pos, partner_body_pos)  # [n_valid, n_specific_joints, n_wrists]

                    # Check distance threshold (batch)
                    min_distances = torch.min(distances, dim=2)[0]  # [n_valid, n_specific_joints]
                    within_distance = (min_distances < distance_threshold)  # [n_valid, n_specific_joints] - keep as bool

                    # Map back to hand_contact_bodies indices (batch)
                    for j, specific_idx in enumerate(specific_joint_indices):
                        if j < within_distance.shape[1]:
                            distance_contact[valid_indices, specific_idx] = within_distance[:, j]
        
        # Combine force-based contact and distance-based contact (both conditions must be satisfied)
        is_contacting = force_contact * distance_contact  # [len(env_ids), n_hand_bodies]
        
        return is_contacting

    
    def _compute_humanoid_obs_v4(self, env_ids):
        """Compute humanoid observations v4 with hand contact information"""
        # Get base observations directly from parent class (bypassing our override)
        # Use the parent class method directly to avoid infinite recursion
        parent_class = super(HumanoidImInterxHelpUp, self)
        base_obs = parent_class._compute_humanoid_obs(env_ids)
        
        # Get hand contact observations
        hand_contact_obs = self._compute_hand_contact_obs(env_ids)
        
        # Combine observations
        if hand_contact_obs.shape[1] > 0:
            full_obs = torch.cat([base_obs, hand_contact_obs], dim=-1)
        else:
            full_obs = base_obs
        
        return full_obs

    def _compute_humanoid_obs_v5(self, env_ids):
        """Compute humanoid observations v5 with hand contact and force information"""
        # Get base observations directly from parent class (bypassing our override)
        # Use the parent class method directly to avoid infinite recursion
        parent_class = super(HumanoidImInterxHelpUp, self)
        base_obs = parent_class._compute_humanoid_obs(env_ids)
        
        # Get hand contact observations
        hand_contact_obs = self._compute_hand_contact_obs(env_ids)
        
        # Get hand force observations
        hand_force_obs = self._compute_hand_force_obs(env_ids)
        
        # Combine observations: [base_obs, hand_contact_obs, hand_force_obs]
        obs_parts = [base_obs]
        if hand_contact_obs.shape[1] > 0:
            obs_parts.append(hand_contact_obs)
        if hand_force_obs.shape[1] > 0:
            obs_parts.append(hand_force_obs)
        
        full_obs = torch.cat(obs_parts, dim=-1)
        
        return full_obs

    def _compute_humanoid_obs(self, env_ids=None):
        """Override to support different self_obs_v versions"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)
        
        # Check self_obs_v version
        self_obs_v = self.cfg["env"].get("self_obs_v", 1)
        
        if self_obs_v == 4:
            return self._compute_humanoid_obs_v4(env_ids)
        elif self_obs_v == 5:
            return self._compute_humanoid_obs_v5(env_ids)
        else:
            # Default to original implementation
            return super()._compute_humanoid_obs(env_ids)

    def _adjust_caregiver_hand_reference(self, env_ids, ref_rb_pos_subset, motion_res):
        """
        Adjust caregiver's hand reference motion positions based on recipient's physical simulation position.

        This function implements adaptive hand positioning: when recipient deviates from reference motion,
        caregiver's hand reference is adjusted to maintain the relative position relationship from the
        reference motion, allowing the caregiver to support the deviated recipient.

        Args:
            env_ids: Environment IDs to process
            ref_rb_pos_subset: Reference body positions [B, num_tracked_bodies, 3]
            motion_res: Full motion library results containing all body positions

        Returns:
            Adjusted ref_rb_pos_subset with caregiver hand positions modified
        """
        # Only activate when caregiver-recipient root distance <= threshold
        DISTANCE_THRESHOLD = self.cfg["env"].get("force_tracking_distance_threshold", 1.3)

        # Identify caregiver and recipient pairs
        caregiver_mask = (env_ids % 2) == 0  # Even env_ids are caregivers
        caregiver_env_ids = env_ids[caregiver_mask]

        # For each caregiver, get corresponding recipient
        recipient_env_ids = caregiver_env_ids + 1

        # Check if recipient env_ids are valid
        valid_mask = recipient_env_ids < self.num_envs
        if not torch.all(valid_mask):
            error_msg = f"Recipient env_ids out of bounds: {recipient_env_ids[~valid_mask].tolist()}"
            raise Exception(error_msg)

        # Get root positions and calculate distances
        caregiver_root_pos = self._rigid_body_pos[caregiver_env_ids, 0, :].clone()  # [N, 3]
         # Cancel out the offset in recipient reference positions for relative position calculation
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        offset_correction = torch.tensor([pair_offset_val, 0.0, 0.0], device=self.device, dtype=caregiver_root_pos.dtype)
        recipient_root_pos = self._rigid_body_pos[recipient_env_ids, 0, :].clone()  # [N, 3]
        recipient_root_pos = recipient_root_pos + offset_correction  # [N, 3]
        distances = torch.norm(caregiver_root_pos - recipient_root_pos, dim=1)  # [N]

        # Filter pairs within threshold
        close_pairs_mask = distances <= DISTANCE_THRESHOLD
        caregiver_env_ids_close = caregiver_env_ids[close_pairs_mask]
        recipient_env_ids_close = recipient_env_ids[close_pairs_mask]

        if len(caregiver_env_ids_close) == 0:
            return ref_rb_pos_subset

        # Get wrist joint IDs
        l_wrist_id = self._body_name_to_id.get('L_Wrist')
        r_wrist_id = self._body_name_to_id.get('R_Wrist')

        if l_wrist_id is None or r_wrist_id is None:
            error_msg = f"Wrist IDs not found in body name to ID mapping: L_Wrist={l_wrist_id}, R_Wrist={r_wrist_id}"
            raise Exception(error_msg)

        # Define hand joint names and get track indices
        hand_joint_names = [
            'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3',
            'L_Middle1', 'L_Middle2', 'L_Middle3',
            'L_Ring1', 'L_Ring2', 'L_Ring3',
            'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
            'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
            'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3',
            'R_Middle1', 'R_Middle2', 'R_Middle3',
            'R_Ring1', 'R_Ring2', 'R_Ring3',
            'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
            'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
        ]

        left_hand_track_indices = []
        right_hand_track_indices = []
        for joint_name in hand_joint_names:
            if joint_name in self._body_name_to_id:
                body_id = self._body_name_to_id[joint_name]
                if body_id in self._track_bodies_id:
                    track_idx = self._track_bodies_id.tolist().index(body_id)
                    if joint_name.startswith('L_'):
                        left_hand_track_indices.append(track_idx)
                    else:
                        right_hand_track_indices.append(track_idx)


        if len(left_hand_track_indices) == 0 and len(right_hand_track_indices) == 0:
            erorr_msg = f"No left or right hand track indices found in body name to ID mapping: {hand_joint_names}"
            raise Exception(erorr_msg)

        # Get indices in env_ids batch for caregivers
        caregiver_indices_in_batch = []
        for cg_id in caregiver_env_ids_close:
            idx = (env_ids == cg_id).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                caregiver_indices_in_batch.append(idx[0].item())

        caregiver_indices_in_batch = torch.tensor(caregiver_indices_in_batch, device=self.device, dtype=torch.long)

        # Get recipient indices in env_ids batch
        recipient_indices_in_batch = []
        for rc_id in recipient_env_ids_close:
            idx = (env_ids == rc_id).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                recipient_indices_in_batch.append(idx[0].item())

        recipient_indices_in_batch = torch.tensor(recipient_indices_in_batch, device=self.device, dtype=torch.long)


        # Get reference and simulation positions (one-time computation)
        caregiver_ref_pos = motion_res["rg_pos"][caregiver_indices_in_batch]  # [M, num_all_bodies, 3]
        recipient_ref_pos = motion_res["rg_pos"][recipient_indices_in_batch].clone()  # [M, num_all_bodies, 3]
        recipient_sim_pos = self._rigid_body_pos[recipient_env_ids_close] + offset_correction  # [M, num_all_bodies, 3]

        recipient_ref_pos_corrected = recipient_ref_pos + offset_correction  # [M, num_all_bodies, 3]

        # Process L_Wrist
        caregiver_l_wrist_ref = caregiver_ref_pos[:, l_wrist_id, :]  # [M, 3]
        distances_l = torch.norm(recipient_ref_pos_corrected - caregiver_l_wrist_ref.unsqueeze(1), dim=2)  # [M, num_all_bodies]
        nearest_recipient_joint_id_l = torch.argmin(distances_l, dim=1)  # [M]

        batch_indices = torch.arange(len(nearest_recipient_joint_id_l), device=self.device)
        recipient_ref_nearest_l = recipient_ref_pos_corrected[batch_indices, nearest_recipient_joint_id_l]  # [M, 3]
        recipient_sim_nearest_l = recipient_sim_pos[batch_indices, nearest_recipient_joint_id_l]  # [M, 3]

        # Calculate relative position in reference motion (now in same coordinate system)
        relative_pos_l = caregiver_l_wrist_ref - recipient_ref_nearest_l  # [M, 3]
        # If the nearest reference joint is within threshold, snap to the recipient's simulation
        # nearest joint position; otherwise, maintain relative offset.
        nearest_ref_distance_thresh = 0.0  # stopped this adjustment
        min_dist_l = distances_l[batch_indices, nearest_recipient_joint_id_l]  # [M]
        close_mask_l = min_dist_l <= nearest_ref_distance_thresh  # [M]
        new_l_hand_target = torch.where(
            close_mask_l.unsqueeze(1),
            recipient_sim_nearest_l,
            recipient_sim_nearest_l + relative_pos_l,
        )  # [M, 3]
        l_wrist_offset = new_l_hand_target - caregiver_l_wrist_ref  # [M, 3]

        # Process R_Wrist
        caregiver_r_wrist_ref = caregiver_ref_pos[:, r_wrist_id, :]  # [M, 3]
        distances_r = torch.norm(recipient_ref_pos_corrected - caregiver_r_wrist_ref.unsqueeze(1), dim=2)  # [M, num_all_bodies]
        nearest_recipient_joint_id_r = torch.argmin(distances_r, dim=1)  # [M]

        recipient_ref_nearest_r = recipient_ref_pos_corrected[batch_indices, nearest_recipient_joint_id_r]  # [M, 3]
        recipient_sim_nearest_r = recipient_sim_pos[batch_indices, nearest_recipient_joint_id_r]  # [M, 3]

        # Calculate relative position in reference motion (now in same coordinate system)
        relative_pos_r = caregiver_r_wrist_ref - recipient_ref_nearest_r  # [M, 3]
        # Conditional snapping similar to left hand
        min_dist_r = distances_r[batch_indices, nearest_recipient_joint_id_r]  # [M]
        close_mask_r = min_dist_r <= nearest_ref_distance_thresh  # [M]
        new_r_hand_target = torch.where(
            close_mask_r.unsqueeze(1),
            recipient_sim_nearest_r,
            recipient_sim_nearest_r + relative_pos_r,
        )  # [M, 3]
        r_wrist_offset = new_r_hand_target - caregiver_r_wrist_ref  # [M, 3]

        # Apply offsets to all hand joints
        for track_idx in left_hand_track_indices:
            ref_rb_pos_subset[caregiver_indices_in_batch, track_idx] += l_wrist_offset

        for track_idx in right_hand_track_indices:
            ref_rb_pos_subset[caregiver_indices_in_batch, track_idx] += r_wrist_offset

        return ref_rb_pos_subset

    def _compute_task_obs_with_role_handling(self, env_ids=None, save_buffer=True):
        """Override to implement role-specific task observations like MultiPulse"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        
        # Note: Removed unused cache initialization (_task_obs_cache, _force_obs_cache)
        
        # Get current states
        body_pos = self._rigid_body_pos[env_ids]
        body_rot = self._rigid_body_rot[env_ids]
        body_vel = self._rigid_body_vel[env_ids]
        body_ang_vel = self._rigid_body_ang_vel[env_ids]
        
        # Get reference motion data using parent class method
        if self._fut_tracks:
            time_steps = self._num_traj_samples
            B = env_ids.shape[0]
            time_internals = torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps) * self._traj_sample_timestep
            motion_times_steps = ((self.progress_buf[env_ids, None] + 1) * self.dt + time_internals + self._motion_start_times[env_ids, None] + self._motion_start_times_offset[env_ids, None]).flatten()
            env_ids_steps = self._sampled_motion_ids[env_ids].repeat_interleave(time_steps)
            motion_res = self._get_state_from_motionlib_cache(env_ids_steps, motion_times_steps, self._global_offset[env_ids].repeat_interleave(time_steps, dim=0).view(-1, 3))
        else:
            motion_times = (self.progress_buf[env_ids] + 1) * self.dt + self._motion_start_times[env_ids] + self._motion_start_times_offset[env_ids]
            time_steps = 1
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids])
        
        ref_rb_pos_subset = motion_res["rg_pos"][..., self._track_bodies_id, :]
        ref_rb_rot_subset = motion_res["rb_rot"][..., self._track_bodies_id, :]
        ref_body_vel_subset = motion_res["body_vel"][..., self._track_bodies_id, :]
        ref_body_ang_vel_subset = motion_res["body_ang_vel"][..., self._track_bodies_id, :]
        
        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]
        
        # Apply motion capture accuracy improvements for hand/finger joints
        # Replace reference for joints beyond elbow (wrist and fingers) with current state
        # This applies to both recipients and caregivers due to poor mocap accuracy for hand movements
        
        # Define joints beyond elbow (wrist and all finger joints)
        beyond_elbow_joints = [
            'L_Wrist',
            'R_Wrist',
        ]
        
        # Get body name to ID mapping if not already done
        if not hasattr(self, '_body_name_to_id'):
            self._setup_body_name_to_id_mapping()
        
        # Find which tracked bodies correspond to beyond-elbow joints
        beyond_elbow_track_indices = []
        for joint_name in beyond_elbow_joints:
            if joint_name in self._body_name_to_id:
                body_id = self._body_name_to_id[joint_name]
                # Find index in _track_bodies_id
                if body_id in self._track_bodies_id:
                    track_idx = self._track_bodies_id.tolist().index(body_id)
                    beyond_elbow_track_indices.append(track_idx)
        
        # # Replace reference with current state for beyond-elbow joints
        # for track_idx in beyond_elbow_track_indices:
        #     ref_rb_pos_subset[:, track_idx] = body_pos_subset[:, track_idx]
        #     ref_rb_rot_subset[:, track_idx] = body_rot_subset[:, track_idx]
        #     ref_body_vel_subset[:, track_idx] = body_vel_subset[:, track_idx]
        #     ref_body_ang_vel_subset[:, track_idx] = body_ang_vel_subset[:, track_idx]

        # Adjust caregiver hand reference positions based on recipient's physical simulation position
        # ref_rb_pos_subset = self._adjust_caregiver_hand_reference(env_ids, ref_rb_pos_subset, motion_res)

        # Compute observations using the modified references
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]
        
        # Use the same observation computation as the parent class based on obs_v
        if self.obs_v == 1:
            from phc.env.tasks.humanoid_im import compute_imitation_observations
            obs = compute_imitation_observations(
                root_pos, root_rot, body_pos_subset, body_rot_subset, 
                body_vel_subset, body_ang_vel_subset,
                ref_rb_pos_subset, ref_rb_rot_subset, 
                ref_body_vel_subset, ref_body_ang_vel_subset, 
                time_steps, self._has_upright_start
            )
        elif self.obs_v == 6:
            from phc.env.tasks.humanoid_im import compute_imitation_observations_v6
            obs = compute_imitation_observations_v6(
                root_pos, root_rot, body_pos_subset, body_rot_subset, 
                body_vel_subset, body_ang_vel_subset,
                ref_rb_pos_subset, ref_rb_rot_subset, 
                ref_body_vel_subset, ref_body_ang_vel_subset, 
                time_steps, self._has_upright_start
            )
        else:
            # Fallback to default parent implementation
            obs = super()._compute_task_obs(env_ids, save_buffer)
        return obs

    def get_obs_size(self):
        """Override to account for contact-enhanced observations and SimpleLiftUp mode"""
        # SimpleLiftUp mode: use actual observation size (no PNN statistics dependency)
        if self.simple_lift_up_mode:
            # Get self observation size only (proprioception)
            self_obs_size = self.get_self_obs_size()
            if self_obs_size is None:
                self_obs_size = 0
            
            # Include aux_features for MLP processing
            aux_features_size = self._get_aux_features_size()
            self.task_obs_size = 1248
            
            # Use actual size for new MLP (no dummy padding needed)
            # Add +1 for role label (env_id % 2)
            total_size = self_obs_size + aux_features_size + self.task_obs_size
            # print(f"DEBUG SimpleLiftUp: self_obs_size={self_obs_size}, aux_features_size={aux_features_size}, total_size={total_size}")
            return total_size
        
        # Normal interaction mode: Get the basic observation size from parent class (base self_obs + task_obs)
        basic_obs_size = super().get_obs_size()
        
        # Handle case where basic_obs_size is None
        if basic_obs_size is None:
            basic_obs_size = 0
        
        # Only add extra sizes if we have a valid configuration
        if not hasattr(self, 'cfg') or self.cfg is None:
            return basic_obs_size
        
        # For self_obs_v=4, the basic_obs_size already includes self contact info
        # We need to calculate aux_features_size correctly
        self_obs_v = self.cfg["env"].get("self_obs_v", 1)
        partner_obs_v = self.cfg["env"].get("partner_obs_v", 1)
        
        # Calculate self contact size
        self_contact_size = 0
        if self_obs_v == 4:
            self_contact_size = self._get_hand_contact_obs_size()
        
        # Aux features: [self_contact, base_partner_obs, partner_contact, reference_contact_diff]
        if partner_obs_v == 3:
            # V3: base partner obs + optional wrist-relative partner obs, separate partner contact
            base_partner_size = self._get_partner_obs_size_v2()
            
            # Check if wrist-relative features are enabled
            enable_wrist_relative = self.cfg["env"].get("enable_wrist_relative_obs", True)
            if enable_wrist_relative:
                base_partner_size += self._get_wrist_relative_partner_obs_size()
                
            partner_contact_size = self._get_hand_contact_obs_size()
        else:
            # V1/V2: partner obs without contact  
            base_partner_size = self._get_partner_obs_size_v2()
            partner_contact_size = 0
        
        # Calculate future trajectory sizes (reduced from 60 to 30 frames, 30 hand joints removed)
        # Each trajectory: 30 frames * (22 bodies * 13 features) = 30 * 22 * 13 = 8580 features per trajectory
        # Two trajectories: self + partner = 8580 * 2 = 17160
        non_hand_bodies = 52 - 30  # Total SMPL-X bodies minus 30 hand joints = 22 bodies
        future_trajectory_size = self.future_size * non_hand_bodies * 13 * 2  # 2 for self+partner trajectories

        # MultiPulse compatibility with role label and future trajectories
        # Total size: [base_self_obs, task_obs, aux_features]
        # aux_features = [self_contact, base_partner_obs, partner_contact, self_future_traj_flat, partner_future_traj_flat, role_label]
        aux_features_size = self_contact_size + base_partner_size + partner_contact_size + future_trajectory_size + 1  # +1 for role label
        total_size = basic_obs_size + aux_features_size
        
        # print(f"DEBUG Environment (MultiPulse compatible): basic_obs_size={basic_obs_size}, self_contact_size={self_contact_size}, base_partner_size={base_partner_size}, partner_contact_size={partner_contact_size}, aux_features_size={aux_features_size}, total_size={total_size}")
        return total_size

    def get_self_obs_size(self):
        """Get self observation size WITHOUT contact information (base proprioception only)"""
        # Get base self observation size from parent (proprioception only)
        base_self_obs_size = super().get_self_obs_size()
        
        # Handle case where base_self_obs_size is None
        if base_self_obs_size is None:
            base_self_obs_size = 0
        
        # NOTE: Contact information is now handled separately in aux_features
        # so we don't add it here to avoid double-counting
        
        return base_self_obs_size


    def _compute_reward(self, actions):
        """Override to implement SimpleLiftUp reward or normal interaction reward"""
        # SimpleLiftUp mode: Check hand distances first
        hand_distance_masks, hand_rewards = self._compute_hand_distance_rewards()
        
        # If any hands are close (within 40cm), use masking reward (higher reward)
        if hand_distance_masks.any():
            self._compute_reward_with_hand_masking(actions, hand_distance_masks, hand_rewards)
            if self.power_reward:
                power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1) 
                # power_reward = -0.00005 * (power ** 2)
                power_reward = -self.power_coefficient * power
                power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

                self.rew_buf[:] += power_reward
                self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)
        else:
            # No close hands, use traditional parent reward computation (lower reward)
            super()._compute_reward(actions)
            
        # Add recipient head height reward for lift-up motion
        recipient_head_height_reward = self._compute_recipient_head_height_reward()
        self.rew_buf[:] += recipient_head_height_reward
        
        # print(self.dof_force_tensor.abs().max())
        if self.power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)

            # Apply different power coefficients for caregiver and recipient
            power_reward = torch.zeros_like(power)
            # Caregiver environments (even indices: 0, 2, 4, ...)
            caregiver_mask = torch.arange(self.num_envs, device=self.device) % 2 == 0
            # Recipient environments (odd indices: 1, 3, 5, ...)
            recipient_mask = torch.arange(self.num_envs, device=self.device) % 2 == 1

            power_reward[caregiver_mask] = -self.caregiver_power_coefficient * power[caregiver_mask]
            power_reward[recipient_mask] = -self.recipient_power_coefficient * power[recipient_mask]

            power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

            self.rew_buf[:] += power_reward
            # self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)

        # Add caregiver touch head penalty
        if self.caregiver_touch_head_reward:
            touch_head_penalty = self._compute_caregiver_touch_head_penalty()
            self.rew_buf[:] += touch_head_penalty

        # Add caregiver torque reduction reward
        if self.caregiver_torque_reduction_reward:
            torque_reduction_reward = self._compute_caregiver_torque_reduction_reward()
            self.rew_buf[:] += torque_reduction_reward

        # Integrate partner rewards with cooperative reward sharing
        self._integrate_partner_rewards()

        return

    def _integrate_partner_rewards(self):
        """Integrate partner rewards with role-based cooperative reward sharing"""
        # Store current rewards before integration
        current_rewards = self.rew_buf.clone()

        # Get partner rewards for each environment
        partner_rewards = self._get_partner_rewards(current_rewards)

        # Role-based reward combination:
        # Caregiver (even env_id): 0.3 * self + 0.7 * partner (focus on partner's success)
        # Recipient (odd env_id): 0.7 * self + 0.3 * partner (focus on own stability)
        env_ids = torch.arange(self.num_envs, device=self.device)
        is_recipient = (env_ids % 2 == 1)

        # Create weight tensors from config
        partner_integration = self.cfg["env"].get("partner_reward_integration", {})
        recipient_self_w = partner_integration.get("recipient_self_weight", 1.0)
        recipient_partner_w = partner_integration.get("recipient_partner_weight", 0.0)
        caregiver_self_w = partner_integration.get("caregiver_self_weight", 0.5)
        caregiver_partner_w = partner_integration.get("caregiver_partner_weight", 0.5)

        self_weights = torch.where(is_recipient, recipient_self_w, caregiver_self_w)
        partner_weights = torch.where(is_recipient, recipient_partner_w, caregiver_partner_w)

        # Combine rewards with role-based weights
        integrated_rewards = self_weights * current_rewards + partner_weights * partner_rewards

        # Previous uniform weighting (commented out)
        # integrated_rewards = 0.5 * current_rewards + 0.5 * partner_rewards

        # Update reward buffer
        self.rew_buf[:] = integrated_rewards

    def _get_partner_rewards(self, current_rewards):
        """Get partner rewards for all environments (vectorized implementation)"""
        # Generate all environment IDs
        env_ids = torch.arange(self.num_envs, device=self.device)

        # Vectorized partner ID computation: even -> +1, odd -> -1
        partner_env_ids = get_partner_env_ids(env_ids)

        # Ensure partner IDs are within bounds
        valid_partners = partner_env_ids < self.num_envs

        # Check if any partners are invalid and raise error
        if not valid_partners.all():
            invalid_env_ids = env_ids[~valid_partners]
            invalid_partner_ids = partner_env_ids[~valid_partners]
            raise Exception(f"Invalid partner environments found: env_ids {invalid_env_ids.tolist()} have partner_ids {invalid_partner_ids.tolist()}, but num_envs={self.num_envs}")

        # Get partner rewards (all partners are valid at this point)
        partner_rewards = current_rewards[partner_env_ids]

        return partner_rewards

    def _compute_reward_with_hand_masking(self, actions, hand_distance_masks, hand_rewards):
        """Compute reward with hand distance masking for finger/wrist tracking"""
        # Get basic body state information
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel

        motion_times = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset

        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset) 

        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        # Use the provided hand distance masks and rewards (already calculated)

        if self.zero_out_far:
            transition_distance = self.cfg["env"].get("zero_out_far_transition_distance", 0.25)
            distance = torch.norm(root_pos - ref_root_pos, dim=-1)

            zeros_subset = distance > transition_distance
            self.reward_raw = torch.zeros((self.num_envs, 5)).to(self.device)

            self.rew_buf, self.reward_raw[:, 0] = compute_point_goal_reward(self._point_goal, distance)

            # Compute imitation reward with hand masking for close environments
            env_ids_subset = torch.arange(self.num_envs, device=self.device)[~zeros_subset]
            im_reward, im_reward_raw = self._compute_masked_imitation_reward(
                root_pos[~zeros_subset, :], root_rot[~zeros_subset, :], 
                body_pos[~zeros_subset, :], body_rot[~zeros_subset, :], 
                body_vel[~zeros_subset, :], body_ang_vel[~zeros_subset, :], 
                ref_rb_pos[~zeros_subset, :], ref_rb_rot[~zeros_subset, :],
                ref_body_vel[~zeros_subset, :], ref_body_ang_vel[~zeros_subset, :], 
                hand_distance_masks[~zeros_subset], self.reward_specs, env_ids=env_ids_subset)

            im_reward_scale = self.cfg["env"].get("im_reward_scale", 0.5)
            self.rew_buf[~zeros_subset] = self.rew_buf[~zeros_subset] + im_reward * im_reward_scale
            self.reward_raw[~zeros_subset, :5] = self.reward_raw[~zeros_subset, :5] + im_reward_raw * im_reward_scale

            # Add hand distance rewards when hands are close
            self.rew_buf += hand_rewards

        else:
            if self._full_body_reward:
                # Use masked imitation reward
                env_ids_all = torch.arange(self.num_envs, device=self.device)
                self.rew_buf[:], self.reward_raw = self._compute_masked_imitation_reward(
                    root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, 
                    ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel, 
                    hand_distance_masks, self.reward_specs, env_ids=env_ids_all)
            else:
                body_pos_subset = body_pos[..., self._track_bodies_id, :]
                body_rot_subset = body_rot[..., self._track_bodies_id, :]
                body_vel_subset = body_vel[..., self._track_bodies_id, :]
                body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

                ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
                ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
                ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
                ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]

                env_ids_all = torch.arange(self.num_envs, device=self.device)
                self.rew_buf[:], self.reward_raw = self._compute_masked_imitation_reward(
                    root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, 
                    ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, 
                    hand_distance_masks, self.reward_specs, env_ids=env_ids_all)

            # Add hand distance rewards when hands are close
            self.rew_buf += hand_rewards

        return

    def _compute_hand_finger_force_multipliers(self):
        """Compute finger force multipliers for each hand (left/right) separately.

        Returns force multipliers based on formula: min(exp(|force| - 5), 1)
        Forces below 5N give exponentially smaller multipliers, forces above 5N are capped at 1.0.

        Returns:
            Tuple of 4 tensors: (care_l_force, care_r_force, recip_l_force, recip_r_force)
            Each tensor has shape [num_envs]
        """
        # Define finger joint body names for left and right hands
        left_finger_bodies = [
            'L_Index1', 'L_Index2', 'L_Index3',
            'L_Middle1', 'L_Middle2', 'L_Middle3',
            'L_Ring1', 'L_Ring2', 'L_Ring3',
            'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
            'L_Thumb1', 'L_Thumb2', 'L_Thumb3'
        ]

        right_finger_bodies = [
            'R_Index1', 'R_Index2', 'R_Index3',
            'R_Middle1', 'R_Middle2', 'R_Middle3',
            'R_Ring1', 'R_Ring2', 'R_Ring3',
            'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
            'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
        ]

        # Setup body name to ID mapping if not already done
        if not hasattr(self, '_body_name_to_id'):
            self._setup_body_name_to_id_mapping()

        # Get finger body IDs for left hand
        left_finger_ids = []
        for body_name in left_finger_bodies:
            body_id = self._body_name_to_id.get(body_name)
            if body_id is not None:
                left_finger_ids.append(body_id)

        # Get finger body IDs for right hand
        right_finger_ids = []
        for body_name in right_finger_bodies:
            body_id = self._body_name_to_id.get(body_name)
            if body_id is not None:
                right_finger_ids.append(body_id)

        left_finger_forces = self._contact_forces[:, left_finger_ids, :]  # [num_envs, num_left_fingers, 3]
        left_force_magnitudes = torch.norm(left_finger_forces, dim=-1)  # [num_envs, num_left_fingers]
        finger_force_threshold = self.cfg["env"].get("finger_force_threshold", 1.0)
        left_force_rewards = torch.clamp(torch.exp(left_force_magnitudes - finger_force_threshold), max=1.0)  # [num_envs, num_left_fingers]
        left_force_multiplier = left_force_rewards.sum(dim=-1)  # [num_envs]

        right_finger_forces = self._contact_forces[:, right_finger_ids, :]  # [num_envs, num_right_fingers, 3]
        right_force_magnitudes = torch.norm(right_finger_forces, dim=-1)  # [num_envs, num_right_fingers]
        right_force_rewards = torch.clamp(torch.exp(right_force_magnitudes - finger_force_threshold), max=1.0)  # [num_envs, num_right_fingers]
        right_force_multiplier = right_force_rewards.sum(dim=-1)  # [num_envs]
        return left_force_multiplier, right_force_multiplier

    def _compute_hand_distance_rewards(self):
        """Calculate hand distances and return masks and rewards (vectorized implementation)"""
        # Get wrist body IDs
        l_wrist_id = self._body_name_to_id.get('L_Wrist')
        r_wrist_id = self._body_name_to_id.get('R_Wrist')

        if l_wrist_id is None or r_wrist_id is None:
            # Return no masking if wrist IDs not found
            error_msg = "L_Wrist or R_Wrist not found in body name to ID mapping"
            raise Exception(error_msg)

        # Initialize rewards and masks
        hand_rewards = torch.zeros(self.num_envs, device=self.device)
        hand_distance_masks = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Hand distance reward parameters from config
        hand_dist_cfg = self.cfg["env"].get("hand_distance_reward", {})
        distance_threshold = hand_dist_cfg.get("distance_threshold", 0.4)  # default 40cm
        exp_decay_rate = hand_dist_cfg.get("exp_decay_rate", 2.5)  # Controls how fast the reward decays with distance
        reward_coefficient = hand_dist_cfg.get("reward_coefficient", 0.5)  # Weight for the exponential reward
        close_hands_bonus = hand_dist_cfg.get("close_hands_bonus", 0.05)  # Additional bonus when hands are within threshold

        # Get upper body joint IDs
        upper_body_joints = [
            'Torso', 'Spine', 'Chest', 'Neck',
            'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Knee',
            'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Knee'
        ]

        upper_body_ids = []
        for joint_name in upper_body_joints:
            body_id = self._body_name_to_id.get(joint_name)
            upper_body_ids.append(body_id)

        if not upper_body_ids:
            error_msg = "No upper body joints found in body name to ID mapping"
            raise Exception(error_msg)

        # Calculate number of environment pairs
        num_pairs = self.num_envs // 2

        # Generate caregiver and recipient environment indices
        caregiver_envs = torch.arange(0, num_pairs * 2, 2, device=self.device)  # [0, 2, 4, ...]
        recipient_envs = caregiver_envs + 1  # [1, 3, 5, ...]

        # Ensure we don't exceed num_envs
        valid_pairs = recipient_envs < self.num_envs
        caregiver_envs = caregiver_envs[valid_pairs]
        recipient_envs = recipient_envs[valid_pairs]

        # Batch fetch wrist positions
        caregiver_l_wrists = self._rigid_body_pos[caregiver_envs, l_wrist_id, :3].clone()  # [num_pairs, 3]
        caregiver_r_wrists = self._rigid_body_pos[caregiver_envs, r_wrist_id, :3].clone()  # [num_pairs, 3]
        # recipient_l_wrists = self._rigid_body_pos[recipient_envs, l_wrist_id, :3].clone()  # [num_pairs, 3]
        # recipient_r_wrists = self._rigid_body_pos[recipient_envs, r_wrist_id, :3].clone()  # [num_pairs, 3]

        # Apply offset compensation for recipients
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], device=self.device, dtype=torch.float32)
        # recipient_l_wrists = recipient_l_wrists + motion_cache_offset
        # recipient_r_wrists = recipient_r_wrists + motion_cache_offset

        # Batch fetch upper body joint positions
        caregiver_joints = self._rigid_body_pos[caregiver_envs][:, upper_body_ids, :3].clone()  # [num_pairs, num_joints, 3]
        recipient_joints = self._rigid_body_pos[recipient_envs][:, upper_body_ids, :3].clone()  # [num_pairs, num_joints, 3]

        # Apply offset compensation for recipient joints
        recipient_joints = recipient_joints + motion_cache_offset.view(1, 1, 3)

        # Vectorized distance calculations
        # Reshape for broadcasting: [num_pairs, 1, 3] - [num_pairs, num_joints, 3] = [num_pairs, num_joints]
        care_l_to_recip_distances = torch.norm(
            recipient_joints - caregiver_l_wrists.unsqueeze(1), dim=2
        )  # [num_pairs, num_joints]
        care_r_to_recip_distances = torch.norm(
            recipient_joints - caregiver_r_wrists.unsqueeze(1), dim=2
        )  # [num_pairs, num_joints]
        # recip_l_to_care_distances = torch.norm(
        #     caregiver_joints - recipient_l_wrists.unsqueeze(1), dim=2
        # )  # [num_pairs, num_joints]
        # recip_r_to_care_distances = torch.norm(
        #     caregiver_joints - recipient_r_wrists.unsqueeze(1), dim=2
        # )  # [num_pairs, num_joints]

        # Find minimum distances for each hand
        min_care_l_to_recip, _ = torch.min(care_l_to_recip_distances, dim=1)  # [num_pairs]
        min_care_r_to_recip, _ = torch.min(care_r_to_recip_distances, dim=1)  # [num_pairs]
        # min_recip_l_to_care, _ = torch.min(recip_l_to_care_distances, dim=1)  # [num_pairs]
        # min_recip_r_to_care, _ = torch.min(recip_r_to_care_distances, dim=1)  # [num_pairs]

        # Check if any hand is close to any upper body joint
        care_l_close = min_care_l_to_recip <= distance_threshold  # [num_pairs]
        care_r_close = min_care_r_to_recip <= distance_threshold  # [num_pairs]
        # recip_l_close = min_recip_l_to_care <= distance_threshold  # [num_pairs]
        # recip_r_close = min_recip_r_to_care <= distance_threshold  # [num_pairs]

        # Apply masking where any hand is close
        caregiver_mask = care_l_close | care_r_close
        # recipient_mask = recip_l_close | recip_r_close

        hand_distance_masks[caregiver_envs] = caregiver_mask
        # hand_distance_masks[recipient_envs] = recipient_mask

        # Compute finger force multipliers for each hand (left and right separately)
        left_finger_force, right_finger_force = self._compute_hand_finger_force_multipliers()

        # Extract force multipliers for valid pairs
        care_l_force = left_finger_force[caregiver_envs]  # [num_pairs]
        care_r_force = right_finger_force[caregiver_envs]  # [num_pairs]
        # recip_l_force = left_finger_force[recipient_envs]  # [num_pairs]
        # recip_r_force = right_finger_force[recipient_envs]  # [num_pairs]

        # Calculate exponential rewards for each close hand, multiplied by finger force rewards, then add bonus
        care_l_rewards = torch.where(
            care_l_close,
            reward_coefficient * torch.exp(-exp_decay_rate * min_care_l_to_recip) * care_l_force + close_hands_bonus,
            torch.zeros_like(min_care_l_to_recip)
        )
        care_r_rewards = torch.where(
            care_r_close,
            reward_coefficient * torch.exp(-exp_decay_rate * min_care_r_to_recip) * care_r_force + close_hands_bonus,
            torch.zeros_like(min_care_r_to_recip)
        )
        # recip_l_rewards = torch.where(
        #     recip_l_close,
        #     reward_coefficient * torch.exp(-exp_decay_rate * min_recip_l_to_care) * recip_l_force + close_hands_bonus,
        #     torch.zeros_like(min_recip_l_to_care)
        # )
        # recip_r_rewards = torch.where(
        #     recip_r_close,
        #     reward_coefficient * torch.exp(-exp_decay_rate * min_recip_r_to_care) * recip_r_force + close_hands_bonus,
        #     torch.zeros_like(min_recip_r_to_care)
        # )

        # Sum all rewards for each pair
        total_pair_rewards = care_l_rewards + care_r_rewards #+ recip_l_rewards + recip_r_rewards

        # Apply rewards to both caregiver and recipient in each pair
        hand_rewards[caregiver_envs] += total_pair_rewards
        # hand_rewards[recipient_envs] += total_pair_rewards

        return hand_distance_masks, hand_rewards

    def _get_curriculum_gains(self):
        """
        Calculate the current Kp and Kd values based on curriculum learning schedule.
        - epochs 0-200: use initial values
        - epochs 200-1200: linearly decrease from initial to final (0.0)

        Returns:
            tuple: (current_kp, current_kd)
        """
        if self._training_agent is None or not hasattr(self._training_agent, 'epoch_num'):
            # If no training agent, use initial values
            return self.recipient_assist_kp_pos_initial, self.recipient_assist_kd_pos_initial

        epoch_num = self._training_agent.epoch_num

        # Before curriculum start: use initial values
        if epoch_num < self.curriculum_start_epoch:
            return self.recipient_assist_kp_pos_initial, self.recipient_assist_kd_pos_initial

        # After curriculum end: use final values
        if epoch_num >= self.curriculum_start_epoch + self.curriculum_duration_epochs:
            return self.curriculum_kp_pos_final, self.curriculum_kd_pos_final

        # During curriculum: linear interpolation for both Kp and Kd
        progress = (epoch_num - self.curriculum_start_epoch) / self.curriculum_duration_epochs
        current_kp = self.recipient_assist_kp_pos_initial + progress * (self.curriculum_kp_pos_final - self.recipient_assist_kp_pos_initial)
        current_kd = self.recipient_assist_kd_pos_initial + progress * (self.curriculum_kd_pos_final - self.recipient_assist_kd_pos_initial)

        return current_kp, current_kd

    def _compute_assist_forces(self, target="recipient"):
        """
        Compute assist forces for recipients using PD control.
        - Position PD control: external force on root body only
        - Uses curriculum learning for recipient Kp value
        """
        # Zero out forces
        if target == "recipient":
            self.recipient_assist_rb_forces.zero_()
        elif target == "caregiver":
            self.caregiver_assist_rb_forces.zero_()

        # Get recipient environment IDs (odd env_ids)
        if target == "recipient":
            target_mask = torch.arange(self.num_envs, device=self.device) % 2 == 1
        elif target == "caregiver":
            target_mask = torch.arange(self.num_envs, device=self.device) % 2 == 0
        target_env_ids = torch.where(target_mask)[0]

        # Get current motion times for recipients
        motion_times = self.progress_buf[target_env_ids].float() * self.dt + self._motion_start_times[target_env_ids] + self._motion_start_times_offset[target_env_ids]

        # Get reference states from motion library (returns dict)
        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids[target_env_ids],
            motion_times,
            self._global_offset[target_env_ids]
        )

        upper_body_joints = ["Pelvis","Spine", "Chest"]
        upper_body_ids = []
        for joint_name in upper_body_joints:
            body_id = self._body_name_to_id.get(joint_name)
            upper_body_ids.append(body_id)
        upper_body_ids = torch.tensor(upper_body_ids, device=self.device, dtype=torch.long)
        upper_body_pos = self._rigid_body_pos[target_env_ids][:, upper_body_ids, :]  # [N, num_upper_body_joints, 3]
        upper_body_vel = self._rigid_body_vel[target_env_ids][:, upper_body_ids, :]  # [N, num_upper_body_joints, 3]
        ref_upper_body_pos = motion_res["rg_pos"][:, upper_body_ids, :] # [N, num_upper_body_joints, 3]
        upper_body_position_error = ref_upper_body_pos - upper_body_pos  # [N, num_upper_body_joints, 3]

        # Use curriculum learning for recipient Kp and Kd, fixed values for caregiver
        if target == "recipient":
            current_kp_pos, current_kd_pos = self._get_curriculum_gains()
        else:
            current_kp_pos = self.recipient_assist_kp_pos_initial
            current_kd_pos = self.recipient_assist_kd_pos_initial

        upper_body_position_force = (
            current_kp_pos * upper_body_position_error
            - current_kd_pos * upper_body_vel
        )  # [N, num_upper_body_joints, 3]
        # Apply force to root body (index 0) using global rigid body indices
        for i, env_id in enumerate(target_env_ids):
            # global_root_body_idx = env_id * self.get_virtual_bodies_per_env() + 0  # Root body index (0)
            # self.recipient_assist_rb_forces[global_root_body_idx, :] =root_position_force[i]
            upper_body_indices = env_id * self.get_virtual_bodies_per_env() + torch.tensor(upper_body_ids, device=self.device)
            self.recipient_assist_rb_forces[upper_body_indices, :] = upper_body_position_force[i]

    def _compute_recipient_head_height_reward(self):
        """Compute head height reward for recipients to encourage lift-up motion (batch processing)"""
        # Get head body ID
        head_id = self._body_name_to_id.get('Head')
        if head_id is None:
            return torch.zeros(self.num_envs, device=self.device)

        # Get all environment IDs as tensor
        all_env_ids = torch.arange(self.num_envs, device=self.device)

        # Create masks for caregiver (even) and recipient (odd) environments
        is_recipient = all_env_ids % 2 == 1
        is_caregiver = all_env_ids % 2 == 0

        # Get head positions for all environments
        all_head_pos = self._rigid_body_pos[:, head_id, :3]  # [num_envs, 3]

        # Initialize recipient head positions
        recipient_head_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # For recipients: use own head position (no offset needed for height calculation)
        if torch.any(is_recipient):
            recipient_head_pos[is_recipient] = all_head_pos[is_recipient]

        # For caregivers: use partner's (recipient's) head position
        if torch.any(is_caregiver):
            caregiver_env_ids = all_env_ids[is_caregiver]
            partner_env_ids = caregiver_env_ids + 1

            # Check bounds
            valid_partners = partner_env_ids < self.num_envs
            valid_caregiver_env_ids = caregiver_env_ids[valid_partners]
            valid_partner_env_ids = partner_env_ids[valid_partners]
            assert torch.all(valid_partners), "Invalid partners"
            recipient_head_pos[valid_caregiver_env_ids] = all_head_pos[valid_partner_env_ids]

        # Calculate height rewards (batch processing)
        head_heights = recipient_head_pos[:, 2]  # Z coordinates

        # Check if we should use a different body part for height
        # Try using 'Neck' or root position if Head is too low

        reward_scale = self.cfg["env"].get("head_height_reward_scale", 1.0)

        # Simple height-based reward (heights seem to be in meters but starting low)
        # Encourage any upward movement from current position
        max_head_height = self.cfg["env"].get("max_head_height_reward", 2.0)
        height_rewards = reward_scale * torch.clamp(head_heights, min=0.0, max=max_head_height)

        return height_rewards

    def _compute_caregiver_touch_head_penalty(self):
        """Compute penalty when caregiver hands touch recipient head with force (vectorized)

        Returns negative reward based on:
        - Distance between caregiver wrists and recipient head < caregiver_touch_threshold
        - Force magnitude on hand joints (wrists and fingers) when close to head
        """
        if not self.caregiver_touch_head_reward:
            return torch.zeros(self.num_envs, device=self.device)

        penalty = torch.zeros(self.num_envs, device=self.device)

        # Get body IDs
        head_id = self._body_name_to_id.get('Head')
        l_wrist_id = self._body_name_to_id.get('L_Wrist')
        r_wrist_id = self._body_name_to_id.get('R_Wrist')

        if head_id is None or l_wrist_id is None or r_wrist_id is None:
            return penalty

        # Get hand joint body names (wrists and fingers)
        hand_joint_names = [
            'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3',
            'L_Middle1', 'L_Middle2', 'L_Middle3',
            'L_Ring1', 'L_Ring2', 'L_Ring3',
            'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
            'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
            'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3',
            'R_Middle1', 'R_Middle2', 'R_Middle3',
            'R_Ring1', 'R_Ring2', 'R_Ring3',
            'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
            'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
        ]

        # Get DOF indices for hand joints (each joint has multiple DOFs)
        hand_dof_indices = []
        for joint_name in hand_joint_names:
            if joint_name in self._dof_names:
                dof_idx = self._dof_names.index(joint_name)
                # Each joint typically has 3 DOFs
                for dof_offset in range(3):
                    actual_dof_idx = dof_idx * 3 + dof_offset
                    if actual_dof_idx < self.dof_force_tensor.shape[1]:
                        hand_dof_indices.append(actual_dof_idx)

        if not hand_dof_indices:
            return penalty

        # Create caregiver and recipient masks (vectorized)
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        caregiver_mask = all_env_ids % 2 == 0
        recipient_mask = all_env_ids % 2 == 1

        # Get caregiver and recipient IDs
        caregiver_ids = all_env_ids[caregiver_mask]
        recipient_ids = caregiver_ids + 1

        # Filter out invalid pairs (where recipient_id >= num_envs)
        valid_pairs = recipient_ids < self.num_envs
        caregiver_ids = caregiver_ids[valid_pairs]
        recipient_ids = recipient_ids[valid_pairs]

        if len(caregiver_ids) == 0:
            return penalty

        # Get recipient head positions [num_pairs, 3]
        recipient_head_pos = self._rigid_body_pos[recipient_ids, head_id, :3]

        # Get caregiver wrist positions [num_pairs, 3]
        caregiver_l_wrist_pos = self._rigid_body_pos[caregiver_ids, l_wrist_id, :3]
        caregiver_r_wrist_pos = self._rigid_body_pos[caregiver_ids, r_wrist_id, :3]

        # Compensate for recipient offset (recipient initialized with -env_spacing * 2 in x direction)
        # To get true distance, add back the offset to recipient position
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], device=self.device, dtype=recipient_head_pos.dtype)
        recipient_head_pos_corrected = recipient_head_pos + motion_cache_offset

        # Compute distances [num_pairs]
        l_wrist_to_head_dist = torch.norm(caregiver_l_wrist_pos - recipient_head_pos_corrected, dim=1)
        r_wrist_to_head_dist = torch.norm(caregiver_r_wrist_pos - recipient_head_pos_corrected, dim=1)

        # Check which caregivers have wrists within threshold [num_pairs]
        within_threshold = (l_wrist_to_head_dist < self.caregiver_touch_threshold) | (r_wrist_to_head_dist < self.caregiver_touch_threshold)

        # For caregivers within threshold, compute hand force penalty
        if torch.any(within_threshold):
            close_caregiver_ids = caregiver_ids[within_threshold]

            # Get hand DOF forces for close caregivers [num_close_caregivers, num_hand_dofs]
            hand_dof_forces = self.dof_force_tensor[close_caregiver_ids][:, hand_dof_indices]

            # Sum absolute forces across all hand DOFs [num_close_caregivers]
            total_hand_forces = torch.sum(torch.abs(hand_dof_forces), dim=1)

            # Apply penalty
            penalty[close_caregiver_ids] = -self.caregiver_touch_head_coefficient * total_hand_forces

        return penalty

    def _apply_recipient_weakness(self):
        """Apply weakness to recipients by modifying their DOF properties"""
        if self.control_mode != "isaac_pd":
            return  # Only works with Isaac PD control
        
        # Get recipient weakness parameters from config
        recipient_weakness_scale = self.cfg["env"].get("recipient_weakness_scale", 0.5)
        recipient_weakness_effort = self.cfg["env"].get("recipient_weakness_effort", 80.0)
            
        # Get lower body joint indices (DOF indices, not body indices)
        lower_body_joint_names = ['L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe']
        lower_body_dof_indices = []
        
        for joint_name in lower_body_joint_names:
            if joint_name in self._dof_names:
                joint_idx = self._dof_names.index(joint_name)
                # Each joint has 3 DOFs (x, y, z rotation)
                for dof_offset in range(3):
                    dof_idx = joint_idx * 3 + dof_offset
                    if dof_idx < self.num_dof:
                        lower_body_dof_indices.append(dof_idx)
        
        if not lower_body_dof_indices:
            print("Warning: No lower body DOFs found for weakness modification")
            return
            
        # Modify DOF properties for recipient environments (odd env_ids)
        modified_count = 0
        for env_id in range(self.num_envs):
            if env_id % 2 == 1:  # Recipient environment
                env_ptr = self.envs[env_id]
                humanoid_handle = self.humanoid_handles[env_id]
                
                # Get current DOF properties
                dof_prop = self.gym.get_actor_dof_properties(env_ptr, humanoid_handle)
                
                # Modify lower body DOF stiffness and damping
                for dof_idx in lower_body_dof_indices:
                    dof_prop['stiffness'][dof_idx] *= recipient_weakness_scale
                    dof_prop['damping'][dof_idx] *= recipient_weakness_scale
                    dof_prop['effort'][dof_idx] = recipient_weakness_effort
                # Apply modified properties
                self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
                modified_count += 1
                
        print(f"Applied recipient weakness (scale={recipient_weakness_scale}) to {modified_count} recipients, affecting {len(lower_body_dof_indices)} lower body DOFs")

    def _compute_caregiver_torque_reduction_reward(self):
        """
        Compute reward for caregivers based on their partner recipient's torque magnitude.
        Lower recipient torque = higher reward (exponential decay).

        Returns:
            reward: Tensor of shape [num_envs] with rewards only for caregiver environments
        """
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Get all environment IDs
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        caregiver_mask = all_env_ids % 2 == 0

        # Get caregiver and recipient IDs
        caregiver_ids = all_env_ids[caregiver_mask]
        recipient_ids = caregiver_ids + 1

        # Filter out invalid pairs (where recipient_id >= num_envs)
        valid_pairs = recipient_ids < self.num_envs
        caregiver_ids = caregiver_ids[valid_pairs]
        recipient_ids = recipient_ids[valid_pairs]

        if len(caregiver_ids) == 0:
            return reward

        # Get current recipient torques from dof_force_tensor [num_recipients, num_dof]
        recipient_torques = self.dof_force_tensor[recipient_ids]

        # Compute total absolute torque for each recipient [num_recipients]
        # Use sum instead of mean to capture total effort across all DOFs
        total_torque = torch.sum(torch.abs(recipient_torques), dim=1)

        # Exponential decay reward: reward = coeff * exp(-torque / scale)
        # When torque=0: reward=coeff (maximum)
        # When torque=scale: reward=coeff * exp(-1) ≈ 0.37 * coeff
        # When torque=2*scale: reward=coeff * exp(-2) ≈ 0.14 * coeff
        torque_reward = self.caregiver_torque_reduction_coefficient * torch.exp(-total_torque / self.caregiver_torque_reduction_scale)
        # Assign rewards to caregiver environments
        reward[caregiver_ids] = torque_reward
        return reward




    def _compute_masked_imitation_reward(self, root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, 
                                       ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, 
                                       hand_masks, rwd_specs, rebase_flg=False, env_ids=None):
        """Compute imitation reward with optional masking of finger/wrist tracking"""
        k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
        w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]
        
        # Partner relative position reward parameters
        k_partner_rel_pos = rwd_specs.get("k_partner_rel_pos", self.cfg["env"].get("partner_rel_pos_k", 4.0))
        w_partner_rel_pos = rwd_specs.get("w_partner_rel_pos", self.cfg["env"].get("partner_rel_pos_weight", 0.0))

        # Get finger/wrist body indices for masking
        finger_wrist_bodies = self._get_finger_wrist_body_indices()
        
        # body position reward
        diff_global_body_pos = ref_body_pos - body_pos

        # Apply masking to finger/wrist bodies when hands are close
        if len(finger_wrist_bodies) > 0:
            if torch.any(hand_masks):
                diff_global_body_pos[hand_masks, :, :][:, finger_wrist_bodies, :] = 0.0

        diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
        r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

        # body rotation reward
        diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))

        # Apply masking to finger/wrist bodies when hands are close (before angle conversion)
        if len(finger_wrist_bodies) > 0:
            if torch.any(hand_masks):
                diff_global_body_rot[hand_masks, :, :][:, finger_wrist_bodies, :] = torch.tensor([0, 0, 0, 1], device=diff_global_body_rot.device, dtype=diff_global_body_rot.dtype)

        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
        diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
        r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

        # body linear velocity reward
        diff_global_vel = ref_body_vel - body_vel

        # Apply masking to finger/wrist bodies when hands are close
        if len(finger_wrist_bodies) > 0:
            if torch.any(hand_masks):
                diff_global_vel[hand_masks, :, :][:, finger_wrist_bodies, :] = 0.0

        diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
        r_vel = torch.exp(-k_vel * diff_global_vel_dist)

        # body angular velocity reward
        diff_global_ang_vel = ref_body_ang_vel - body_ang_vel

        # Apply masking to finger/wrist bodies when hands are close
        if len(finger_wrist_bodies) > 0:
            if torch.any(hand_masks):
                diff_global_ang_vel[hand_masks, :, :][:, finger_wrist_bodies, :] = 0.0

        diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
        r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

        # Partner relative position reward
        r_partner_rel_pos = torch.ones_like(r_body_pos)
        if w_partner_rel_pos > 0.0 and env_ids is not None:
            try:
                # Get partner reference states
                partner_ref_body_pos, _ = self._get_partner_reference_states(env_ids, ref_body_pos, ref_body_rot)
                
                # Get partner actual body positions
                partner_actual_body_pos = self._get_partner_actual_body_positions(env_ids)
                
                # Compute partner relative position reward
                r_partner_rel_pos = self._compute_partner_relative_pos_reward(
                    body_pos, ref_body_pos, partner_ref_body_pos, partner_actual_body_pos, k_partner_rel_pos, env_ids
                )
            except Exception:
                error_msg = f"Error computing partner relative position reward"
                raise Exception(error_msg)
        else:
            r_partner_rel_pos = torch.zeros_like(r_body_pos)
            

        reward = (w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel + 
                 w_partner_rel_pos * r_partner_rel_pos)
        reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel, r_partner_rel_pos], dim=-1)
        
        return reward, reward_raw

    def _get_finger_wrist_body_indices(self):
        """Get body indices for finger and wrist bodies that should be masked"""
        finger_wrist_names = [
            'L_Wrist','R_Wrist',
        ]
        
        finger_wrist_indices = []
        for body_name in finger_wrist_names:
            if body_name in self._body_name_to_id:
                # Convert global body ID to tracked body index
                global_body_id = self._body_name_to_id[body_name]
                if hasattr(self, '_track_bodies_id'):
                    # Find index in tracked bodies
                    try:
                        tracked_idx = self._track_bodies_id.tolist().index(global_body_id)
                        finger_wrist_indices.append(tracked_idx)
                    except ValueError:
                        # Body not in tracked list, use global ID
                        finger_wrist_indices.append(global_body_id)
                else:
                    finger_wrist_indices.append(global_body_id)
        
        return finger_wrist_indices

    def _get_partner_reference_states(self, env_ids, ref_body_pos, ref_body_rot):
        """Get partner's reference states for relative position calculation"""
        # Simple approach: even env_id -> +1, odd env_id -> -1
        partner_env_ids = get_partner_env_ids(env_ids)
        
        # Create mapping from env_id to batch index
        env_to_idx = {env_id.item(): i for i, env_id in enumerate(env_ids)}
        
        partner_ref_body_pos = torch.zeros_like(ref_body_pos)
        partner_ref_body_rot = torch.zeros_like(ref_body_rot)
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        
        for i, partner_env_id in enumerate(partner_env_ids):
            partner_env_id_val = partner_env_id.item() if hasattr(partner_env_id, 'item') else partner_env_id
            partner_idx = env_to_idx.get(partner_env_id_val)
            if partner_idx is not None:
                partner_ref_body_pos[i] = ref_body_pos[partner_idx].clone()
                partner_ref_body_rot[i] = ref_body_rot[partner_idx]
                
                # Apply offset compensation if partner is recipient
                if partner_env_id_val % 2 == 1:  # Partner is recipient
                    motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], 
                                                     device=self.device, dtype=partner_ref_body_pos.dtype)
                    partner_ref_body_pos[i] = partner_ref_body_pos[i] + motion_cache_offset.unsqueeze(0)
        
        return partner_ref_body_pos, partner_ref_body_rot

    def _compute_partner_relative_pos_reward(self, body_pos, ref_body_pos, partner_ref_body_pos, 
                                           partner_actual_body_pos, k_partner_rel_pos, env_ids):
        """Compute reward for partner-relative position consistency"""
        # Apply offset compensation to self body positions for recipients
        corrected_body_pos = body_pos.clone()
        corrected_ref_body_pos = ref_body_pos.clone()
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        
        for i, env_id in enumerate(env_ids):
            env_id_val = env_id.item() if hasattr(env_id, 'item') else env_id
            if env_id_val % 2 == 1:  # Current env is recipient
                motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], 
                                                 device=self.device, dtype=body_pos.dtype)
                corrected_body_pos[i] = corrected_body_pos[i] + motion_cache_offset.unsqueeze(0)
                corrected_ref_body_pos[i] = corrected_ref_body_pos[i] + motion_cache_offset.unsqueeze(0)
        
        # Calculate reference relative positions (my_ref_body - partner_ref_root)
        partner_ref_root = partner_ref_body_pos[:, 0:1, :]  # [batch, 1, 3]
        ref_rel_pos = corrected_ref_body_pos - partner_ref_root  # [batch, num_bodies, 3]
        
        # Calculate actual relative positions (my_corrected_body - partner_actual_root)
        partner_actual_root = partner_actual_body_pos[:, 0:1, :]  # [batch, 1, 3]
        actual_rel_pos = corrected_body_pos - partner_actual_root  # [batch, num_bodies, 3]
        
        # Calculate difference in relative positions
        diff_rel_pos = ref_rel_pos - actual_rel_pos  # [batch, num_bodies, 3]
        diff_rel_pos_dist = (diff_rel_pos**2).mean(dim=-1).mean(dim=-1)  # [batch]
        
        # Compute reward
        r_partner_rel_pos = torch.exp(-k_partner_rel_pos * diff_rel_pos_dist)
        
        return r_partner_rel_pos

    def _get_partner_actual_body_positions(self, env_ids):
        """Get partner's current actual body positions"""
        # Simple approach: even env_id -> +1, odd env_id -> -1
        partner_env_ids = get_partner_env_ids(env_ids)
        
        partner_body_pos_list = []
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        
        for i, env_id in enumerate(env_ids):
            env_id_val = env_id.item() if hasattr(env_id, 'item') else env_id
            partner_env_id_val = partner_env_ids[i].item() if hasattr(partner_env_ids[i], 'item') else partner_env_ids[i]
            
            if partner_env_id_val < self.num_envs and partner_env_id_val >= 0:
                # Get partner's body positions with offset compensation
                partner_body_pos = self._rigid_body_pos[partner_env_id_val, :, :3]
                
                # Apply motion cache offset compensation
                if partner_env_id_val % 2 == 1:  # Partner is recipient
                    motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], 
                                                     device=self.device, dtype=partner_body_pos.dtype)
                    partner_body_pos = partner_body_pos + motion_cache_offset.unsqueeze(0)

            else:
                error_msg = f"Partner env_id {partner_env_id_val} out of bounds"
                raise Exception(error_msg)
            
            partner_body_pos_list.append(partner_body_pos)
        
        return torch.stack(partner_body_pos_list)



    def _sample_ref_state(self, env_ids):
        """Override to apply consistent motion initialization in SimpleLiftUp mode"""
        # Get parent class behavior first
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = super()._sample_ref_state(env_ids)
        
        # In SimpleLiftUp mode, use trajectory buffer or start from time 0
        if self.simple_lift_up_mode:
            # Try to use trajectory buffer for initialization (disabled during evaluation)
            used_mask = None
            # Check if we're in evaluation mode (test flag or periodic validation)
            is_validation = self._is_validation_step()
            if not flags.test and not self.evaluation_mode and not is_validation:  # Only use RSI during training
                used_mask = self._sample_from_trajectory_buffer(env_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel)

            if used_mask is None or (~used_mask).all():
                # No buffer usage: start all from time 0
                motion_times[:] = 0

                # Re-compute motion state with time 0 for consistency
                if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                    motion_res = self._get_state_from_motionlib_cache(motion_ids, motion_times, self._global_offset[env_ids])
                    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel = \
                        motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"]
                    ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                        motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
                else:
                    root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(motion_ids, motion_times)
            else:
                # Partial buffer usage: only reset non-used envs to time 0 and recompute their states
                not_used_idx = torch.nonzero(~used_mask, as_tuple=False).squeeze(-1)
                if not_used_idx.numel() > 0:
                    motion_times[not_used_idx] = 0
                    if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                        motion_res = self._get_state_from_motionlib_cache(motion_ids, motion_times, self._global_offset[env_ids])
                        root_pos[not_used_idx] = motion_res["root_pos"][not_used_idx]
                        root_rot[not_used_idx] = motion_res["root_rot"][not_used_idx]
                        dof_pos[not_used_idx] = motion_res["dof_pos"][not_used_idx]
                        root_vel[not_used_idx] = motion_res["root_vel"][not_used_idx]
                        root_ang_vel[not_used_idx] = motion_res["root_ang_vel"][not_used_idx]
                        dof_vel[not_used_idx] = motion_res["dof_vel"][not_used_idx]
                        if ref_rb_pos is not None:
                            ref_rb_pos[not_used_idx] = motion_res["rg_pos"][not_used_idx]
                        if ref_rb_rot is not None:
                            ref_rb_rot[not_used_idx] = motion_res["rb_rot"][not_used_idx]
                        if ref_body_vel is not None:
                            ref_body_vel[not_used_idx] = motion_res["body_vel"][not_used_idx]
                        if ref_body_ang_vel is not None:
                            ref_body_ang_vel[not_used_idx] = motion_res["body_ang_vel"][not_used_idx]
                    else:
                        rpos, rrot, dpos, rvel, ravel, dvel, _ = self._motion_lib.get_motion_state(motion_ids, motion_times)
                        root_pos[not_used_idx] = rpos[not_used_idx]
                        root_rot[not_used_idx] = rrot[not_used_idx]
                        dof_pos[not_used_idx] = dpos[not_used_idx]
                        root_vel[not_used_idx] = rvel[not_used_idx]
                        root_ang_vel[not_used_idx] = ravel[not_used_idx]
                        dof_vel[not_used_idx] = dvel[not_used_idx]
        
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel

    def create_sim(self):
        """Override to apply recipient mass scaling for SimpleLiftUp mode"""
        # Call parent method first
        super().create_sim()
        
        # Apply recipient mass scaling if in SimpleLiftUp mode
        if self.simple_lift_up_mode and hasattr(self, 'recipient_mass_scale') and self.recipient_mass_scale != 1.0:
            print(f"[_create_sim] SimpleLiftUp mode detected - applying mass scaling")
            self._apply_recipient_mass_scaling()
        else:
            print(f"[_create_sim] Mass scaling not applied - simple_lift_up_mode: {self.simple_lift_up_mode}, recipient_mass_scale: {getattr(self, 'recipient_mass_scale', 'NOT_SET')}")
        
        return
    
    
    def _apply_recipient_mass_scaling(self):
        """Apply mass scaling to recipient humanoids to make them lighter"""
        # print(f"[MASS_SCALING] Applying recipient mass scaling: {self.recipient_mass_scale}")
        # print(f"[MASS_SCALING] Test mode: {getattr(__import__('phc.utils.flags'), 'flags', type('', (), {'test': 'UNKNOWN'})).test}")
        
        # Scale mass for recipient environments
        for env_id in range(self.num_envs):
            role = self.env_role_assignments.get(env_id, 'unknown')
            if env_id % 2 == 1:
                env_ptr = self.envs[env_id]
                humanoid_handle = self.humanoid_handles[env_id]
                
                # Get current rigid body properties for this actor
                rigid_props = self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)
                
                # Scale mass for all body parts
                for i, prop in enumerate(rigid_props):
                    original_mass = prop.mass
                    prop.mass = original_mass * self.recipient_mass_scale
                    # print(f"Env {env_id}, body {i}: {original_mass:.3f} -> {prop.mass:.3f}")
                    # print(f"Env {env_id}, body {i}: {original_mass:.3f} -> {prop.mass:.3f}")
                
                # Apply the modified properties back to the actor
                self.gym.set_actor_rigid_body_properties(env_ptr, humanoid_handle, rigid_props)
        # print(f"[MASS_SCALING] Mass scaling applied successfully")
        # assert False
        return 


    def _get_aux_features_size(self):
        """Calculate auxiliary features size for SimpleLiftUp mode"""
        if not hasattr(self, 'cfg') or self.cfg is None:
            return 0
            
        self_obs_v = self.cfg["env"].get("self_obs_v", 1)
        partner_obs_v = self.cfg["env"].get("partner_obs_v", 1)
        
        # Calculate self contact and force size
        self_contact_size = 0
        self_force_size = 0
        if self_obs_v == 4:
            self_contact_size = self._get_hand_contact_obs_size()
        elif self_obs_v == 5:
            self_contact_size = self._get_hand_contact_obs_size()
            self_force_size = self._get_hand_force_obs_size()
        
        # Aux features: [self_contact, base_partner_obs, partner_contact, reference_contact_diff, role_flag]
        if partner_obs_v == 3:
            # V3: base partner obs + optional wrist-relative partner obs, separate partner contact
            base_partner_size = self._get_partner_obs_size_v2()
            
            # Check if wrist-relative features are enabled
            enable_wrist_relative = self.cfg["env"].get("enable_wrist_relative_obs", True)
            if enable_wrist_relative:
                base_partner_size += self._get_wrist_relative_partner_obs_size()
                
            partner_contact_size = self._get_hand_contact_obs_size()
        else:
            # V1/V2: partner obs without contact  
            base_partner_size = self._get_partner_obs_size_v2()
            partner_contact_size = 0
        
        # Calculate future trajectory sizes (reduced from 60 to 30 frames, 30 hand joints removed)
        # Each trajectory: 30 frames * (22 bodies * 13 features) = 30 * 22 * 13 = 8580 features per trajectory
        # Two trajectories: self + partner = 8580 * 2 = 17160
        # Calculate non-hand bodies dynamically based on humanoid type
        if hasattr(self, 'num_bodies'):
            humanoid_type = getattr(self, 'humanoid_type', 'smpl')
            print(f"[DEBUG] num_bodies: {self.num_bodies}, humanoid_type: {humanoid_type}")
            if humanoid_type == 'smplx':
                # Force to 22 bodies to match network expectation
                non_hand_bodies = 22  # Fixed to match PNN network
            else:
                non_hand_bodies = self.num_bodies  # SMPL/SMPLH don't have hand joints
        else:
            # Fallback values during early initialization
            humanoid_type = getattr(self, 'humanoid_type', 'smpl')
            if humanoid_type == 'smplx':
                non_hand_bodies = 22  # Fixed to match PNN network expectation
            else:
                non_hand_bodies = 23  # SMPL default
        print(f"[DEBUG] Using non_hand_bodies: {non_hand_bodies} for trajectory calculation")
        future_trajectory_size = self.future_size * non_hand_bodies * 13 * 2  # 2 for self+partner trajectories

        # Get action size for previous actions
        action_size = getattr(self, 'num_actions', 153)  # Default to 153 if not set yet

        # aux_features = [self_contact, self_force, base_partner_obs, partner_contact, self_future_traj_flat, partner_future_traj_flat, prev_actions, role_label]
        aux_features_size = self_contact_size + self_force_size + base_partner_size + partner_contact_size + future_trajectory_size + action_size + 1  # +action_size for prev_actions, +1 for role label (at the end)

        return aux_features_size

    def _sample_time(self, motion_ids):
        # 常にシーケンスのt=0から開始（訓練時もテスト時も同じ動作）
        return torch.zeros(motion_ids.shape, device=self.device)
    
    def post_physics_step(self):
        """Override to add trajectory tracking and torque collection"""
        # Call parent implementation
        super().post_physics_step()

        # Only capture trajectories if RSI is enabled (trajectory_tracking_pairs > 0)
        if hasattr(self, 'trajectory_tracking_env_ids') and len(self.trajectory_tracking_env_ids) > 0:
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._capture_trajectory_step(all_env_ids)
            
        # Track recipient torques at each step during testing
        if flags.test and flags.im_eval:
            self._track_recipient_torques()
    
    def _reset_envs(self, env_ids):
        """Override to start trajectory tracking for reset environments"""
        # Save torque data before reset during testing
        # if flags.test and flags.im_eval:
        #     self._save_torque_data_for_reset_envs(env_ids)
        
        # Call parent implementation first
        super()._reset_envs(env_ids)


        # Reset previous actions for reset environments
        self.prev_actions[env_ids] = 0

        # Update env_id to motion_unique_id mapping at episode start
        self._update_env_motion_mapping(env_ids)

        # Start trajectory tracking for reset environments
        self._start_trajectory_tracking(env_ids)

    def store_successful_trajectories(self, env_ids, episode_rewards):
        """Store successful episode trajectories to buffer"""
        self._save_successful_trajectories(env_ids, episode_rewards)
    
    def set_evaluation_mode(self, is_evaluation):
        """
        Set evaluation mode to control trajectory buffer usage.
        
        Args:
            is_evaluation (bool): True to disable trajectory buffer, False to enable
        """
        self.evaluation_mode = is_evaluation
        
    def get_evaluation_mode(self):
        """
        Get current evaluation mode.
        
        Returns:
            bool: True if in evaluation mode, False otherwise
        """
        return self.evaluation_mode
    
    def set_training_agent(self, agent):
        """
        Set reference to training agent for accessing epoch_num.
        
        Args:
            agent: Training agent instance with epoch_num attribute
        """
        self._training_agent = agent
    
    def _is_validation_step(self):
        """
        Check if we're currently in a validation step based on epoch_num.
        
        Returns:
            bool: True if this is a validation step (epoch_num % save_frequency == 0)
        """
        if self._training_agent is None or not hasattr(self._training_agent, 'epoch_num'):
            return False
        try:
            epoch_num = self._training_agent.epoch_num
            # Validation occurs at save_frequency intervals
            return epoch_num > 0 and epoch_num % self._save_frequency == 0
        except Exception as e:
            raise e


    def _extract_future_trajectories(self, env_ids, future_frames):
        """
        Extract future reference motion trajectories for the specified environments.
        Uses efficient vectorized operations to avoid for-loops.

        Args:
            env_ids: Environment IDs to extract trajectories for
            future_frames: Number of future frames to extract

        Returns:
            torch.Tensor: [batch_size, future_frames, feature_dim] future trajectory features
        """
        batch_size = len(env_ids)
        device = self.device
        dt = self.dt

        # Create time offsets including current frame (t to t+future_frames-1)
        future_time_offsets = torch.arange(0, future_frames, device=device, dtype=torch.float32) * dt

        # Expand for all environments: [batch_size, future_frames]
        future_time_offsets = future_time_offsets.unsqueeze(0).expand(batch_size, -1)

        # Current motion times for each environment
        current_times = (
            self.progress_buf[env_ids] * dt +
            self._motion_start_times[env_ids] +
            self._motion_start_times_offset[env_ids]
        )

        # Future motion times: [batch_size, future_frames]
        future_motion_times = current_times.unsqueeze(1) + future_time_offsets

        # Flatten for batch processing
        future_motion_times_flat = future_motion_times.flatten()
        motion_ids_flat = self._sampled_motion_ids[env_ids].repeat_interleave(future_frames)
        offsets_flat = self._global_offset[env_ids].repeat_interleave(future_frames, dim=0)

        
        offsets_flat = offsets_flat 

        # Get motion states for all future time points with offset applied
        motion_res = self._motion_lib.get_motion_state(motion_ids_flat, future_motion_times_flat, offset=offsets_flat)

        # Create filtered track_bodies_id excluding 30 hand-related joints
        if not hasattr(self, '_non_hand_track_bodies_id'):
            # SMPL-X hand joints to exclude (15 per hand × 2 hands = 30 joints)
            hand_joints = [
                # Left hand joints (15)
                'L_Index1', 'L_Index2', 'L_Index3',
                'L_Middle1', 'L_Middle2', 'L_Middle3',
                'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
                'L_Ring1', 'L_Ring2', 'L_Ring3',
                'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
                # Right hand joints (15)
                'R_Index1', 'R_Index2', 'R_Index3',
                'R_Middle1', 'R_Middle2', 'R_Middle3',
                'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
                'R_Ring1', 'R_Ring2', 'R_Ring3',
                'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
            ]

            # Find indices of hand joints in _track_bodies_id
            hand_indices_to_remove = []
            for joint_name in hand_joints:
                if joint_name in self._body_name_to_id:
                    body_id = self._body_name_to_id[joint_name]
                    if body_id in self._track_bodies_id:
                        track_idx = self._track_bodies_id.tolist().index(body_id)
                        hand_indices_to_remove.append(track_idx)

            # Create mask for non-hand joints
            num_track_bodies = len(self._track_bodies_id)
            non_hand_mask = torch.ones(num_track_bodies, dtype=torch.bool, device=self.device)
            non_hand_mask[hand_indices_to_remove] = False

            # Create filtered track_bodies_id
            self._non_hand_track_bodies_id = self._track_bodies_id[non_hand_mask]
            self._hand_filter_mask = non_hand_mask
            # # Debug: Print filtering results
            # print(f"DEBUG: Original track_bodies count: {len(self._track_bodies_id)}")
            # print(f"DEBUG: Hand joints to remove: {len(hand_indices_to_remove)} indices: {hand_indices_to_remove}")
            # print(f"DEBUG: Remaining non-hand bodies count: {len(self._non_hand_track_bodies_id)}")
            # print(f"DEBUG: Non-hand body IDs: {self._non_hand_track_bodies_id.tolist()}")
            # if len(self._non_hand_track_bodies_id) < 5:  # Print body names if small number
            #     body_names = [list(self._body_name_to_id.keys())[list(self._body_name_to_id.values()).index(body_id.item())] for body_id in self._non_hand_track_bodies_id]
            #     print(f"DEBUG: Non-hand body names: {body_names}")
            # assert False

        # Use filtered track_bodies_id for trajectory features
        track_bodies_id = self._non_hand_track_bodies_id
        hand_filter_mask = self._hand_filter_mask

        # Extract features similar to task_obs computation: rg_pos, rb_rot, body_vel, body_ang_vel
        # These are the rigid body positions, rotations, velocities used in task observations
        features_list = []

        # Rigid body positions (rg_pos) - filtered subset excluding hand joints
        ref_rb_pos_subset = motion_res["rg_pos"][..., track_bodies_id, :]  # [batch*frames, num_non_hand_bodies, 3]
        ref_rb_pos_subset_flat = ref_rb_pos_subset.reshape(batch_size * future_frames, -1)  # [batch*frames, num_non_hand_bodies*3]
        features_list.append(ref_rb_pos_subset_flat)

        # Rigid body rotations (rb_rot) - filtered subset excluding hand joints
        ref_rb_rot_subset = motion_res["rb_rot"][..., track_bodies_id, :]  # [batch*frames, num_non_hand_bodies, 4]
        ref_rb_rot_subset_flat = ref_rb_rot_subset.reshape(batch_size * future_frames, -1)  # [batch*frames, num_non_hand_bodies*4]
        features_list.append(ref_rb_rot_subset_flat)

        # Body linear velocities (body_vel) - filtered subset excluding hand joints
        ref_body_vel_subset = motion_res["body_vel"][..., track_bodies_id, :]  # [batch*frames, num_non_hand_bodies, 3]
        ref_body_vel_subset_flat = ref_body_vel_subset.reshape(batch_size * future_frames, -1)  # [batch*frames, num_non_hand_bodies*3]
        features_list.append(ref_body_vel_subset_flat)

        # Body angular velocities (body_ang_vel) - filtered subset excluding hand joints
        ref_body_ang_vel_subset = motion_res["body_ang_vel"][..., track_bodies_id, :]  # [batch*frames, num_non_hand_bodies, 3]
        ref_body_ang_vel_subset_flat = ref_body_ang_vel_subset.reshape(batch_size * future_frames, -1)  # [batch*frames, num_non_hand_bodies*3]
        features_list.append(ref_body_ang_vel_subset_flat)

        # Concatenate all features
        future_features_flat = torch.cat(features_list, dim=1)  # [batch_size * future_frames, feature_dim]

        # Reshape to [batch_size, future_frames, feature_dim]
        feature_dim = future_features_flat.shape[1]
        future_features = future_features_flat.view(batch_size, future_frames, feature_dim)

        # Clean up intermediate tensors to prevent memory accumulation
        del motion_res, features_list, future_features_flat
        del ref_rb_pos_subset, ref_rb_pos_subset_flat
        del ref_rb_rot_subset, ref_rb_rot_subset_flat
        del ref_body_vel_subset, ref_body_vel_subset_flat
        del ref_body_ang_vel_subset, ref_body_ang_vel_subset_flat
        del future_time_offsets, future_motion_times, future_motion_times_flat
        del motion_ids_flat, offsets_flat

        return future_features

    def _extract_partner_future_trajectories(self, env_ids, future_frames):
        """
        Extract future reference motion trajectories for partner environments.
        Uses vectorized operations to avoid for-loops for better performance.

        Args:
            env_ids: Environment IDs to extract partner trajectories for
            future_frames: Number of future frames to extract

        Returns:
            torch.Tensor: [batch_size, future_frames, feature_dim] partner future trajectory features
        """
        # Find partner environment IDs using vectorized operations
        # Convert env_ids to tensor if needed
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # Vectorized partner ID computation
        # For even IDs: partner = id + 1, For odd IDs: partner = id - 1
        is_even = (env_ids % 2) == 0
        partner_env_ids = torch.where(is_even, env_ids + 1, env_ids - 1)

        # Ensure all partner IDs are valid (< num_envs), otherwise use self
        valid_partner_mask = partner_env_ids < self.num_envs
        partner_env_ids = torch.where(valid_partner_mask, partner_env_ids, env_ids)

        # Extract future trajectories for partners
        partner_future_trajectories = self._extract_future_trajectories(partner_env_ids, future_frames)

        return partner_future_trajectories

    def _normalize_trajectories_to_current_frame(self, trajectories, env_ids, target_env_ids):
        """
        Normalize future trajectory data relative to current self root position and orientation.
        This makes the trajectory data invariant to the current position/orientation in the world.

        Args:
            trajectories: [batch_size, future_frames, feature_dim] trajectory data
            env_ids: Environment IDs for current frame reference
            target_env_ids: Environment IDs for the trajectories (can be different for partner trajectories)

        Returns:
            torch.Tensor: [batch_size, future_frames, feature_dim] normalized trajectory data
        """
        batch_size, future_frames, feature_dim = trajectories.shape
        device = self.device


        # Get current self root position and rotation for normalization reference
        current_root_pos = self._rigid_body_pos[env_ids, 0, :]  # [batch_size, 3]
        current_root_rot = self._rigid_body_rot[env_ids, 0, :]  # [batch_size, 4]

        # Calculate actual number of bodies from feature_dim
        # Each body has: pos(3) + rot(4) + vel(3) + ang_vel(3) = 13 features per body
        features_per_body = 13
        actual_num_bodies = feature_dim // features_per_body

        # Use actual calculated number instead of hardcoded 22
        non_hand_bodies = actual_num_bodies
        pos_dim = non_hand_bodies * 3
        rot_dim = non_hand_bodies * 4
        vel_dim = non_hand_bodies * 3
        ang_vel_dim = non_hand_bodies * 3

        # Verify dimensions add up correctly
        expected_feature_dim = pos_dim + rot_dim + vel_dim + ang_vel_dim
        assert expected_feature_dim == feature_dim, f"Feature dimension calculation error: expected {expected_feature_dim}, got {feature_dim}"

        # Split trajectory features
        pos_end = pos_dim
        rot_end = pos_end + rot_dim
        vel_end = rot_end + vel_dim
        ang_vel_end = vel_end + ang_vel_dim

        traj_positions = trajectories[:, :, :pos_end]  # [batch, frames, pos_dim]
        traj_rotations = trajectories[:, :, pos_end:rot_end]  # [batch, frames, rot_dim]
        traj_velocities = trajectories[:, :, rot_end:vel_end]  # [batch, frames, vel_dim]
        traj_ang_velocities = trajectories[:, :, vel_end:ang_vel_end]  # [batch, frames, ang_vel_dim]

        # Reshape for processing: [batch, frames, num_bodies, 3/4]
        traj_pos_reshaped = traj_positions.view(batch_size, future_frames, non_hand_bodies, 3)
        traj_rot_reshaped = traj_rotations.view(batch_size, future_frames, non_hand_bodies, 4)
        traj_vel_reshaped = traj_velocities.view(batch_size, future_frames, non_hand_bodies, 3)
        traj_ang_vel_reshaped = traj_ang_velocities.view(batch_size, future_frames, non_hand_bodies, 3)

        # Compute inverse of current root rotation for transformation
        current_root_rot_inv = quat_conjugate(current_root_rot)  # [batch_size, 4]

        # Normalize positions: translate to current root, then rotate to current frame
        # Shape expansions for broadcasting
        current_root_pos_expanded = current_root_pos.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, 3]
        current_root_rot_inv_expanded = current_root_rot_inv.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, 4]

        # Translate positions relative to current root
        traj_pos_translated = traj_pos_reshaped - current_root_pos_expanded

        # Rotate positions to current frame orientation
        traj_pos_normalized = quat_apply(
            current_root_rot_inv_expanded.expand(-1, future_frames, non_hand_bodies, -1).reshape(-1, 4),
            traj_pos_translated.reshape(-1, 3)
        ).view(batch_size, future_frames, non_hand_bodies, 3)


        # Normalize rotations: multiply by inverse of current root rotation
        traj_rot_normalized = quat_mul(
            current_root_rot_inv_expanded.expand(-1, future_frames, non_hand_bodies, -1).reshape(-1, 4),
            traj_rot_reshaped.reshape(-1, 4)
        ).view(batch_size, future_frames, non_hand_bodies, 4)

        # Normalize linear velocities: rotate to current frame
        traj_vel_normalized = quat_apply(
            current_root_rot_inv_expanded.expand(-1, future_frames, non_hand_bodies, -1).reshape(-1, 4),
            traj_vel_reshaped.reshape(-1, 3)
        ).view(batch_size, future_frames, non_hand_bodies, 3)

        # Normalize angular velocities: rotate to current frame
        traj_ang_vel_normalized = quat_apply(
            current_root_rot_inv_expanded.expand(-1, future_frames, non_hand_bodies, -1).reshape(-1, 4),
            traj_ang_vel_reshaped.reshape(-1, 3)
        ).view(batch_size, future_frames, non_hand_bodies, 3)

        # Reshape back to flat feature format
        traj_pos_flat = traj_pos_normalized.view(batch_size, future_frames, pos_dim)
        traj_rot_flat = traj_rot_normalized.view(batch_size, future_frames, rot_dim)
        traj_vel_flat = traj_vel_normalized.view(batch_size, future_frames, vel_dim)
        traj_ang_vel_flat = traj_ang_vel_normalized.view(batch_size, future_frames, ang_vel_dim)

        # Concatenate normalized features back together
        normalized_trajectories = torch.cat([
            traj_pos_flat, traj_rot_flat, traj_vel_flat, traj_ang_vel_flat
        ], dim=2)  # [batch_size, future_frames, feature_dim]

        # Clean up intermediate tensors to prevent memory accumulation
        del traj_positions, traj_rotations, traj_velocities, traj_ang_velocities
        del traj_pos_reshaped, traj_rot_reshaped, traj_vel_reshaped, traj_ang_vel_reshaped
        del traj_pos_translated, traj_pos_normalized, traj_rot_normalized, traj_vel_normalized, traj_ang_vel_normalized
        del traj_pos_flat, traj_rot_flat, traj_vel_flat, traj_ang_vel_flat
        del current_root_pos_expanded, current_root_rot_inv_expanded, current_root_rot_inv

        return normalized_trajectories
    
    def _track_recipient_torques(self):
        """Track recipient torques at each step during testing"""
        # Get recipient environment IDs (odd indices)
        recipient_env_ids = torch.arange(1, self.num_envs, 2, device=self.device)
        
        # Get DOF forces (torques) for recipient environments
        if len(recipient_env_ids) > 0:
            recipient_torques = self.dof_force_tensor[recipient_env_ids].cpu().numpy()
            
            # Store torques for each recipient environment
            for i, env_id in enumerate(recipient_env_ids):
                torque_data = recipient_torques[i]  # All joint torques for this recipient
                
                # Get current motion information
                current_motion_id = None
                motion_key = None
                if hasattr(self._motion_lib, '_curr_motion_ids') and self._motion_lib._curr_motion_ids is not None:
                    if env_id.item() < len(self._motion_lib._curr_motion_ids):
                        current_motion_id = self._motion_lib._curr_motion_ids[env_id.item()].item()
                        
                        # Get motion key from current motion ID
                        if hasattr(self._motion_lib, '_motion_data_keys') and current_motion_id < len(self._motion_lib._motion_data_keys):
                            motion_key = self._motion_lib._motion_data_keys[current_motion_id]
                
                self.recipient_torque_buffer.append({
                    'env_id': env_id.item(),
                    'step': self.progress_buf[env_id].item(),
                    'torques': torque_data.copy(),
                    'motion_id': current_motion_id,
                    'motion_key': motion_key
                })
    
    def _save_torque_data_for_reset_envs(self, env_ids):
        """Save torque data for environments that are about to reset"""
        for env_id in env_ids:
            # Only save for recipient environments (odd indices)
            if env_id % 2 == 1:
                if len(self.recipient_torque_buffer) > 0:
                    # Filter torque data for this environment
                    env_torque_data = [data for data in self.recipient_torque_buffer if data['env_id'] == env_id.item()]
                    
                    if len(env_torque_data) > 0:
                        # Get motion filename from the stored motion information
                        motion_filename = get_motion_filename_from_data(env_torque_data[0], env_id)
                        
                        # Convert to numpy array [num_steps, num_joints]
                        torque_timeseries = np.array([data['torques'] for data in env_torque_data])
                        
                        # Save to file only if we have significant data (more than just 1-2 steps)
                        if torque_timeseries.shape[0] > 5:  # Only save if episode lasted more than 5 steps
                            import os
                            import time
                            # Use exp_name from config if available, otherwise use default
                            exp_name = getattr(self.cfg, 'exp_name', 'AA-RM-wo-FullAssist-v4')
                            output_dir = f"output/{exp_name}"
                            os.makedirs(output_dir, exist_ok=True)
                            
                            # Add timestamp to ensure unique filenames
                            timestamp = int(time.time() * 1000) % 100000  # Last 5 digits of milliseconds
                            save_path = os.path.join(output_dir, f"{motion_filename}_{timestamp}_torque_timeseries.npy")
                            np.save(save_path, torque_timeseries)
                            print(f"Saved torque data: {save_path} with shape {torque_timeseries.shape}")
                        
                        # Clear buffer for this environment
                        self.recipient_torque_buffer = [data for data in self.recipient_torque_buffer if data['env_id'] != env_id.item()]
