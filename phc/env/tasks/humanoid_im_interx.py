import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgym.torch_utils import quat_rotate_inverse, quat_conjugate, quat_mul, quat_apply
import joblib
from phc.utils.flags import flags
from phc.utils import torch_utils
import random

# Import the parent class
from phc.env.tasks.humanoid_im import HumanoidIm, compute_point_goal_reward

# Import RSI module
from phc.env.tasks.RSI import RSIMixin


class HumanoidImInterx(HumanoidIm, RSIMixin):
    """
    Humanoid Interaction class for single-humanoid per environment interaction tasks.
    Following the instruction: 1 environment = 1 humanoid, with collision groups for pair interaction.
    """

    # def pre_physics_step(self, actions):
    #     # Apply recipient action masking (Option 1 implementation)
    #     # Recipients (odd env_ids) should not take self-initiated actions
    #     for env_id in range(self.num_envs):
    #         if env_id % 2 == 1:  # Recipient environments
    #             actions[env_id] = torch.zeros_like(actions[env_id])
        
    #     # Call parent pre_physics_step with masked actions
    #     super().pre_physics_step(actions)
        
    #     return

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
        self.recipient_mass_scale = cfg["env"].get("recipient_mass_scale", 0.3)

        # Failed motion weighted sampling settings
        self.failed_motion_weight = cfg["env"].get("failed_motion_weight", False)
        self.failed_weight_multiplier = cfg["env"].get("failed_weight_multiplier", 2.0)
        
        print(f"SimpleLiftUp mode: {self.simple_lift_up_mode}")
        if self.simple_lift_up_mode:
            print(f"  - Task reward only: {self.task_reward_only}")
            print(f"  - Dense height reward: {self.dense_height_reward}")
            print(f"  - Recipient mass scale: {self.recipient_mass_scale}")

        print(f"Failed motion weighted sampling: {self.failed_motion_weight}")
        if self.failed_motion_weight:
            print(f"  - Failed weight multiplier: {self.failed_weight_multiplier}")

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
        self.min_required_hand_contacts = self.cfg["env"].get("min_required_hand_contacts", 10)  # Minimum contacts per episode
        self.zero_reward_on_poor_contact = self.cfg["env"].get("zero_reward_on_poor_contact", False)
        
        # Contact absence termination tracking
        self.no_contact_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.max_no_contact_steps = self.cfg["env"].get("max_no_contact_steps", 60)  # Terminate after 60 steps without contact
        
        # Hand distance reward configuration
        self.hand_target_flg = self.cfg["env"].get("hand_target_flg", False)  # False: nearest upper body joint, True: hand-to-hand
        
        # Static termination tracking (30 frames with joint movement < 1cm)
        self.static_frames_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.prev_joint_positions = None  # Will be initialized in first step
        self.STATIC_FRAME_THRESHOLD = 30  # 30 frames
        self.JOINT_MOVEMENT_THRESHOLD = 0.01  # 1cm
        
        if self.zero_reward_on_poor_contact:
            print(f"Zero reward on poor contact enabled. Min required contacts: {self.min_required_hand_contacts}")
        
        print(f"No contact termination: {self.max_no_contact_steps} steps")
        
        return

    def _load_motion(self, motion_train_file, motion_test_file=[]):
        """Override to setup interaction motion data after parent loads motion"""
        # Use the interaction data path if no specific motion file is provided
        if not motion_train_file or motion_train_file == "":
            motion_train_file = self.interx_data_path
        
        # Call parent method first
        super()._load_motion(motion_train_file, motion_test_file)
        
        # Setup interaction motion data after parent has loaded motion (with safety checks)
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
        self_future_trajectories = self._extract_future_trajectories(env_ids, future_frames=20)

        # Extract partner future trajectories (30 frames - 1 second)
        partner_future_trajectories = self._extract_partner_future_trajectories(env_ids, future_frames=20)

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

        # Create aux_features list with future trajectories
        aux_features_parts = [self_contact_obs]
        if self.cfg["env"].get("self_obs_v", 1) == 5:
            aux_features_parts.append(self_force_obs)
        aux_features_parts.extend([base_partner_obs, partner_contact_obs, self_future_trajectory_flat, partner_future_trajectory_flat, role_labels])

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
        partner_env_ids = torch.where(env_ids % 2 == 0, env_ids + 1, env_ids - 1)

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
            # V1: Relative joint positions (vectorized)
            # 1. Root relative position in current humanoid's frame
            rel_root_pos = partner_root_pos - current_root_pos  # [batch_size, 3]
            rel_root_pos_local = quat_rotate_inverse(current_root_rot, rel_root_pos)  # [batch_size, 3]

            # 2. Joint relative positions in current humanoid's frame
            rel_joint_pos = partner_body_pos - current_body_pos  # [batch_size, num_bodies, 3]
            batch_size, num_bodies, _ = rel_joint_pos.shape

            # Expand quaternions for batch processing
            current_root_rot_expanded = current_root_rot.unsqueeze(1).expand(-1, num_bodies, -1)  # [batch_size, num_bodies, 4]
            rel_joint_pos_flat = rel_joint_pos.reshape(-1, 3)  # [batch_size * num_bodies, 3]
            current_root_rot_flat = current_root_rot_expanded.reshape(-1, 4)  # [batch_size * num_bodies, 4]

            rel_joint_pos_local_flat = quat_rotate_inverse(current_root_rot_flat, rel_joint_pos_flat)
            rel_joint_pos_local = rel_joint_pos_local_flat.reshape(batch_size, -1)  # [batch_size, num_bodies * 3]

            # 3. Joint relative velocities in current humanoid's frame
            rel_joint_vel = partner_body_vel - current_body_vel  # [batch_size, num_bodies, 3]
            rel_joint_vel_flat = rel_joint_vel.reshape(-1, 3)  # [batch_size * num_bodies, 3]

            rel_joint_vel_local_flat = quat_rotate_inverse(current_root_rot_flat, rel_joint_vel_flat)
            rel_joint_vel_local = rel_joint_vel_local_flat.reshape(batch_size, -1)  # [batch_size, num_bodies * 3]

            # Combine partner observations
            partner_obs = torch.cat([
                rel_root_pos_local,      # [batch_size, 3]
                rel_joint_pos_local,     # [batch_size, num_bodies * 3]
                rel_joint_vel_local      # [batch_size, num_bodies * 3]
            ], dim=1)

        elif (partner_obs_v == 2) or (partner_obs_v == 3):
            # V2: Absolute partner positions/velocities in self coordinate frame (vectorized)
            # 1. Root position in current humanoid's frame
            rel_root_pos = partner_root_pos - current_root_pos  # [batch_size, 3]
            partner_root_pos_local = quat_rotate_inverse(current_root_rot, rel_root_pos)  # [batch_size, 3]

            # 2. All partner joint positions in current humanoid's frame
            partner_pos_relative = partner_body_pos - current_root_pos.unsqueeze(1)  # [batch_size, num_bodies, 3]
            batch_size, num_bodies, _ = partner_pos_relative.shape

            # Expand quaternions for batch processing
            current_root_rot_expanded = current_root_rot.unsqueeze(1).expand(-1, num_bodies, -1)  # [batch_size, num_bodies, 4]
            partner_pos_relative_flat = partner_pos_relative.reshape(-1, 3)  # [batch_size * num_bodies, 3]
            current_root_rot_flat = current_root_rot_expanded.reshape(-1, 4)  # [batch_size * num_bodies, 4]

            partner_pos_local_flat = quat_rotate_inverse(current_root_rot_flat, partner_pos_relative_flat)
            partner_pos_local = partner_pos_local_flat.reshape(batch_size, -1)  # [batch_size, num_bodies * 3]

            # 3. All partner joint velocities in current humanoid's frame
            partner_body_vel_flat = partner_body_vel.reshape(-1, 3)  # [batch_size * num_bodies, 3]
            partner_vel_local_flat = quat_rotate_inverse(current_root_rot_flat, partner_body_vel_flat)
            partner_vel_local = partner_vel_local_flat.reshape(batch_size, -1)  # [batch_size, num_bodies * 3]

            # Combine partner observations
            partner_obs = torch.cat([
                partner_root_pos_local,  # [batch_size, 3]
                partner_pos_local,       # [batch_size, num_bodies * 3]
                partner_vel_local        # [batch_size, num_bodies * 3]
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

    def _compute_partner_contact_obs(self, env_ids):
        """Compute partner contact observations for given environment IDs"""
        # Vectorized computation - handle all environments at once
        device = self.device

        # Convert env_ids to tensor if not already
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=device)

        # Find partner environment IDs (vectorized pair logic: even/odd pairing)
        partner_env_ids = torch.where(env_ids % 2 == 0, env_ids + 1, env_ids - 1)

        # Check if all partners exist
        valid_partners = partner_env_ids < self.num_envs
        if not torch.all(valid_partners):
            invalid_partners = partner_env_ids[~valid_partners]
            raise Exception(f"Partner env_ids out of bounds: {invalid_partners.tolist()}")

        # Get partner's hand contact observations (batch)
        partner_hand_contact = self._compute_hand_contact_obs(partner_env_ids)

        return partner_hand_contact

    def _compute_partner_task_obs(self, env_ids):
        """Compute task observations for partners - required for asymmetric critic"""
        partner_task_obs_list = []
        
        # Initialize cache if not exists
        if not hasattr(self, '_task_obs_cache'):
            self._task_obs_cache = {}
        
        for env_id in env_ids:
            env_id_val = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # Find partner environment (pair logic: even/odd pairing)
            if env_id_val % 2 == 0:
                partner_env_id = env_id_val + 1
            else:
                partner_env_id = env_id_val - 1
            
            # Try to reuse cached result first
            if partner_env_id in self._task_obs_cache:
                partner_task_obs_list.append(self._task_obs_cache[partner_env_id])
            elif partner_env_id < self.num_envs:
                # Fallback: compute if not in cache
                partner_env_tensor = torch.tensor([partner_env_id], device=self.device, dtype=torch.long)
                partner_task_obs = self._compute_task_obs_with_role_handling(partner_env_tensor, save_buffer=False)
                partner_task_obs_list.append(partner_task_obs.squeeze(0))
            else:
                msg = f"Partner env_id {partner_env_id} out of bounds"
                raise Exception(msg)
        
        return torch.stack(partner_task_obs_list)

    def _compute_partner_force_obs(self, env_ids):
        """Compute force observations for partners - required for asymmetric critic"""
        partner_force_obs_list = []
        
        # Initialize cache if not exists
        if not hasattr(self, '_force_obs_cache'):
            self._force_obs_cache = {}
        
        for env_id in env_ids:
            env_id_val = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # Find partner environment (pair logic: even/odd pairing)
            if env_id_val % 2 == 0:
                partner_env_id = env_id_val + 1
            else:
                partner_env_id = env_id_val - 1
            
            # Try to reuse cached result first
            if partner_env_id in self._force_obs_cache and self.cfg["env"].get("self_obs_v", 1) == 5:
                partner_force_obs_list.append(self._force_obs_cache[partner_env_id])
            elif partner_env_id < self.num_envs:
                # Fallback: extract from partner's self_obs buffer or compute
                if self.cfg["env"].get("self_obs_v", 1) == 5:
                    # Try to get from self_obs_buf first (faster)
                    if hasattr(self, 'self_obs_buf') and partner_env_id < self.self_obs_buf.shape[0]:
                        force_size = self._get_hand_force_obs_size()
                        partner_force_obs = self.self_obs_buf[partner_env_id, -force_size:]
                    else:
                        # Fallback: compute partner's self observation
                        partner_env_tensor = torch.tensor([partner_env_id], device=self.device, dtype=torch.long)
                        partner_self_obs = self._compute_humanoid_obs(partner_env_tensor)
                        force_size = self._get_hand_force_obs_size()
                        partner_force_obs = partner_self_obs[0, -force_size:]
                else:
                    partner_force_obs = torch.zeros(0, device=self.device)
                partner_force_obs_list.append(partner_force_obs)
            else:
                error_msg = f"Partner env_id {partner_env_id} out of bounds"
                raise Exception(error_msg)
        
        return torch.stack(partner_force_obs_list)

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
        partner_env_ids = torch.where(env_ids % 2 == 0, env_ids + 1, env_ids - 1)

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
        current_l_wrist_rot = self._normalize_quaternion_batch(current_l_wrist_rot)
        current_r_wrist_rot = self._normalize_quaternion_batch(current_r_wrist_rot)

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
    
    def _normalize_quaternion(self, quat):
        """Normalize quaternion and handle edge cases"""
        quat_norm = torch.norm(quat)
        if quat_norm < 1e-6:  # quaternion is too small/zero
            return torch.tensor([0.0, 0.0, 0.0, 1.0], device=quat.device)  # identity
        else:
            return quat / torch.clamp(quat_norm, min=1e-8)  # safe normalize

    def _normalize_quaternion_batch(self, quat_batch):
        """Normalize batch of quaternions and handle edge cases"""
        # quat_batch: [N, 4]
        quat_norms = torch.norm(quat_batch, dim=-1, keepdim=True)  # [N, 1]

        # Handle small/zero quaternions
        small_quat_mask = (quat_norms < 1e-6).squeeze(-1)  # [N]
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=quat_batch.device, dtype=quat_batch.dtype)

        # Safe normalize
        normalized_quats = quat_batch / torch.clamp(quat_norms, min=1e-8)

        # Replace small quaternions with identity
        normalized_quats[small_quat_mask] = identity_quat

        return normalized_quats
    
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
        # Based on current implementation
        if hasattr(self, '_body_names') and self._body_names is not None:
            num_bodies = len(self._body_names)
        else:
            num_bodies = 52  # Default SMPL-X body count (including fingers)
        return 3 + num_bodies * 3 + num_bodies * 3  # root_pos + joint_pos + joint_vel
    
    def _get_wrist_relative_partner_obs_size(self):
        """Calculate wrist-relative partner observation size"""
        # Based on current implementation: 2 wrists * num_bodies * 3D coordinates
        if hasattr(self, '_body_names') and self._body_names is not None:
            num_bodies = len(self._body_names)
        else:
            num_bodies = 52  # Default SMPL-X body count (including fingers)
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
                'L_Index1', 'L_Index2', 'L_Index3',
                'L_Middle1', 'L_Middle2', 'L_Middle3',
                'L_Ring1', 'L_Ring2', 'L_Ring3',
                'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
                'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
                'R_Index1', 'R_Index2', 'R_Index3',
                'R_Middle1', 'R_Middle2', 'R_Middle3',
                'R_Ring1', 'R_Ring2', 'R_Ring3',
                'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
                'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
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
        # ['L_Wrist', 'L_Index3', 'L_Middle3', 'L_Ring3', 'L_Pinky3', 'L_Thumb3',
        #  'R_Wrist', 'R_Index3', 'R_Middle3', 'R_Ring3', 'R_Pinky3', 'R_Thumb3',
        #  'L_Elbow', 'R_Elbow']
        # 14 bodies * 3D force vector = 42 values
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

    def _get_reference_contact_obs_size(self):
        """Calculate reference contact observation size"""
        # Handle case where cfg is not fully initialized
        if not hasattr(self, 'cfg') or self.cfg is None:
            return 0
        
        # Reference contact difference has same size as hand contact
        return self._get_hand_contact_obs_size()

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
        distance_threshold = 0.2  # 20cm distance threshold
        
        # Specific hand joints for distance check: Wrist, Index3, Middle3, Ring3, Pinky3, Thumb3
        specific_hand_joints = ['L_Wrist', 'L_Index3', 'L_Middle3', 'L_Ring3', 'L_Pinky3', 'L_Thumb3',
                               'R_Wrist', 'R_Index3', 'R_Middle3', 'R_Ring3', 'R_Pinky3', 'R_Thumb3']
        
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
            partner_env_ids = torch.where(env_ids_tensor % 2 == 0, env_ids_tensor + 1, env_ids_tensor - 1)

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
                wrist_body_names = ['L_Wrist', 'R_Wrist', 'L_Thorax', 'R_Thorax', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow']
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

    def _compute_reference_contact_obs(self, env_ids):
        """
        Compute reference contact observations from motion data
        Uses hand-level contact flags (left_hand_contact/right_hand_contact) 
        and applies to all joints of the corresponding hand
        """
        hand_contact_bodies = self.cfg["env"].get("hand_contact_bodies", [])
        
        if not hand_contact_bodies:
            msg = f"No hand contact bodies found in config"
            raise Exception(msg)
        
        # Classify hand bodies into left and right
        left_hand_bodies = [body for body in hand_contact_bodies if body.startswith('L_')]
        right_hand_bodies = [body for body in hand_contact_bodies if body.startswith('R_')]
        
        reference_contact_obs_list = []
        
        # Get current motion times (same logic as _compute_task_obs)
        motion_times = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
        
        for env_id in env_ids:
            env_id_val = env_id.item() if torch.is_tensor(env_id) else env_id
            
            # Get motion assignment for this environment
            motion_key = self.env_motion_assignments.get(env_id_val, "unknown")
            
            reference_contact = torch.zeros(len(hand_contact_bodies), device=self.device)
            
            if motion_key in self.interaction_data:
                motion_data = self.interaction_data[motion_key]
                
                # Get current frame contact flags
                current_time = motion_times[env_id].item()
                fps = motion_data.get('fps', 30)
                frame_idx = int(current_time * fps)
                
                # Get hand-level contact flags from motion data
                left_contact_flag = self._get_hand_contact_from_motion_data(motion_data, frame_idx, 'left')
                right_contact_flag = self._get_hand_contact_from_motion_data(motion_data, frame_idx, 'right')
                
                # Set contact flags for all joints of the contacting hand
                for i, body_name in enumerate(hand_contact_bodies):
                    if body_name in left_hand_bodies and left_contact_flag:
                        reference_contact[i] = 1.0
                    elif body_name in right_hand_bodies and right_contact_flag:
                        reference_contact[i] = 1.0
            
            reference_contact_obs_list.append(reference_contact)
        
        return torch.stack(reference_contact_obs_list)

    def _get_hand_contact_from_motion_data(self, motion_data, frame_idx, hand_side):
        """
        Get hand contact flag from motion data for specified frame and hand side
        
        Args:
            motion_data: Motion data dictionary
            frame_idx: Frame index to query
            hand_side: 'left' or 'right'
        
        Returns:
            bool: Contact flag for the specified hand
        """
        contact_key = f'{hand_side}_hand_contact'
        
        if contact_key not in motion_data:
            return False
        
        contact_sequence = motion_data[contact_key]
        
        # Clamp frame index to valid range
        if frame_idx >= len(contact_sequence):
            frame_idx = len(contact_sequence) - 1
        elif frame_idx < 0:
            frame_idx = 0
        
        return bool(contact_sequence[frame_idx])

    def _compute_reference_contact_difference(self, env_ids):
        """
        Compute reference contact - current contact difference
        
        Positive values: reference expects contact but currently not contacting
        Negative values: reference expects no contact but currently contacting
        
        Returns:
            torch.Tensor: Contact difference [batch, n_hand_bodies]
        """
        # Get current contact flags
        current_contact = self._compute_hand_contact_obs(env_ids)  # [batch, n_bodies]
        
        # Get reference contact flags  
        reference_contact = self._compute_reference_contact_obs(env_ids)  # [batch, n_bodies]
        
        # Compute difference: reference - current
        contact_difference = reference_contact - current_contact  # [batch, n_bodies]
        
        return contact_difference
    
    def _compute_humanoid_obs_v4(self, env_ids):
        """Compute humanoid observations v4 with hand contact information"""
        # Get base observations directly from parent class (bypassing our override)
        # Use the parent class method directly to avoid infinite recursion
        parent_class = super(HumanoidImInterx, self)
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
        parent_class = super(HumanoidImInterx, self)
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

    def _compute_task_obs_with_role_handling(self, env_ids=None, save_buffer=True):
        """Override to implement role-specific task observations like MultiPulse"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        
        # Initialize task_obs_cache if not exists
        if not hasattr(self, '_task_obs_cache'):
            self._task_obs_cache = {}
            self._force_obs_cache = {}
        
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
        
        # Cache results if save_buffer is True (normal actor computation)
        if save_buffer:
            for i, env_id in enumerate(env_ids):
                env_id_val = env_id.item() if torch.is_tensor(env_id) else env_id
                self._task_obs_cache[env_id_val] = obs[i].clone()
                
                # Also cache force observations if available (self_obs_v=5)
                if self.cfg["env"].get("self_obs_v", 1) == 5:
                    # Extract force obs from current self observation buffer
                    if hasattr(self, 'self_obs_buf') and env_id_val < self.self_obs_buf.shape[0]:
                        force_size = self._get_hand_force_obs_size()
                        force_obs = self.self_obs_buf[env_id_val, -force_size:].clone()
                        self._force_obs_cache[env_id_val] = force_obs
        
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
            total_size = self_obs_size + aux_features_size + self.task_obs_size + 1
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
        future_trajectory_size = 20 * non_hand_bodies * 13 * 2  # 2 for self+partner trajectories

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

    def _compute_hand_contact_reward(self):
        """Compute hand contact reward comparing simulator forces with ground truth flags"""
        # Hand contact reward parameters from config
        hand_contact_reward_weight = self.cfg["env"].get("hand_contact_reward_weight", 0.1)
        hand_contact_force_threshold = self.cfg["env"].get("hand_contact_force_threshold", 0.1)
        
        # Debug: Print parameters occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        # DISABLED: Remove debug prints to improve performance
        # if self._debug_counter % 1000 == 0:  # Print every 1000 steps
        #     print(f"\nDEBUG Hand Contact Reward: weight={hand_contact_reward_weight}, threshold={hand_contact_force_threshold}")
        #     print(f"  Has interaction_data: {hasattr(self, 'interaction_data')}")
        #     if hasattr(self, 'interaction_data'):
        #         print(f"  Interaction data keys: {len(self.interaction_data) if self.interaction_data else 0}")
        
        # Check if hand contact reward is enabled
        if hand_contact_reward_weight <= 0 or not hasattr(self, 'interaction_data'):
            msg = f"Hand contact reward weight is not positive or interaction data is not available"
            raise Exception(msg)
        
        # Get current motion times to fetch ground truth contact flags
        motion_times = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
        
        hand_contact_rewards = torch.zeros(self.num_envs, device=self.device)
        
        for env_id in range(self.num_envs):
            # Get motion assignment for this environment
            if env_id not in self.env_motion_assignments:
                continue
                
            motion_key = self.env_motion_assignments[env_id]
            role = self.env_role_assignments[env_id]
            
            # Get motion data for this environment
            if motion_key not in self.interaction_data:
                continue
                
            motion_data = self.interaction_data[motion_key]
            
            # Check if ground truth contact data exists
            if 'left_hand_contact' not in motion_data or 'right_hand_contact' not in motion_data:
                continue
            
            # Get current frame index based on motion time
            current_time = motion_times[env_id].item()
            fps = motion_data.get('fps', 30)
            frame_idx = int(current_time * fps)
            
            # Clamp frame index to valid range
            left_contact_gt = motion_data['left_hand_contact']
            right_contact_gt = motion_data['right_hand_contact']
            
            if frame_idx >= len(left_contact_gt):
                frame_idx = len(left_contact_gt) - 1
            elif frame_idx < 0:
                frame_idx = 0
            
            # Get ground truth contact flags for current frame
            left_contact_flag = left_contact_gt[frame_idx]
            right_contact_flag = right_contact_gt[frame_idx]
            
            # Compute simulator hand contact forces
            left_hand_force, right_hand_force = self._get_simulator_hand_forces(env_id)
            
            # Get hand heights to check if hands are close to ground
            left_hand_height, right_hand_height = self._get_hand_heights(env_id)
            ground_height_threshold = 0.2  # 20cm from ground

            # Apply same height gating to GT flags (AND with height condition)
            left_contact_flag = 1.0 if (left_contact_flag == 1 and 
                                     left_hand_height > ground_height_threshold) else 0.0
            right_contact_flag = 1.0 if (right_contact_flag == 1 and 
                                      right_hand_height > ground_height_threshold) else 0.0
            
            # Convert forces to binary contact flags using threshold
            # Only consider contact if hand is close to ground (within 20cm)
            left_contact_sim = 1.0 if (left_hand_force > hand_contact_force_threshold and 
                                     left_hand_height > ground_height_threshold) else 0.0
            right_contact_sim = 1.0 if (right_hand_force > hand_contact_force_threshold and 
                                      right_hand_height > ground_height_threshold) else 0.0
            
            # Compute reward: 1.0 if prediction matches ground truth, 0.0 otherwise
            # left_reward = 1.0 if (left_contact_sim == left_contact_flag) else 0.0
            # right_reward = 1.0 if (right_contact_sim == right_contact_flag) else 0.0
            
                        # Compute reward: 5.0 if GT contact is 1 and prediction matches, 1.0 for other correct predictions, 0.0 otherwise
            left_reward = 30.0 if (left_contact_sim == left_contact_flag and left_contact_flag == 1) else (1.0 if left_contact_sim == left_contact_flag else 0.0)
            right_reward = 30.0 if (right_contact_sim == right_contact_flag and right_contact_flag == 1) else (1.0 if right_contact_sim == right_contact_flag else 0.0)
            
            # Average left and right hand rewards
            hand_contact_rewards[env_id] = (left_reward + right_reward) * 0.5 * hand_contact_reward_weight
            
            # Track hand contact statistics if reward validation is enabled
            if self.zero_reward_on_poor_contact:
                # Count as good contact if either hand has proper contact (GT=1 and Sim matches)
                good_contact = ((left_contact_flag == 1 and left_contact_sim == 1) or 
                               (right_contact_flag == 1 and right_contact_sim == 1))
                if good_contact:
                    self.episode_hand_contact_count[env_id] += 1
            
            target_env_ids = torch.randint(0, self.num_envs, (5,))
            if self._debug_counter % 15 == 0 and env_id in target_env_ids:
                print(f"  Env {env_id}: GT(L:{left_contact_flag}, R:{right_contact_flag}), "
                      f"Sim(L:{left_hand_force:.3f}@{left_hand_height:.2f}m->{left_contact_sim}, "
                      f"R:{right_hand_force:.3f}@{right_hand_height:.2f}m->{right_contact_sim}), "
                      f"Reward: {hand_contact_rewards[env_id]:.4f}")
        
        if self._debug_counter % 1000 == 0:
            active_rewards = (hand_contact_rewards > 0).sum().item()
            mean_reward = hand_contact_rewards.mean().item()
            print(f"  Total: {active_rewards}/{self.num_envs} active, mean={mean_reward:.4f}")
        
        return hand_contact_rewards

    def _get_simulator_hand_forces(self, env_id):
        """Get maximum force magnitude for left and right hands from simulator"""
        # Hand body names for left and right hands (wrist + finger tips)
        left_hand_bodies = ['L_Wrist', 'L_Index3', 'L_Middle3', 'L_Ring3', 'L_Pinky3', 'L_Thumb3']
        right_hand_bodies = ['R_Wrist', 'R_Index3', 'R_Middle3', 'R_Ring3', 'R_Pinky3', 'R_Thumb3']
        
        # Get body name to ID mapping
        if not hasattr(self, '_body_name_to_id'):
            self._setup_body_name_to_id_mapping()
        
        # Get contact forces for left hand bodies
        left_max_force = 0.0
        for body_name in left_hand_bodies:
            if body_name in self._body_name_to_id:
                body_id = self._body_name_to_id[body_name]
                if body_id < self._contact_forces.shape[1]:
                    force_magnitude = torch.norm(self._contact_forces[env_id, body_id, :]).item()
                    left_max_force = max(left_max_force, force_magnitude)
        
        # Get contact forces for right hand bodies
        right_max_force = 0.0
        for body_name in right_hand_bodies:
            if body_name in self._body_name_to_id:
                body_id = self._body_name_to_id[body_name]
                if body_id < self._contact_forces.shape[1]:
                    force_magnitude = torch.norm(self._contact_forces[env_id, body_id, :]).item()
                    right_max_force = max(right_max_force, force_magnitude)
        
        return left_max_force, right_max_force

    def _collect_contact_statistics(self, hand_contact_reward):
        """Collect detailed contact statistics for wandb logging"""
        # Initialize contact stats dictionary if not exists
        if not hasattr(self, '_contact_stats'):
            self._contact_stats = {}
        
        # Basic statistics
        active_rewards = (hand_contact_reward > 0).sum().item()
        mean_reward = hand_contact_reward.mean().item()
        max_reward = hand_contact_reward.max().item()
        
        # Store statistics
        self._contact_stats.update({
            'contact/hand_reward_mean': mean_reward,
            'contact/hand_reward_max': max_reward,
            'contact/active_contact_envs': active_rewards,
            'contact/active_contact_ratio': active_rewards / self.num_envs,
        })
        
        # Collect detailed force and height statistics for a subset of environments
        if hasattr(self, '_debug_counter') and self._debug_counter % 100 == 0:
            sample_envs = min(10, self.num_envs)  # Sample up to 10 environments
            sample_forces_left = []
            sample_forces_right = []
            sample_heights_left = []
            sample_heights_right = []
            
            for env_id in range(sample_envs):
                try:
                    left_force, right_force = self._get_simulator_hand_forces(env_id)
                    left_height, right_height = self._get_hand_heights(env_id)
                    
                    sample_forces_left.append(left_force)
                    sample_forces_right.append(right_force)
                    sample_heights_left.append(left_height)
                    sample_heights_right.append(right_height)
                except:
                    pass  # Skip if any error occurs
            
            if sample_forces_left:  # Only update if we have samples
                self._contact_stats.update({
                    'contact/sample_force_left_mean': sum(sample_forces_left) / len(sample_forces_left),
                    'contact/sample_force_right_mean': sum(sample_forces_right) / len(sample_forces_right),
                    'contact/sample_height_left_mean': sum(sample_heights_left) / len(sample_heights_left),
                    'contact/sample_height_right_mean': sum(sample_heights_right) / len(sample_heights_right),
                    'contact/force_threshold': self.cfg["env"].get("hand_contact_force_threshold", 0.1),
                    'contact/reward_weight': self.cfg["env"].get("hand_contact_reward_weight", 1.0),
                })

    def _get_hand_heights(self, env_id):
        """Get heights of left and right hands (wrist positions) from ground"""
        # Get body name to ID mapping
        if not hasattr(self, '_body_name_to_id'):
            self._setup_body_name_to_id_mapping()
        
        # Get wrist heights
        left_height = 0.0
        right_height = 0.0
        
        if 'L_Wrist' in self._body_name_to_id:
            body_id = self._body_name_to_id['L_Wrist']
            if body_id < self._rigid_body_pos.shape[1]:
                left_height = self._rigid_body_pos[env_id, body_id, 2].item()  # z-coordinate
        
        if 'R_Wrist' in self._body_name_to_id:
            body_id = self._body_name_to_id['R_Wrist']
            if body_id < self._rigid_body_pos.shape[1]:
                right_height = self._rigid_body_pos[env_id, body_id, 2].item()  # z-coordinate
        
        return left_height, right_height

    def _get_head_height(self, env_id):
        """Get head height for given environment ID"""
        # Setup body name to ID mapping if not already done
        if not hasattr(self, '_body_name_to_id'):
            self._setup_body_name_to_id_mapping()
        
        head_height = 0.0
        
        if 'Head' in self._body_name_to_id:
            body_id = self._body_name_to_id['Head']
            if body_id < self._rigid_body_pos.shape[1]:
                head_height = self._rigid_body_pos[env_id, body_id, 2].item()  # z-coordinate
        
        return head_height

    def _check_caregiver_recipient_hand_contact(self, caregiver_env_id, recipient_env_id):
        """Check if at least one caregiver hand is in contact with the recipient.
        Uses simulator hand contact forces and proximity check to approximate inter-agent contact.
        """
        # Thresholds
        hand_force_threshold = self.cfg["env"].get("hand_contact_threshold", 0.1)
        touch_distance_threshold = self.cfg["env"].get("hand_touch_distance_threshold", 0.15)

        # Ensure body mapping exists
        if not hasattr(self, '_body_name_to_id'):
            self._setup_body_name_to_id_mapping()

        # Determine hand bodies to check
        hand_bodies = self.cfg["env"].get("hand_contact_bodies", [])
        if not hand_bodies:
            hand_bodies = [
                'L_Wrist', 'R_Wrist',
                'L_Index3', 'L_Middle3', 'L_Ring3', 'L_Pinky3', 'L_Thumb3',
                'R_Index3', 'R_Middle3', 'R_Ring3', 'R_Pinky3', 'R_Thumb3',
            ]

        # Pre-fetch positions for recipient (all bodies)
        if recipient_env_id >= self._rigid_body_pos.shape[0]:
            return False
        recipient_body_pos = self._rigid_body_pos[recipient_env_id, :, :3].clone()  # Need clone since we modify with +=
        recipient_body_pos[:, 0] += self.cfg["env"].get('env_spacing', 5.0) * 2
        

        # Iterate caregiver hands
        for body_name in hand_bodies:
            body_id = self._body_name_to_id.get(body_name)
            if body_id is None:
                continue
            if body_id >= self._contact_forces.shape[1] or body_id >= self._rigid_body_pos.shape[1]:
                continue

            # Caregiver hand contact force
            force = torch.norm(self._contact_forces[caregiver_env_id, body_id, :]).item()
            if force <= hand_force_threshold:
                continue

            # Proximity to recipient: any recipient body within threshold
            caregiver_hand_pos = self._rigid_body_pos[caregiver_env_id, body_id, :3]
            # Compute distances to all recipient bodies (vectorized)
            diffs = recipient_body_pos - caregiver_hand_pos.unsqueeze(0)
            dists = torch.norm(diffs, dim=-1)
            min_dist = torch.min(dists).item()
            if min_dist <= touch_distance_threshold:
                return True

        return False

    def _collect_simpleliftup_statistics(self):
        """Collect SimpleLiftUp specific statistics for wandb logging"""
        # Initialize contact stats dictionary if not exists
        if not hasattr(self, '_contact_stats'):
            self._contact_stats = {}
        
        # Collect episode max recipient heights statistics if available
        if hasattr(self, 'episode_max_recipient_heights') and self.simple_lift_up_mode:
            # Get recipient environment IDs (odd env_ids)
            recipient_env_ids = [env_id for env_id in range(self.num_envs) if env_id % 2 == 1]
            
            if recipient_env_ids:
                recipient_heights = self.episode_max_recipient_heights[recipient_env_ids]
                # Only consider non-zero heights (episodes that had contact)
                non_zero_heights = recipient_heights[recipient_heights > 0]
                
                if len(non_zero_heights) > 0:
                    episode_max_height_mean = non_zero_heights.mean().item()
                    self._contact_stats.update({
                        'simpleliftup/episode_max_height_mean': episode_max_height_mean,
                        'simpleliftup/episode_max_height_max': non_zero_heights.max().item(),
                        'simpleliftup/episode_max_height_min': non_zero_heights.min().item(),
                        'simpleliftup/episodes_with_contact': len(non_zero_heights),
                        'simpleliftup/episodes_with_contact_ratio': len(non_zero_heights) / len(recipient_env_ids),
                    })
                    # Store for main logging (similar to episode_lengths)
                    self._episode_max_height_mean = episode_max_height_mean
                else:
                    self._contact_stats.update({
                        'simpleliftup/episode_max_height_mean': 0.0,
                        'simpleliftup/episode_max_height_max': 0.0,
                        'simpleliftup/episode_max_height_min': 0.0,
                        'simpleliftup/episodes_with_contact': 0,
                        'simpleliftup/episodes_with_contact_ratio': 0.0,
                    })
                    # Store for main logging
                    self._episode_max_height_mean = 0.0
        
        # Also collect current recipient head heights statistics
        if self.simple_lift_up_mode:
            current_recipient_heights = []
            for env_id in range(self.num_envs):
                if env_id % 2 == 1:  # Recipient environments
                    head_height = self._get_head_height(env_id)  # Current head height
                    current_recipient_heights.append(head_height)
            
            if current_recipient_heights:
                current_heights_tensor = torch.tensor(current_recipient_heights, device=self.device)
                self._contact_stats.update({
                    'simpleliftup/current_height_mean': current_heights_tensor.mean().item(),
                    'simpleliftup/current_height_max': current_heights_tensor.max().item(),
                    'simpleliftup/current_height_min': current_heights_tensor.min().item(),
                })

    def _compute_min_hand_distance(self, caregiver_env_id, recipient_env_id):
        """Compute minimum distance between caregiver hands and recipient upper body joints"""
        # Setup body name to ID mapping if not already done
        if not hasattr(self, '_body_name_to_id'):
            self._setup_body_name_to_id_mapping()
        
        # Caregiver hand body names
        caregiver_hand_bodies = ['L_Wrist', 'R_Wrist']
        
        # Recipient upper body joint names
        recipient_upper_body_joints = [
            'Torso', 'Spine', 'Chest', 'Neck', 'Head',
            'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
            'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist'
        ]
        
        min_distance = float('inf')
        
        # Get caregiver hand positions
        caregiver_hand_positions = {}
        for hand_name in caregiver_hand_bodies:
            if hand_name in self._body_name_to_id:
                body_id = self._body_name_to_id[hand_name]
                if body_id < self._rigid_body_pos.shape[1]:
                    caregiver_hand_positions[hand_name] = self._rigid_body_pos[caregiver_env_id, body_id, :3]  # Read-only, no clone needed
        
        # Get recipient upper body joint positions
        recipient_upper_body_positions = {}
        for joint_name in recipient_upper_body_joints:
            if joint_name in self._body_name_to_id:
                body_id = self._body_name_to_id[joint_name]
                if body_id < self._rigid_body_pos.shape[1]:
                    recipient_upper_body_positions[joint_name] = self._rigid_body_pos[recipient_env_id, body_id, :3].clone()  # Need clone since we modify with +=
                    # recipientに対するoffsetを考慮にいれる
                    recipient_upper_body_positions[joint_name] += torch.tensor([self.cfg["env"].get('env_spacing', 5.0) * 2, 0.0, 0.0], device=self.device)

        # Compute distances between all caregiver hands and all recipient upper body joints
        for caregiver_hand in caregiver_hand_positions:
            for recipient_joint in recipient_upper_body_positions:
                distance = torch.norm(caregiver_hand_positions[caregiver_hand] - 
                                    recipient_upper_body_positions[recipient_joint]).item()
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 10.0  # Fallback large distance

    def _compute_root_distance(self, caregiver_env_id, recipient_env_id):
        """Compute distance between caregiver and recipient root positions"""
        # Get root positions
        caregiver_root_pos = self._rigid_body_pos[caregiver_env_id, 0, :3]  # Read-only, no clone needed
        recipient_root_pos = self._rigid_body_pos[recipient_env_id, 0, :3]  # Read-only, no clone needed
        
        # Apply motion cache offset compensation (same as in hand distance)
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        if recipient_env_id % 2 == 1:  # Recipient offset compensation
            motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], 
                                             device=self.device, dtype=torch.float32)
            recipient_root_pos = recipient_root_pos + motion_cache_offset
        
        if caregiver_env_id % 2 == 1:  # Caregiver offset compensation (shouldn't happen but safety)
            motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], 
                                             device=self.device, dtype=torch.float32)
            caregiver_root_pos = caregiver_root_pos + motion_cache_offset
        
        # Compute root-to-root distance
        root_distance = torch.norm(caregiver_root_pos - recipient_root_pos).item()
        
        return root_distance

    def _compute_reward(self, actions):
        """Override to implement SimpleLiftUp reward or normal interaction reward"""
        if self.simple_lift_up_mode:
            # SimpleLiftUp mode: Check hand distances first
            hand_distance_masks, hand_rewards = self._compute_hand_distance_rewards()
            
            # If any hands are close (within 40cm), use masking reward (higher reward)
            if hand_distance_masks.any():
                self._compute_reward_with_hand_masking(actions, hand_distance_masks, hand_rewards)
                reward_type = "hand_masking"
            else:
                # No close hands, use traditional parent reward computation (lower reward)
                super()._compute_reward(actions)
                reward_type = "traditional"

            # Add recipient head height reward for lift-up motion
            recipient_head_height_reward = self._compute_recipient_head_height_reward()
            self.rew_buf[:] += recipient_head_height_reward

        else:
            # Normal interaction mode: imitation + hand contact + role-aware
            # Call parent reward computation first (normal imitation + power reward)
            super()._compute_reward(actions)
            hand_contact_reward = torch.zeros(self.num_envs, device=self.device)
            
            # Store hand contact reward for logging
            self.hand_contact_reward = hand_contact_reward
            
        # Zero out recipient rewards to exclude them from learning (Option 1 implementation)  
        # for env_id in range(self.num_envs):
        #     if env_id % 2 == 1:  # Recipient environments (odd env_ids)
        #         self.rew_buf[env_id] = 0.0
        
        # Add hand contact reward to reward_raw for tracking (only in normal mode)
        if not (self.simple_lift_up_mode and self.task_reward_only):
            self.reward_raw = torch.cat([self.reward_raw, hand_contact_reward[:, None]], dim=-1)
        
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

        # Create weight tensors
        self_weights = torch.where(is_recipient, 0.8, 0.2)  # recipient=0.8, caregiver=0.2
        partner_weights = torch.where(is_recipient, 0.2, 0.8)  # recipient=0.2, caregiver=0.8

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
        partner_env_ids = torch.where(env_ids % 2 == 0, env_ids + 1, env_ids - 1)

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
            transition_distance = 0.25
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

            self.rew_buf[~zeros_subset] = self.rew_buf[~zeros_subset] + im_reward * 0.5
            self.reward_raw[~zeros_subset, :5] = self.reward_raw[~zeros_subset, :5] + im_reward_raw * 0.5

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

        # Distance threshold (40cm = 0.4m)
        distance_threshold = 0.4

        # Exponential reward parameters (higher rewards for closer hands)
        exp_decay_rate = 2.5  # Controls how fast the reward decays with distance
        reward_coefficient = 0.05  # Weight for the exponential reward
        close_hands_bonus = 0.05  # Additional bonus when hands are within threshold

        # Get upper body joint IDs
        upper_body_joints = [
            'Torso', 'Spine', 'Chest', 'Neck', 'Head',
            'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
            'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist'
        ]

        upper_body_ids = []
        for joint_name in upper_body_joints:
            body_id = self._body_name_to_id.get(joint_name)
            if body_id is not None:
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
        recipient_l_wrists = self._rigid_body_pos[recipient_envs, l_wrist_id, :3].clone()  # [num_pairs, 3]
        recipient_r_wrists = self._rigid_body_pos[recipient_envs, r_wrist_id, :3].clone()  # [num_pairs, 3]

        # Apply offset compensation for recipients
        pair_offset_val = self.cfg["env"].get('env_spacing', 5.0) * 2
        motion_cache_offset = torch.tensor([pair_offset_val, 0.0, 0.0], device=self.device, dtype=torch.float32)
        recipient_l_wrists = recipient_l_wrists + motion_cache_offset
        recipient_r_wrists = recipient_r_wrists + motion_cache_offset

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
        recip_l_to_care_distances = torch.norm(
            caregiver_joints - recipient_l_wrists.unsqueeze(1), dim=2
        )  # [num_pairs, num_joints]
        recip_r_to_care_distances = torch.norm(
            caregiver_joints - recipient_r_wrists.unsqueeze(1), dim=2
        )  # [num_pairs, num_joints]

        # Find minimum distances for each hand
        min_care_l_to_recip, _ = torch.min(care_l_to_recip_distances, dim=1)  # [num_pairs]
        min_care_r_to_recip, _ = torch.min(care_r_to_recip_distances, dim=1)  # [num_pairs]
        min_recip_l_to_care, _ = torch.min(recip_l_to_care_distances, dim=1)  # [num_pairs]
        min_recip_r_to_care, _ = torch.min(recip_r_to_care_distances, dim=1)  # [num_pairs]

        # Check if any hand is close to any upper body joint
        care_l_close = min_care_l_to_recip <= distance_threshold  # [num_pairs]
        care_r_close = min_care_r_to_recip <= distance_threshold  # [num_pairs]
        recip_l_close = min_recip_l_to_care <= distance_threshold  # [num_pairs]
        recip_r_close = min_recip_r_to_care <= distance_threshold  # [num_pairs]

        # Apply masking where any hand is close
        caregiver_mask = care_l_close | care_r_close
        recipient_mask = recip_l_close | recip_r_close

        hand_distance_masks[caregiver_envs] = caregiver_mask
        hand_distance_masks[recipient_envs] = recipient_mask

        # Calculate exponential rewards for each close hand
        care_l_rewards = torch.where(
            care_l_close,
            reward_coefficient * torch.exp(-exp_decay_rate * min_care_l_to_recip) + close_hands_bonus,
            torch.zeros_like(min_care_l_to_recip)
        )
        care_r_rewards = torch.where(
            care_r_close,
            reward_coefficient * torch.exp(-exp_decay_rate * min_care_r_to_recip) + close_hands_bonus,
            torch.zeros_like(min_care_r_to_recip)
        )
        recip_l_rewards = torch.where(
            recip_l_close,
            reward_coefficient * torch.exp(-exp_decay_rate * min_recip_l_to_care) + close_hands_bonus,
            torch.zeros_like(min_recip_l_to_care)
        )
        recip_r_rewards = torch.where(
            recip_r_close,
            reward_coefficient * torch.exp(-exp_decay_rate * min_recip_r_to_care) + close_hands_bonus,
            torch.zeros_like(min_recip_r_to_care)
        )

        # Sum all rewards for each pair
        total_pair_rewards = care_l_rewards + care_r_rewards + recip_l_rewards + recip_r_rewards

        # Apply rewards to both caregiver and recipient in each pair
        hand_rewards[caregiver_envs] += total_pair_rewards
        hand_rewards[recipient_envs] += total_pair_rewards

        return hand_distance_masks, hand_rewards

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
        height_rewards = reward_scale * torch.clamp(head_heights, min=0.0, max=2.0)

        return height_rewards

    def _apply_recipient_weakness(self):
        """Apply weakness to recipients by modifying their DOF properties"""
        if self.control_mode != "isaac_pd":
            return  # Only works with Isaac PD control
        
        # Get recipient weakness scale from config
        recipient_weakness_scale = self.cfg["env"].get("recipient_weakness_scale", 1.0)
            
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
                    dof_prop['effort'][dof_idx] = 80.0
                
                # Apply modified properties
                self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
                modified_count += 1
                
        print(f"Applied recipient weakness (scale={recipient_weakness_scale}) to {modified_count} recipients, affecting {len(lower_body_dof_indices)} lower body DOFs")
    
    def _quat_rotate_bn(self, q_B4: torch.Tensor, v_BN3: torch.Tensor) -> torch.Tensor:
        """
        Isaacのquat_rotateは (B,4)×(B,3) 前提なので、(B,N,3) を (B*N,3) に潰して回す。
        q_B4: (B,4)
        v_BN3: (B,N,3)
        return: (B,N,3)
        """
        B, N, _ = v_BN3.shape
        v_flat = v_BN3.reshape(B * N, 3)
        q_flat = q_B4.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4)
        out_flat = torch_utils.quat_rotate(q_flat, v_flat)  # (B*N,3)
        return out_flat.view(B, N, 3)

    def _quat_mul_bn(self, q_B4: torch.Tensor, r_BN4: torch.Tensor) -> torch.Tensor:
        """
        (B,4) と (B,N,4) を掛ける（右に r、左に q）。
        return: (B,N,4)
        """
        B, N, _ = r_BN4.shape
        q_flat = q_B4.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4)
        r_flat = r_BN4.reshape(B * N, 4)
        out_flat = torch_utils.quat_mul(q_flat, r_flat)  # (B*N,4)
        return out_flat.view(B, N, 4)

    def _normalize_quat(self, q_BN4: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        return q_BN4 / (q_BN4.norm(dim=-1, keepdim=True).clamp_min(eps))

    def _extract_yaw_from_quaternion(self, quat):
        """Extract yaw angle from quaternion (x, y, z, w)"""
        x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw
    
    def _quat_from_yaw(self, yaw):
        """Create quaternion from yaw angle (rotation around Z-axis)"""
        half_yaw = yaw / 2.0
        zeros = torch.zeros_like(half_yaw)
        quat = torch.stack([zeros, zeros, torch.sin(half_yaw), torch.cos(half_yaw)], dim=-1)
        return quat
    
    def _rebase_reference_motion(self, root_pos, root_rot, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel):
        """
        Rebase reference motion to current root with yaw-only alignment.
        Uses root positions/rotations as reference anchors.
        """
        # Get reference root data from first body (assume index 0 is root/pelvis)
        ref_root_pos = ref_body_pos[:, 0, :]  # (B, 3)
        ref_root_rot = ref_body_rot[:, 0, :]  # (B, 4)
        ref_root_vel = ref_body_vel[:, 0, :]  # (B, 3)
        ref_root_ang_vel = ref_body_ang_vel[:, 0, :]  # (B, 3)
        
        # Convert reference to anchor-relative coordinates
        ref_root_rot_conj = torch_utils.quat_conjugate(ref_root_rot)
        ref_pos_diff = ref_body_pos - ref_root_pos.unsqueeze(1)
        p_rel = self._quat_rotate_bn(ref_root_rot_conj, ref_pos_diff)
        
        q_rel = torch_utils.quat_mul(
            ref_root_rot_conj.unsqueeze(1).expand(-1, ref_body_rot.size(1), -1).reshape(-1,4),
            ref_body_rot.reshape(-1,4)
        ).view(ref_body_rot.size(0), ref_body_rot.size(1), 4)
        
        ref_vel_diff = ref_body_vel - ref_root_vel.unsqueeze(1)
        v_rel = self._quat_rotate_bn(ref_root_rot_conj, ref_vel_diff)
        
        ref_ang_vel_diff = ref_body_ang_vel - ref_root_ang_vel.unsqueeze(1)
        w_rel = self._quat_rotate_bn(ref_root_rot_conj, ref_ang_vel_diff)
        
        # Compute yaw-only transformation
        yaw_current = self._extract_yaw_from_quaternion(root_rot)
        yaw_ref = self._extract_yaw_from_quaternion(ref_root_rot)
        yaw_delta = yaw_current - yaw_ref
        
        q_delta = self._quat_from_yaw(yaw_delta)
        
        # Translation delta: [root_pos.x, root_pos.y, ref_root_pos.z]
        p_delta = torch.stack([
            root_pos[:, 0],  # current X
            root_pos[:, 1],  # current Y
            # ref_root_pos[:, 2]  # reference Z
            root_pos[:, 2]  # current Z
        ], dim=-1)
        
        # Transform relative data back to world coordinates with yaw-only alignment
        ref_pos_rebased = self._quat_rotate_bn(q_delta, p_rel) + p_delta.unsqueeze(1)
        ref_rot_rebased = self._quat_mul_bn(q_delta, q_rel)
        # ref_vel_rebased = self._quat_rotate_bn(q_delta, v_rel)
        # ref_ang_rebased = self._quat_rotate_bn(q_delta, w_rel)
        
        ref_pos_rebased[:, 0, :] = ref_body_pos[:, 0, :]
        ref_rot_rebased[:, 0, :] = ref_body_rot[:, 0, :]
        # ref_vel_rebased[:, 0, :] = ref_body_vel[:, 0, :]
        # ref_ang_rebased[:, 0, :] = ref_body_ang_vel[:, 0, :]
        ref_rot_rebased = self._normalize_quat(ref_rot_rebased)
        
        return ref_pos_rebased, ref_body_rot, ref_body_vel, ref_body_ang_vel

    def _compute_masked_imitation_reward(self, root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, 
                                       ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, 
                                       hand_masks, rwd_specs, rebase_flg=False, env_ids=None):
        """Compute imitation reward with optional masking of finger/wrist tracking"""
        k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
        w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]
        
        # Partner relative position reward parameters
        k_partner_rel_pos = rwd_specs.get("k_partner_rel_pos", self.cfg["env"].get("partner_rel_pos_k", 4.0))
        w_partner_rel_pos = rwd_specs.get("w_partner_rel_pos", self.cfg["env"].get("partner_rel_pos_weight", 0.0))
        
        # Apply rebasing if requested
        if rebase_flg:
            ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel = self._rebase_reference_motion(
                root_pos, root_rot, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel
            )

        # Get finger/wrist body indices for masking
        finger_wrist_bodies = self._get_finger_wrist_body_indices()
        
        # body position reward
        diff_global_body_pos = ref_body_pos - body_pos
        
        # Apply masking to finger/wrist bodies when hands are close
        if len(finger_wrist_bodies) > 0:
            # for env_idx in range(diff_global_body_pos.shape[0]):
            #     if hand_masks[env_idx]:
            #         # Zero out finger/wrist position differences for this environment
            #         diff_global_body_pos[env_idx, finger_wrist_bodies, :] = 0.0
            # batch mask - use broadcasting-compatible indexing
            if torch.any(hand_masks):
                diff_global_body_pos[hand_masks, :, :][:, finger_wrist_bodies, :] = 0.0
        
        diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
        r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

        # body rotation reward
        diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
        
        # Apply masking to finger/wrist bodies when hands are close (before angle conversion)
        if len(finger_wrist_bodies) > 0:
            # for env_idx in range(diff_global_body_rot.shape[0]):
            #     if hand_masks[env_idx]:
            #         # Set finger/wrist quaternions to identity (no rotation difference)
            #         diff_global_body_rot[env_idx, finger_wrist_bodies, :] = torch.tensor([0, 0, 0, 1], device=diff_global_body_rot.device, dtype=diff_global_body_rot.dtype)
            if torch.any(hand_masks):
                diff_global_body_rot[hand_masks, :, :][:, finger_wrist_bodies, :] = torch.tensor([0, 0, 0, 1], device=diff_global_body_rot.device, dtype=diff_global_body_rot.dtype)
        
        diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
        diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
        r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

        # body linear velocity reward
        diff_global_vel = ref_body_vel - body_vel
        
        # Apply masking to finger/wrist bodies when hands are close
        if len(finger_wrist_bodies) > 0:
            # for env_idx in range(diff_global_vel.shape[0]):
            #     if hand_masks[env_idx]:
            #         # Zero out finger/wrist velocity differences for this environment
            #         diff_global_vel[env_idx, finger_wrist_bodies, :] = 0.0
            if torch.any(hand_masks):
                diff_global_vel[hand_masks, :, :][:, finger_wrist_bodies, :] = 0.0
        
        diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
        r_vel = torch.exp(-k_vel * diff_global_vel_dist)

        # body angular velocity reward
        diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
        
        # Apply masking to finger/wrist bodies when hands are close
        if len(finger_wrist_bodies) > 0:
            # for env_idx in range(diff_global_ang_vel.shape[0]):
            #     if hand_masks[env_idx]:
            #         # Zero out finger/wrist angular velocity differences for this environment
            #         diff_global_ang_vel[env_idx, finger_wrist_bodies, :] = 0.0
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
            'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3',
            'L_Middle1', 'L_Middle2', 'L_Middle3',
            'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
            'L_Ring1', 'L_Ring2', 'L_Ring3',
            'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
            'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3',
            'R_Middle1', 'R_Middle2', 'R_Middle3',
            'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
            'R_Ring1', 'R_Ring2', 'R_Ring3',
            'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
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
        partner_env_ids = torch.where(env_ids % 2 == 0, env_ids + 1, env_ids - 1)
        
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
        partner_env_ids = torch.where(env_ids % 2 == 0, env_ids + 1, env_ids - 1)
        
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


    def _get_role_tracking_weights(self):
        """Get role-specific tracking weights based on current epoch and role"""
        # Get caregiver tracking weight based on epoch
        caregiver_weight = self._get_caregiver_tracking_weight()
        recipient_weight = 1.0  # Recipients always use full tracking
        
        # Create per-environment weights
        role_weights = torch.ones(self.num_envs, device=self.device)
        
        for env_id in range(self.num_envs):
            if env_id in self.env_role_assignments:
                role = self.env_role_assignments[env_id]
                if role == 'caregiver':
                    role_weights[env_id] = caregiver_weight
                else:  # recipient
                    role_weights[env_id] = recipient_weight
            else:
                # Fallback: even env_id = caregiver, odd env_id = recipient
                if env_id % 2 == 0:  # caregiver
                    role_weights[env_id] = caregiver_weight
                else:  # recipient
                    role_weights[env_id] = recipient_weight
        
        return role_weights
    
    def _get_caregiver_tracking_weight(self):
        """Get caregiver tracking weight based on curriculum learning"""
        if not hasattr(self, '_caregiver_initial_weight'):
            self._caregiver_initial_weight = getattr(self, 'caregiver_initial_weight', 0.1)
        if not hasattr(self, '_caregiver_final_weight'):
            self._caregiver_final_weight = getattr(self, 'caregiver_final_weight', 0.033)  # 1/3 of initial
        if not hasattr(self, '_caregiver_curriculum_epoch'):
            self._caregiver_curriculum_epoch = getattr(self, 'caregiver_curriculum_epoch', 500)
        
        # Get current epoch from training (fallback to 0 if not available)
        current_epoch = getattr(self, '_current_epoch', 0)
        
        if current_epoch < self._caregiver_curriculum_epoch:
            return self._caregiver_initial_weight
        else:
            return self._caregiver_final_weight

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
        non_hand_bodies = 52 - 30  # Total SMPL-X bodies minus 30 hand joints = 22 bodies
        future_trajectory_size = 20 * non_hand_bodies * 13 * 2  # 2 for self+partner trajectories

        # aux_features = [self_contact, self_force, base_partner_obs, partner_contact, self_future_traj_flat, partner_future_traj_flat, role_label]
        aux_features_size = self_contact_size + self_force_size + base_partner_size + partner_contact_size + future_trajectory_size   # +1 for role label

        return aux_features_size

    def _sample_time(self, motion_ids):
        # 常にシーケンスのt=0から開始（訓練時もテスト時も同じ動作）
        return torch.zeros(motion_ids.shape, device=self.device)
    
    def post_physics_step(self):
        """Override to add trajectory tracking"""
        # Call parent implementation
        super().post_physics_step()
        
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._capture_trajectory_step(all_env_ids)
    
    def _reset_envs(self, env_ids):
        """Override to start trajectory tracking for reset environments"""
        # Call parent implementation first
        super()._reset_envs(env_ids)
        
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
            hand_joints = ["L_Index1", "L_Index2", "L_Index3",
                          "L_Middle1", "L_Middle2", "L_Middle3", "L_Pinky1", "L_Pinky2", "L_Pinky3",
                          "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3",
                          "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3",
                          "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3",
                          "R_Thumb1", "R_Thumb2", "R_Thumb3"]

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
            assert len(self._non_hand_track_bodies_id) == 22
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

        return normalized_trajectories
