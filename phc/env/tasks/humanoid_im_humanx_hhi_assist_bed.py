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

from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict


class HumanoidImHumanxHelpUpBed(HumanoidIm, RSIMixin):
    """
    Humanoid Interaction class for humanx per environment interaction tasks with bed.
    Following the instruction: 1 environment = 1 humanoid, with collision groups for pair interaction.
    Includes bed asset in recipient environments.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Store interaction-specific config
        self.hhi_assist_bed_data_path = cfg["env"].get("hhi_assist_bed_data_path",
                                               "../PHC/sample_data/hhi-assist_processed_v6.pkl")
        print(f"hhi_assist_bed_data_path: {self.hhi_assist_bed_data_path}")
        self.humanoid_number = 2  # As per instruction: pair them with collision groups

        # Asset name configuration
        self.asset_name = cfg["env"].get("asset_name", "bed")
        print(f"Asset name: {self.asset_name}")

        self.task_reward_only = cfg["env"].get("task_reward_only", False)
        self.recipient_mass_scale = cfg["env"].get("recipient_mass_scale", 0.3)


        # Power coefficient for caregiver and recipient
        self.caregiver_power_coefficient = cfg["env"].get("caregiver_power_coefficient", 0.001)
        self.recipient_power_coefficient = cfg["env"].get("recipient_power_coefficient", 0.05)

        # Ankle height termination threshold (meters)
        self.ankle_height_termination_threshold = cfg["env"].get("ankle_height_termination_threshold", 1000)

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

        # Initialize bed handles list
        self._bed_handles = []

        # Do not force numObservations here; it will be derived dynamically by parent init via get_obs_size()

        # Initialize parent class
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine,
                        device_type=device_type, device_id=device_id, headless=headless)

        # Setup body name to ID mapping for contact processing
        self._setup_body_name_to_id_mapping()

        # Update the observation buffer size after initialization
        self._update_obs_buf_size()

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

        # Static termination tracking (30 frames with joint movement < 1cm)
        self.static_frames_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.prev_joint_positions = None  # Will be initialized in first step
        self.STATIC_FRAME_THRESHOLD = 30  # 30 frames
        self.JOINT_MOVEMENT_THRESHOLD = 0.01  # 1cm

        # Initialize torque tracking for recipient joints
        self.recipient_torque_buffer = []  # List to store torque data for each step
        self.current_motion_filename = None  # Current motion filename for saving

        # Store environment IDs as a tensor for network access
        self.env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Build bed-related tensors after all initialization is complete
        self._build_bed_tensors()
        
        # Caregiver recipient torque reduction reward settings
        self.caregiver_torque_reduction_reward = cfg["env"].get("caregiver_torque_reduction_reward", True)
        self.caregiver_torque_reduction_coefficient = cfg["env"].get("caregiver_torque_reduction_coefficient", 0.5)
        self.caregiver_torque_reduction_scale = cfg["env"].get("caregiver_torque_reduction_scale", 150.0) 

        return

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



    def _load_motion(self, motion_train_file, motion_test_file=[]):
        """Override to setup interaction motion data after parent loads motion"""
        # Use the interaction data path if no specific motion file is provided
        if not motion_train_file or motion_train_file == "":
            motion_train_file = self.hhi_assist_bed_data_path
        
        # Call parent method first
        super()._load_motion(motion_train_file, motion_test_file)
        
        # Setup interaction motion data after parent has loaded motion (with safety checks)
        self._setup_interaction_motion_data()
        
        return

    def _setup_interaction_motion_data(self):
        """Setup motion data for interaction tasks"""
        print(f"Loading interaction data from: {self.hhi_assist_bed_data_path}")

        # Safety check for motion data file
        try:
            # Load interaction motion data
            self.interaction_data = joblib.load(self.hhi_assist_bed_data_path)
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

    def _load_bed_asset(self):
        """Load furniture asset (bed or chair) for spawning in environments

        TEST: Using create_box instead of URDF to test if URDF is causing memory leak
        Bed dimensions from bed.urdf: 190.5cm L x 99.06cm W x 45.72cm H
        In meters: 1.905 L x 0.9906 W x 0.4572 H
        Chair dimensions: 43.18cm L x 44.45cm W x 46.99-54.61cm H (average 50.8cm)
        In meters: 0.4318 L x 0.4445 W x 0.508 H
        Reference: share_humanoid_amp_carryobject.py's box creation pattern
        """
        if self.asset_name == "chair":
            # Chair dimensions in meters (43.18 L x 44.45 W x 46.99–54.61 H cm)
            furniture_length = 0.4318
            furniture_width = 0.4445
            furniture_height = 0.508  # Average of 46.99-54.61 cm
            print(f"Loading chair asset: {furniture_length}m L x {furniture_width}m W x {furniture_height}m H")
        else:
            
            furniture_length = 1.905
            furniture_width = 0.9906
            furniture_height = 0.42
            print(f"Loading bed asset: {furniture_length}m L x {furniture_width}m W x {furniture_height}m H")

        # Create a single shared furniture asset and reuse it for all environments
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 30.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # Create box with furniture dimensions (shared asset)
        if self.asset_name == "chair":
            self._chair_asset = self.gym.create_box(self.sim, furniture_length, furniture_width, furniture_height, asset_options)
        else:
            self._bed_asset = self.gym.create_box(self.sim, furniture_length, furniture_width, furniture_height, asset_options)

        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Override to create single-humanoid environments with collision groups for interaction"""

        # Load bed asset before creating environments
        self._bed_handles = []
        self._load_bed_asset()

        # Call parent method to create basic environments
        super()._create_envs(num_envs, spacing, num_per_row)

        # Apply recipient weakness modification after all environments are created
        self._apply_recipient_weakness()
        # self._setup_collision_groups_and_positioning()

        return



    def get_num_actors_per_env(self):
        """Override to account for beds in environments"""
        # Calculate dynamically from actual tensor size like parent class
        if hasattr(self, '_root_states') and self._root_states is not None:
            num_actors = self._root_states.shape[0] // self.num_envs
            return num_actors
        else:
            # Fallback during initialization before tensors are setup
            return super().get_num_actors_per_env()

    def get_virtual_bodies_per_env(self):
        """Override to account for beds in environments"""
        num_actors = self.get_num_actors_per_env()
        if num_actors == 1:
            return 52
        elif num_actors == 2: # humanoid root (1) + bed (1)
            return 53
        elif num_actors == 53:  # humanoid root (1) + markers (52)
            return 104
        elif num_actors == 54:  # humanoid root (1) + bed (1) + markers (52)
            return 105 # 52 markers + 1 bed + 1 humanoid root

    def _setup_tensors(self):
        """Override to handle bed bodies in rigid body tensor"""
        # Call parent to setup basic tensors
        # super()._setup_tensors()

        # Actor root states
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._root_states = gymtorch.wrap_tensor(actor_root_state)

        # Calculate actual num_actors from tensor size (handles marker actors in non-headless mode)
        total_actors = self._root_states.shape[0]
        actual_num_actors_per_env = total_actors // self.num_envs

        print(f"[DEBUG _setup_tensors] num_envs: {self.num_envs}, actual actors per env: {actual_num_actors_per_env}")
        print(f"[DEBUG _setup_tensors] _root_states shape: {self._root_states.shape}")
        print(f"[DEBUG _setup_tensors] Total actors: {total_actors}")

        # Extract humanoid root states (always actor index 0 in each environment)
        # In non-headless mode: Actor 0=humanoid, Actor 1=bed, Actors 2+=markers
        self._humanoid_root_states = self._root_states.view(self.num_envs, actual_num_actors_per_env, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = actual_num_actors_per_env * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # DOF states
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        # Rigid body states with modified assertion for beds
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs

        print(f"[DEBUG] bodies_per_env with beds: {bodies_per_env}") # when headless = False, bodies_per_env = 105 (52 joint + bed + 52 markers)

        # Modified assertion: SMPL has 24 rigid bodies + bed body (1)
        # With markers: 24*2 + 1 = 49, without markers: 24 + 1 = 25
        expected_bodies_options = [24, 48, 52, 104, 24 + 1, 48 + 1, 52 +1, 104 + 1]  # SMPL bodies + bed
        assert bodies_per_env in expected_bodies_options, f"bodies_per_env should be one of {expected_bodies_options} (SMPL(x) + bed), got {bodies_per_env}"

        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        # History buffers if needed
        if hasattr(self, 'self_obs_v') and self.self_obs_v == 2:
            self._rigid_body_pos_hist = torch.zeros((self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device, dtype=torch.float)
            self._rigid_body_rot_hist = torch.zeros((self.num_envs, self.past_track_steps, self.num_bodies, 4), device=self.device, dtype=torch.float)
            self._rigid_body_vel_hist = torch.zeros((self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device, dtype=torch.float)
            self._rigid_body_ang_vel_hist = torch.zeros((self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device, dtype=torch.float)

        # Contact forces
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        # DOF force tensor (for power reward calculation)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        # Terminate buffer
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        # Build termination heights and contact body IDs
        self._build_termination_heights()

        # Initialize key body IDs and contact body IDs (required for AMP)
        contact_bodies = self.cfg["env"]["contact_bodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(self.key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)



    def _setup_character_props(self, key_bodies):
        """Override to ensure AMP observation size is calculated correctly with beds"""
        # Call parent implementation but ensure key_bodies only includes humanoid bodies
        # Filter out any bed-related bodies that might have been added
        humanoid_key_bodies = [body for body in key_bodies if 'bed' not in body.lower()]
        print(f"[DEBUG] Original key_bodies: {key_bodies}")
        print(f"[DEBUG] Filtered humanoid_key_bodies: {humanoid_key_bodies}")

        super()._setup_character_props(humanoid_key_bodies)

    def _build_bed_tensors(self):
        # 実測
        actors_per_env = self._root_states.shape[0] // self.num_envs
        self._bed_states = self._root_states.view(self.num_envs, actors_per_env, self._root_states.shape[-1])[..., -1, :]

        # 各 env の bed actor の"フラットな"インデックス
        base = to_torch(np.arange(self.num_envs), dtype=torch.int32, device=self.device) * actors_per_env
        self._bed_actor_ids = base + (actors_per_env - 1)

        # # 接触テンソルも bodies_per_env の末尾がベッド、として切る
        # bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        # contact_force_tensor = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        # self._bed_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., -1, :]

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        """Override to add bed spawning for recipient environments"""
        # Call parent method to build humanoid
        super()._build_env(env_id, env_ptr, humanoid_asset)

        # Spawn bed for all environments to maintain consistent actor count
        # For caregiver environments (even env_ids), bury the bed underground
        self._build_bed(env_id, env_ptr)

        return

    def _build_bed(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        col_filter = 0
        segmentation_id = 0

        # Furniture dimensions based on asset type
        if self.asset_name == "chair":
            furniture_height = 0.49  # Chair height
        else:
            furniture_height = 0.4572  # Bed height (default)

        if env_id % 2 == 1:
            # Recipient environment: bed should collide with recipient
            # env_spacing * 2だけx方向に移動
            default_pose.p.x = - self.cfg["env"].get("env_spacing", 5.0) * 2 + 0.3
            default_pose.p.z = furniture_height / 2.0
            if self.asset_name == "chair":
                default_pose.p.y = 0.1
            else:
                default_pose.p.y = 0.19
            # Use same collision group as humanoid (env_id // 2) so recipient can lie on bed
            bed_col_group = env_id // 2
        else:
            # Caregiver environment: bury bed underground to avoid collision
            default_pose.p.z = -10.0
            # Use different collision group
            bed_col_group = self.num_envs + (env_id // 2)

        # Use shared furniture asset for all environments (following CooHOI pattern)
        if self.asset_name == "chair":
            furniture_asset = self._chair_asset
            actor_name = "chair"
        else:
            furniture_asset = self._bed_asset
            actor_name = "bed"
        target_handle = self.gym.create_actor(env_ptr, furniture_asset, default_pose, actor_name, bed_col_group, col_filter, segmentation_id)

        # Set friction properties (following CooHOI pattern)
        props = self.gym.get_actor_rigid_body_properties(env_ptr, target_handle)
        for prop in props:
            prop.mass = 100.0  # Make bed heavy so it doesn't move
        self.gym.set_actor_rigid_body_properties(env_ptr, target_handle, props)

        self._bed_handles.append(target_handle)

        # Configure shape properties to reduce contact workload
        shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        if env_id % 2 == 0:
            # Caregiver env: disable collisions with first 16 groups (common Gym default)
            disable_all_16 = (1 << 16) - 1  # 0xFFFF
            for sp in shape_props:
                sp.filter = disable_all_16
            self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, shape_props)
        else:
            # Recipient env: stabilize contact and block ground (bit 0)
            for sp in shape_props:
                sp.contact_offset = 0.01
                sp.rest_offset = 0.0
                sp.friction = 1.0
                sp.rolling_friction = 0.0
                sp.torsion_friction = 0.0
                sp.restitution = 0.0
                # Block ground (bit 0). Also mark bed with a dedicated bed-only bit (bit 15)
                # so we can disable only bed<->finger contacts without affecting human-human.
                sp.filter = (sp.filter | 1 | (1 << 15))
            self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, shape_props)

            # Disable bed collisions with recipient finger joints (block all fingers; keep wrists only)
            # Ensure mapping is available
            if not hasattr(self, '_body_name_to_id') or len(getattr(self, '_body_name_to_id', {})) == 0:
                self._setup_body_name_to_id_mapping()

            humanoid_handle = self.humanoid_handles[env_id]
            h_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)

            # Use a dedicated bed-only filter bit (bit 15) that is not used by humanoids
            bed_group_bit = 1 << 15

            # Block all finger joints (1,2,3) for Index/Middle/Ring/Pinky/Thumb, both hands
            block_joint_names = [
                # Left hand
                'L_Index1', 'L_Index2', 'L_Index3',
                'L_Middle1', 'L_Middle2', 'L_Middle3',
                'L_Ring1', 'L_Ring2', 'L_Ring3',
                'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
                'L_Thumb1', 'L_Thumb2', 'L_Thumb3',
                # Right hand
                'R_Index1', 'R_Index2', 'R_Index3',
                'R_Middle1', 'R_Middle2', 'R_Middle3',
                'R_Ring1', 'R_Ring2', 'R_Ring3',
                'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
                'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
            ]

            for name in block_joint_names:
                body_id = self._body_name_to_id.get(name)
                h_shape_props[body_id].filter = h_shape_props[body_id].filter | bed_group_bit

            self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, h_shape_props)
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
        """Override to implement pair-based termination logic and ankle height termination"""
        # Call parent method to compute individual resets
        super()._compute_reset()

        # Add ankle height termination condition (applies to both training and testing)
        # Get ankle body IDs
        l_ankle_id = None
        r_ankle_id = None
        if hasattr(self, '_body_names') and self._body_names is not None:
            if 'L_Ankle' in self._body_names:
                l_ankle_id = self._body_names.index('L_Ankle')
            if 'R_Ankle' in self._body_names:
                r_ankle_id = self._body_names.index('R_Ankle')

        # Check ankle heights and terminate if either ankle exceeds threshold
        if l_ankle_id is not None and r_ankle_id is not None:
            l_ankle_height = self._rigid_body_pos[:, l_ankle_id, 2]  # z-coordinate
            r_ankle_height = self._rigid_body_pos[:, r_ankle_id, 2]

            # Terminate if either ankle height exceeds threshold
            ankle_too_high = torch.logical_or(
                l_ankle_height > self.ankle_height_termination_threshold,
                r_ankle_height > self.ankle_height_termination_threshold
            )

            # Only apply after first few steps to avoid initial state issues
            ankle_too_high = torch.logical_and(ankle_too_high, self.progress_buf > 1)

            # Debug: Print ankle height terminations periodically
            if torch.any(ankle_too_high) and hasattr(self, 'progress_buf') and self.progress_buf[0] % 100 == 0:
                num_terminated = torch.sum(ankle_too_high).item()
                max_l_ankle = l_ankle_height.max().item()
                max_r_ankle = r_ankle_height.max().item()
                print(f"[Ankle Height Debug] Step {self.progress_buf[0].item()}: {num_terminated} envs terminated (max L: {max_l_ankle:.3f}m, max R: {max_r_ankle:.3f}m)")

            # Update termination buffer
            self._terminate_buf = torch.where(ankle_too_high, torch.ones_like(self._terminate_buf), self._terminate_buf)
            self.reset_buf = torch.where(ankle_too_high, torch.ones_like(self.reset_buf), self.reset_buf)

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

        return

    def get_motion_file(self):
        """Override to use interaction motion file"""
        return self.hhi_assist_bed_data_path


    def _compute_observations(self, env_ids=None):
        """Override to add partner relative position/velocity to observations"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        # Get the base self observations (proprioception)
        self_obs = super()._compute_humanoid_obs(env_ids)

        # Check if self_obs_buf size needs to be updated
        if self.self_obs_buf.shape[1] != self_obs.shape[1]:
            print(f"Resizing self_obs_buf from {self.self_obs_buf.shape} to match new self observation size {self_obs.shape[1]}")
            self.self_obs_buf = torch.zeros((self.num_envs, self_obs.shape[1]),
                                          device=self.device, dtype=self_obs.dtype)

        self.self_obs_buf[env_ids] = self_obs

        # Get task observations (reference differences) with role-specific handling
        task_obs =  super()._compute_task_obs(env_ids, save_buffer=True)

        # Compute partner relative position/velocity observations with safety
        partner_obs = self._compute_partner_obs_base(env_ids)

        # Extract base self observation (without contact info)
        base_self_obs = self_obs

        base_partner_obs = partner_obs

        # Add role indicator: 0.0 for recipient (odd env_id), 1.0 for caregiver (even env_id)
        role_indicator = (env_ids % 2 == 0).float().unsqueeze(-1)  # Shape: [num_envs, 1]
        
        global_root_pos = self._rigid_body_pos[env_ids, 0, :2]
        
        assert base_self_obs.shape[1] + task_obs.shape[1] == 2026, print(f"base_self_obs.shape[1] : {base_self_obs.shape[1]}, task_obs.shape[1] : {task_obs.shape[1]}")
        obs = torch.cat([base_self_obs, task_obs, base_partner_obs, global_root_pos, role_indicator], dim=-1)
        self.obs_buf[env_ids] = obs

        return obs
    

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

    def _compute_partner_obs_base(self, env_ids):
        """Compute partner observations (self_obs + task_obs concatenated)

        IMPORTANT: Partner's self_obs is computed FRESH (not from buffer) to avoid
        stale observations. Partner's self_obs now includes:
        - All standard observations (body positions, velocities, etc.)
        - Root XY position (absolute global position)
        - Global root rotation (if local_root_obs=false)

        This allows the agent to:
        1. Know partner's absolute position via root XY in partner's self_obs
        2. Compute relative position by comparing own root_xy with partner's root_xy
        3. Observe partner's reference tracking performance via task_obs

        Offset compensation: Task_obs is computed with offset handling in
        _get_state_from_motionlib_cache, so no additional offset needed here.
        """
        # Get partner environment IDs
        partner_env_ids = torch.where(env_ids % 2 == 0, env_ids + 1, env_ids - 1)

        # Ensure partner IDs are within bounds
        valid_partners = partner_env_ids < self.num_envs
        if not valid_partners.all():
            invalid_env_ids = env_ids[~valid_partners]
            raise Exception(f"Invalid partner environments: {invalid_env_ids.tolist()}")

        # Compute partner's self observations FRESH (includes root XY position)
        partner_self_obs = self._compute_humanoid_obs(partner_env_ids)

        # Get partner's task observations (reference motion diff, offset handled in motion lib)
        partner_task_obs = super()._compute_task_obs(partner_env_ids, save_buffer=False)
        
        global_root_pos = self._rigid_body_pos[partner_env_ids, 0, :2]
        # Concatenate partner's self and task observations
        partner_obs = torch.cat([partner_self_obs, partner_task_obs, global_root_pos], dim=-1)
        

        return partner_obs

    def _get_partner_obs_size(self):
        """Override to calculate partner observation size based on version"""
        self_obs_size = self.get_self_obs_size()
        task_obs_size = self.task_obs_size
        global_root_pos_size = 2
        return self_obs_size + task_obs_size + global_root_pos_size

    def _get_partner_obs_base(self):
        """Return partner observation size (self_obs + task_obs)"""
        self_obs_size = self.get_self_obs_size()
        task_obs_size = self.task_obs_size
        return self_obs_size + task_obs_size


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


    def get_obs_size(self):
        """Override to return observation size for humanx (self + task + partner_self + partner_task + role_indicator)"""
        # Get self observation size (proprioception)
        self_obs_size = self.get_self_obs_size()
        if self_obs_size is None:
            self_obs_size = 0

        # Task observation size
        self.task_obs_size = 1248

        # Partner observation size (partner_self_obs + partner_task_obs)
        partner_obs_size = self._get_partner_obs_size()

        # Role indicator size (1 dimension)
        role_indicator_size = 1
        
        global_root_pos_size = 2

        # Total: self_obs + task_obs + partner_self_obs + partner_task_obs + role_indicator
        total_size = self_obs_size + self.task_obs_size + partner_obs_size + role_indicator_size + global_root_pos_size

        return total_size


    def get_self_obs_size(self):
        """Override to account for additional root XY position"""
        base_size = super().get_self_obs_size()
        return base_size


    def _compute_reward(self, actions):
        """Override to implement SimpleLiftUp reward or normal interaction reward"""
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

        # Create weight tensors
        self_weights = torch.where(is_recipient, 1.0, 0.5)  # recipient=1.0, caregiver=0.5
        partner_weights = torch.where(is_recipient, 0.0, 0.5)  # recipient=0.0, caregiver=0.5

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

    def _apply_recipient_weakness(self):
        """Apply weakness to recipients by modifying their DOF properties"""
        if self.control_mode != "isaac_pd":
            return  # Only works with Isaac PD control
        
        # Get recipient weakness scale from config
        recipient_weakness_scale = self.cfg["env"].get("recipient_weakness_scale", 0.5)
            
        # Get lower body joint indices (DOF indices, not body indices)
        lower_body_joint_names = ['L_Knee', 'L_Ankle', 'L_Toe', 'R_Knee', 'R_Ankle', 'R_Toe']
        lower_body_dof_indices = []
        
        for joint_name in lower_body_joint_names:
            if joint_name in self._dof_names:
                joint_idx = self._dof_names.index(joint_name)
                # Each joint has 3 DOFs (x, y, z rotation)
                for dof_offset in range(3):
                    dof_idx = joint_idx * 3 + dof_offset
                    if dof_idx < self.num_dof:
                        lower_body_dof_indices.append(dof_idx)
            else:
                print(f"Warning: Joint {joint_name} not found in DOF names")
                print(self._dof_names)
                assert False
        
        upper_body_joint_names = ['Torso', 'Chest', 'Spine']
        upper_body_dof_indices = []
        for joint_name in upper_body_joint_names:
            if joint_name in self._dof_names:
                joint_idx = self._dof_names.index(joint_name)
                for dof_offset in range(3):
                    dof_idx = joint_idx * 3 + dof_offset
                    if dof_idx < self.num_dof:
                        upper_body_dof_indices.append(dof_idx)
                        
        hip_body_joint_names = ['L_Hip', 'R_Hip']
        hip_body_dof_indices = []
        for joint_name in hip_body_joint_names:
            if joint_name in self._dof_names:
                joint_idx = self._dof_names.index(joint_name)
                for dof_offset in range(3):
                    dof_idx = joint_idx * 3 + dof_offset
                    if dof_idx < self.num_dof:
                        hip_body_dof_indices.append(dof_idx)
            
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
                    dof_prop['effort'][dof_idx] = 80
                
                for dof_idx in upper_body_dof_indices:
                    dof_prop['stiffness'][dof_idx] *= recipient_weakness_scale
                    dof_prop['damping'][dof_idx] *= recipient_weakness_scale
                    dof_prop['effort'][dof_idx] = 40
                for dof_idx in hip_body_dof_indices:
                    dof_prop['stiffness'][dof_idx] *= recipient_weakness_scale
                    dof_prop['damping'][dof_idx] *= recipient_weakness_scale
                    dof_prop['effort'][dof_idx] = 20
                # Apply modified properties
                self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)
                modified_count += 1
                
        print(f"Applied recipient weakness (scale={recipient_weakness_scale}) to {modified_count} recipients, affecting {len(lower_body_dof_indices)} lower body DOFs")

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



    def _sample_ref_state(self, env_ids):
        """Override to apply consistent motion initialization in SimpleLiftUp mode"""
        # Get parent class behavior first
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = super()._sample_ref_state(env_ids)

        # In SimpleLiftUp mode, use trajectory buffer or start from time 0
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
        print(f"[_create_sim] SimpleLiftUp mode detected - applying mass scaling")
        self._apply_recipient_mass_scaling()
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
