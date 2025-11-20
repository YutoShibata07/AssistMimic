"""
Knowledge Distillation Environment for SimpleLiftUp with DAgger + PPO
Extends HumanoidImInterx to support multiple teacher policies based on motion groups.
Teacher policies are loaded and used to provide expert actions for imitation learning.
"""

import torch
import copy
import re
import numpy as np
from collections import defaultdict
from phc.env.tasks.humanoid_im_interx import HumanoidImInterx
from phc.utils.flags import flags


class HumanoidImInterxDist(HumanoidImInterx):
    """
    Distillation environment that provides expert actions from multiple teacher policies.
    Motion groups (e.g., G012, G029) are extracted from motion filenames and used to
    select the appropriate teacher policy for providing expert actions.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Store teacher policy configuration before parent initialization
        self.teacher_configs = cfg["env"].get("teacher_policies", {})
        self.default_teacher_group = cfg["env"].get("default_teacher_group", "G029")
        
        # Check if we're in test mode (don't load teachers for testing)
        self.is_test_mode = cfg.get("test", False)

        # Initialize parent class first
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # Initialize tracking for group assignments per environment
        self._env_to_group = {}  # Maps env_id -> group name (e.g., "G012")
        self._env_to_teacher = {}  # Maps env_id -> teacher policy

        # Only load teacher policies and extract groups if not in test mode
        if not self.is_test_mode:
            # Load teacher policies after parent initialization
            self._load_teacher_policies()
            # Extract group names from motion assignments
            self._extract_motion_groups()

        if not self.is_test_mode:
            print(f"Loaded {len(self._teacher_policies)} teacher policies for distillation")
            print(f"Environment-to-group assignments: {len(self._env_to_group)} environments")
        else:
            print("Test mode: Skipping teacher policy loading")
            self._teacher_policies = {}

        return

    def _load_teacher_policies(self):
        """Load multiple teacher policies from checkpoints"""
        self._teacher_policies = {}

        print("=" * 80)
        print("Loading teacher policies for knowledge distillation")
        print("=" * 80)

        for group_name, teacher_config in self.teacher_configs.items():
            checkpoint_path = teacher_config.get("checkpoint_path")
            obs_size = teacher_config.get("obs_size", 2026)
            action_size = teacher_config.get("action_size", 153)

            # IMPORTANT: Load checkpoint to current device (handles distributed training)
            # In distributed training, each rank has its own device (cuda:0, cuda:1, etc.)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Create a simple wrapper for the teacher policy
            # We'll load the network builder to reconstruct the exact architecture
            # Pass the full checkpoint to access running_mean_std at top level
            teacher_model = self._build_teacher_network(checkpoint, obs_size, action_size)
            teacher_model.to(self.device)
            teacher_model.eval()

            # Handle both single group (e.g., "G029") and multiple groups (e.g., "G044-045-046")
            # Extract individual group IDs and map each to the same teacher policy
            group_ids = self._parse_group_name(group_name)

            for gid in group_ids:
                self._teacher_policies[gid] = teacher_model

            # Debug: Print which device the teacher is on
            import os
            rank = int(os.environ.get('LOCAL_RANK', 0))
            print(f"✓ [Rank {rank}] Loaded teacher policy for group {group_name} (IDs: {group_ids}) from {checkpoint_path} to device {self.device}")

        # Ensure default teacher exists
        if self.default_teacher_group not in self._teacher_policies:
            if len(self._teacher_policies) > 0:
                self.default_teacher_group = list(self._teacher_policies.keys())[0]
                print(f"Warning: Default teacher group not found, using {self.default_teacher_group}")
            else:
                raise ValueError("No teacher policies loaded! Cannot proceed with distillation.")

        print("=" * 80)
        return

    def _build_teacher_network(self, checkpoint, _obs_size, action_size):
        """
        Build teacher network from checkpoint using amp_network_pnn_multi_builder.
        This ensures the teacher network has exactly the same architecture as the trained model.

        Args:
            checkpoint: Full checkpoint dict (contains 'running_mean_std', 'model', etc.)
            _obs_size: NOT USED - kept for API compatibility
            action_size: Action dimension (153 for SMPL-X)

        Returns:
            network: Teacher network ready for inference
        """
        from phc.learning.amp_network_pnn_multi_builder import AMPPNNMultiBuilder
        from phc.utils.running_mean_std import RunningMeanStd

        # Extract model weights from checkpoint
        if "model" in checkpoint:
            model_state_dict = checkpoint["model"]
        else:
            model_state_dict = checkpoint

        # CRITICAL: Infer actual observation size from checkpoint's running_mean_std
        # The teacher was trained with specific observation dimensions that must match exactly
        if 'running_mean_std' in checkpoint:
            actual_obs_size = checkpoint['running_mean_std']['running_mean'].shape[0]
            print(f"✓ Inferred teacher observation size from checkpoint: {actual_obs_size}")
        else:
            raise ValueError("No running_mean_std found in checkpoint")

        # Load network configuration from Hydra learning config instead of hardcoding
        # Source: /data/user_data/yutos/MultiLiftUpRL/phc/data/cfg/learning/im_simpleliftup_mlp.yaml
        # cfg structure: cfg['learning']['params']['network']
        network_config = copy.deepcopy(self.cfg["learning"]["params"]["network"])  # type: ignore[index]
        # Ensure aux_mlp output size matches current action size
        if "aux_mlp" in network_config and isinstance(network_config["aux_mlp"], dict):
            network_config["aux_mlp"]["output_size"] = action_size

        cont_space = network_config.get("space", {}).get("continuous", {})
        if cont_space.get("mu_activation", None) is None:
            cont_space["mu_activation"] = 'None'
        if cont_space.get("sigma_activation", None) is None:
            cont_space["sigma_activation"] = 'None'
        # Write back in case nested dicts were copies
        if "space" in network_config and "continuous" in network_config["space"]:
            network_config["space"]["continuous"] = cont_space

        # Task observation size details matching teacher training environment
        # env_im_interx_simpleliftup.yaml configuration
        base_self_obs_size = 778
        task_obs_size = 1248  # obs_v=6, enableTaskObs=true
        aux_features_size = actual_obs_size - base_self_obs_size - task_obs_size

        task_obs_size_detail = {
            'fut_tracks': False,
            'obs_v': 6,
            'num_traj_samples': 10,
            'track_bodies': [],
            'num_prim': 1,
            'training_prim': 0,
            'models_path': [''],
            'actors_to_load': 0,
            'has_lateral': True,
            'partner_obs_v': 3,  # env_im_interx_simpleliftup.yaml
            'task_obs_size': task_obs_size,
        }

        # print(f"Teacher network configuration:")
        # print(f"  base_self_obs_size: {base_self_obs_size}")
        # print(f"  task_obs_size: {task_obs_size}")
        # print(f"  aux_features_size: {aux_features_size}")
        # print(f"  total_obs_size: {actual_obs_size}")

        # Create network builder
        builder = AMPPNNMultiBuilder()
        builder.params = network_config

        # Load running_mean_std from checkpoint if available
        if 'running_mean_std' in checkpoint:
            running_mean_std = RunningMeanStd((actual_obs_size,))
            running_mean_std.running_mean = checkpoint['running_mean_std']['running_mean'].to(self.device)
            running_mean_std.running_var = checkpoint['running_mean_std']['running_var'].to(self.device)
            running_mean_std.count = checkpoint['running_mean_std']['count'].to(self.device)
        else:
            raise ValueError("No running_mean_std found in checkpoint")

        # Build the network with correct observation size
        # amp_input_shape must match the discriminator input from checkpoint
        # Infer from checkpoint discriminator weights if available
        if '_disc_mlp.0.weight' in model_state_dict:
            disc_input_size = model_state_dict['_disc_mlp.0.weight'].shape[1]
            amp_input_shape = (disc_input_size,)
            print(f"✓ Inferred AMP input shape from checkpoint: {amp_input_shape}")
        elif 'a2c_network.module._disc_mlp.0.weight' in model_state_dict:
            disc_input_size = model_state_dict['a2c_network.module._disc_mlp.0.weight'].shape[1]
            amp_input_shape = (disc_input_size,)
            print(f"✓ Inferred AMP input shape from checkpoint (DDP): {amp_input_shape}")
        else:
            amp_input_shape = (1586,)  # Fallback
            print(f"✗ Warning: Using default AMP input shape: {amp_input_shape}")

        network = builder.build('teacher',
                               input_shape=(actual_obs_size,),
                               actions_num=action_size,
                               self_obs_size=base_self_obs_size,
                               task_obs_size=task_obs_size,
                               task_obs_size_detail=task_obs_size_detail,
                               mean_std=running_mean_std,
                               amp_input_shape=amp_input_shape)

        # Ensure trajectory CNN is constructed to match checkpoint keys
        # Input feature dim: 22 bodies * 13 features * 2 (self + partner) = 572
        try:
            if getattr(network, 'trajectory_cnn', None) is None or not getattr(network, 'trajectory_cnn_initialized', False):
                network._create_trajectory_cnn(22 * 13 * 2)
        except Exception as e:
            raise ValueError(f"Error initializing trajectory CNN: {e}")

        # Load state dict with strict=True to catch any mismatch
        # Remove 'a2c_network.module.' or 'a2c_network.' prefix and exclude running_mean_std
        cleaned_state_dict = {}
        for key, value in model_state_dict.items():
            if key == 'running_mean_std':
                continue  # Already loaded above

            new_key = key
            # Remove 'a2c_network.module.' prefix (DDP models)
            if new_key.startswith('a2c_network.module.'):
                new_key = new_key[len('a2c_network.module.'):]
            # Remove 'a2c_network.' prefix (non-DDP models)
            elif new_key.startswith('a2c_network.'):
                new_key = new_key[len('a2c_network.'):]

            cleaned_state_dict[new_key] = value

        # Load weights with strict=True to ensure exact match
        missing_keys, unexpected_keys = network.load_state_dict(cleaned_state_dict, strict=True)

        # Check for critical missing keys
        critical_missing = [k for k in missing_keys if not k.startswith('trajectory_cnn') and not k.startswith('_')]
        if len(critical_missing) > 0:
            raise ValueError(f"Missing {len(critical_missing)} critical keys in teacher network")

        if len(unexpected_keys) > 0:
            raise ValueError(f"Unexpected {len(unexpected_keys)} keys in checkpoint")

        print(f"✓ Teacher network loaded successfully")
        print(f"  Loaded parameters: {len(cleaned_state_dict)}")

        return network

    def _extract_motion_groups(self):
        """
        Extract group names from motion assignments.
        Motion keys format: G012T007A035R003_caregiver -> group = G012
        """
        print("Extracting motion groups from environment assignments...")

        for env_id, motion_key in self.env_motion_assignments.items():
            group = self._extract_group_from_motion_key(motion_key)

            # Assign teacher policy for this environment
            teacher = self._teacher_policies[group]

            self._env_to_group[env_id] = group
            self._env_to_teacher[env_id] = teacher

        # Count group distribution
        group_counts = {}
        for group in self._env_to_group.values():
            group_counts[group] = group_counts.get(group, 0) + 1

        print(f"Group distribution: {group_counts}")
        return

    def _parse_group_name(self, group_name):
        """
        Parse group name to extract individual group IDs.
        Examples:
          - "G029" -> ["G029"]
          - "G044-045-046" -> ["G044", "G045", "G046"]

        Args:
            group_name: Group name string from config

        Returns:
            List of individual group IDs
        """
        # Check if group_name contains multiple groups separated by hyphens
        # Pattern: G044-045-046 or G029
        if '-' in group_name:
            # Extract the prefix (e.g., "G") and digits
            match = re.match(r'(G)(\d+)', group_name)
            if not match:
                raise ValueError(f"Invalid group name format: {group_name}")

            prefix = match.group(1)  # "G"

            # Split by hyphen and build full group IDs
            parts = group_name.split('-')
            group_ids = []

            for part in parts:
                if part.startswith('G'):
                    # First part: "G044"
                    group_ids.append(part)
                else:
                    # Subsequent parts: "045", "046"
                    group_ids.append(f"{prefix}{part}")

            return group_ids
        else:
            # Single group: "G029"
            return [group_name]

    def _extract_group_from_motion_key(self, motion_key):
        """
        Extract group name from motion key.
        Examples:
          - G012T007A035R003_caregiver -> G012
          - G029T001A001R001_recipient -> G029
        """
        # Use regex to extract group (G followed by digits)
        match = re.match(r'(G\d+)', motion_key)
        return match.group(1)

    def _get_expert_actions(self, env_ids, obs):
        """
        Get expert actions from teacher policies for given environments.

        Args:
            env_ids: Tensor of environment IDs or None (reset all)
            obs: Observation tensor [batch, obs_dim]

        Returns:
            expert_mus: Expert action means [batch, action_dim]
            expert_actions: Expert actions (same as mus for deterministic policy) [batch, action_dim]
        """
        # Handle env_ids=None case (reset all environments)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        batch_size = len(env_ids)
        action_dim = 153  # SMPL-X humanoid action dimension

        expert_mus = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=self.device)
        expert_actions = torch.zeros((batch_size, action_dim), dtype=torch.float32, device=self.device)

        # Group environments by teacher policy for efficient batching (vectorized)
        teacher_to_indices = {}

        # Convert env_ids to numpy for fast indexing
        if torch.is_tensor(env_ids):
            env_ids_np = env_ids.cpu().numpy()
        else:
            env_ids_np = np.array(env_ids)

        # Vectorized teacher lookup
        teacher_to_indices = defaultdict(list)

        for i, env_id_val in enumerate(env_ids_np):
            teacher = self._env_to_teacher[env_id_val]
            teacher_to_indices[teacher].append(i)

        # Convert lists to numpy arrays for faster indexing
        teacher_to_indices = {k: np.array(v, dtype=np.int64) for k, v in teacher_to_indices.items()}

        # Get expert actions from each teacher
        with torch.no_grad():
            for teacher, indices in teacher_to_indices.items():
                batch_obs = obs[indices]

                # CRITICAL: Normalize observations using teacher's running_mean_std
                # Teacher network was trained with normalized observations
                # Must apply the same normalization during inference
                obs_mean = teacher.running_mean.to(batch_obs.dtype)
                obs_var = teacher.running_var.to(batch_obs.dtype)
                # Normalize: (obs - mean) / sqrt(var + epsilon)
                batch_obs_normalized = (batch_obs - obs_mean) / torch.sqrt(obs_var + 1e-5)
                batch_obs_normalized = torch.clamp(batch_obs_normalized, min=-5.0, max=5.0)

                # Use eval_actor method to get action means from teacher network
                # The network expects obs_dict format with normalized observations
                obs_dict = {'obs': batch_obs_normalized}
                mu, _ = teacher.eval_actor(obs_dict)

                # Store in output tensors (vectorized indexing)
                expert_mus[indices] = mu
                expert_actions[indices] = mu  # Deterministic policy (use mean)
        return expert_mus, expert_actions

    def reset(self, env_ids=None):
        """
        Override reset to return expert actions along with observations.
        This is required for DAgger algorithm to access expert policy.
        """
        # Call parent reset to get observations
        obs_dict = super().reset(env_ids)

        # Handle the case where obs_dict might be None during initialization
        if obs_dict is None:
            return obs_dict

        # Get expert actions for reset observations
        if isinstance(obs_dict, dict):
            obs_tensor = obs_dict.get("obs")
            if obs_tensor is None:
                obs_tensor = obs_dict if isinstance(obs_dict, torch.Tensor) else None
        else:
            obs_tensor = obs_dict if isinstance(obs_dict, torch.Tensor) else None

        # If we still don't have observations, return without expert actions
        if obs_tensor is None:
            return obs_dict

        expert_mus, expert_actions = self._get_expert_actions(env_ids, obs_tensor)

        # Package expert information
        expert_dict = {
            "mus": expert_mus,
            "actions": expert_actions,
        }

        return obs_dict, expert_dict

    def step(self, actions):
        """
        Override step to return expert actions along with observations.
        This is required for DAgger algorithm to access expert policy at each step.

        Note: Isaac Gym tasks don't return from step(), they update buffers.
        VecTaskWrapper reads from these buffers. We only need to compute expert actions.
        """
        # Call parent step (updates buffers but returns None)
        super().step(actions)

        # Parent step() doesn't return anything in Isaac Gym
        # VecTaskWrapper will read obs_buf, rew_buf, reset_buf, and extras
        # We don't need to return anything here; VecTaskWrapper handles it
        # The only modification we need is in VecTaskWrapper.step() to add expert actions