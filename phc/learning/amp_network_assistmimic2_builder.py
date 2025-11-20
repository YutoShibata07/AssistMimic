
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np
import copy
from phc.learning.pnn import PNN
from rl_games.algos_torch import torch_ext
import gc

DISC_LOGIT_INIT_SCALE = 1.0


class AMPAssistMimic2PNNBuilder(AMPBuilder):
    """
    Simplified PNN builder for humanx task with single primitive (num_prim=1).
    Input: concat(self_obs, task_obs, partner_self_obs, partner_task_obs)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPAssistMimic2PNNBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']

            # Extract task observation details
            self.fut_tracks = self.task_obs_size_detail.get('fut_tracks', False)
            self.obs_v = self.task_obs_size_detail.get('obs_v', 6)
            self.num_traj_samples = self.task_obs_size_detail.get('num_traj_samples', 10)
            self.track_bodies = self.task_obs_size_detail.get('track_bodies', [])

            # Humanx specific: single primitive
            self.num_prim = 1
            self.training_prim = 0
            self.has_lateral = False

            # Weight sharing configuration
            self.weight_share = params.get("weight_share", True)  # Default to True for backward compatibility
            print(f"DEBUG: weight_share enabled: {self.weight_share}")

            # Freeze recipient configuration
            self.freeze_recipient = params.get("freeze_recipient", False)  # Default to False for backward compatibility
            print(f"DEBUG: freeze_recipient enabled: {self.freeze_recipient}")

            # Input shape: self_obs + task_obs + partner_obs (partner_self + partner_task) + role_indicator
            # Partner obs size = self_obs_size + task_obs_size
            self.global_root_pos_size = 2
            self.partner_obs_size = self.self_obs_size + self.task_obs_size + self.global_root_pos_size
            # PNN input size (without role indicator, as we remove it before passing to PNN)
            self.future_trajectory_size = 20 * 22 * 13 * 2
            # Total observation size including role indicator (for environment and normalization)
            total_obs_size_with_role = self.self_obs_size + self.task_obs_size + self.partner_obs_size + 1 + self.global_root_pos_size + self.future_trajectory_size
            self.future_trajectory_cnn_output_size = 128
            pnn_input_size = self.self_obs_size + self.task_obs_size + self.partner_obs_size + self.global_root_pos_size + self.future_trajectory_cnn_output_size

            # Recipient PNN input size depends on freeze_recipient setting
            if self.freeze_recipient:
                # Frozen: lightweight input without future trajectory features
                recipient_pnn_input_size = self.self_obs_size + self.task_obs_size
            else:
                # Trainable: same input as caregiver
                recipient_pnn_input_size = pnn_input_size

            # Set input shape to include role indicator for normalization statistics
            kwargs['input_shape'] = (total_obs_size_with_role,)
            
            self.critic_obs_size = total_obs_size_with_role

            super().__init__(params, **kwargs)
            
            self._rebuild_critic_mlp()

            # PNN uses input without role indicator
            caregiver_mlp_args = {
                'input_size': self._calc_input_size((pnn_input_size,), self.actor_cnn),
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
            }
            recipient_mlp_args = {
                'input_size': self._calc_input_size((recipient_pnn_input_size,), self.actor_cnn),
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
            }
            del self.actor_mlp
            self.discrete = params.get("discrete", False)

            # Create dummy aux_mlp_units to satisfy PNN's zero_fc requirement
            # This enables the aux fusion pathway even though we won't use it
            aux_mlp_units = self.units  # Use same units as main MLP

            # Separate weights - create separate PNNs for caregiver and recipient
            print("Creating separate PNNs for caregiver and recipient (weight_share=False)")
            self.caregiver_pnn = PNN(
                caregiver_mlp_args,
                output_size=kwargs['actions_num'],
                numCols=self.num_prim,
                has_lateral=self.has_lateral,
                aux_mlp_units=aux_mlp_units
            )
            self.recipient_pnn = PNN(
                recipient_mlp_args,
                output_size=kwargs['actions_num'],
                numCols=self.num_prim,
                has_lateral=self.has_lateral,
                aux_mlp_units=aux_mlp_units
            )

            # Freeze nothing since we only have one primitive and it's trainable
            self.caregiver_pnn.freeze_pnn(self.training_prim)
            self.recipient_pnn.freeze_pnn(self.training_prim)

            # Optionally freeze recipient_pnn based on config
            if self.freeze_recipient:
                for param in self.recipient_pnn.parameters():
                    param.requires_grad = False
                self.recipient_pnn.eval()  # Set to eval mode
                print("✓ Recipient PNN frozen: requires_grad=False, eval mode enabled")
            else:
                print("✓ Recipient PNN trainable: requires_grad=True")

            # Set reference to caregiver PNN for compatibility
            self.pnn = self.caregiver_pnn

            # Pre-initialize trajectory CNN with expected input feature dimension
            # Expected input: (self_traj + partner_traj) = 2 * (20 frames * 22 bodies * 13 features) features per frame
            # After reshape: [batch, 20, 22*13*2] = [batch, 20, 572]
            expected_trajectory_feature_dim = 22 * 13 * 2  # 572 features per frame (self + partner)
            self._create_trajectory_cnn(expected_trajectory_feature_dim)
            self.trajectory_cnn_initialized = True

            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var
            self.running_count = kwargs['mean_std'].count
            
            # Determine checkpoint path
            use_exp_checkpoint = False
            exp_name = None
            
            # Method 1: Check task_obs_size_detail for experiment info
            if hasattr(self.task_obs_size_detail, 'get'):
                exp_name = self.task_obs_size_detail.get('exp_name', None)
            elif isinstance(self.task_obs_size_detail, dict):
                exp_name = self.task_obs_size_detail.get('exp_name', None)
            
            # Method 2: Check environment variables or command line args
            import os
            import sys
            if not exp_name:
                # Check command line arguments for exp_name
                for arg in sys.argv:
                    if 'exp_name=' in arg:
                        exp_name = arg.split('exp_name=')[1]
                        break
            if exp_name and exp_name != 'phc_x_pnn':
                # Construct experiment checkpoint path
                checkpoint_path = f"output/HumanoidIm/{exp_name}/Humanoid.pth"
                use_exp_checkpoint = True
                print(f"Loading from experiment checkpoint: {checkpoint_path}")
            else:
                # Use hardcoded checkpoint
                checkpoint_path = "output/HumanoidIm/phc_interx_recipient_finetune2/Humanoid.pth"
                print(f"Loading from hardcoded checkpoint: {checkpoint_path}")
                
            if not os.path.exists(checkpoint_path):
                print(f"✗ Checkpoint file not found: {checkpoint_path}")
                checkpoint_path = "output/HumanoidIm/phc_interx_recipient_finetune2/Humanoid.pth"
                # checkpoint_path = "/data/user_data/yutos/MultiLiftUpRL/output/HumanoidIm/G009-rsi-fixed/Humanoid_00000450.pth"
                print(f"Using hardcoded checkpoint: {checkpoint_path}")
            
            # Load checkpoint with proper device mapping to avoid CUDA device mismatch
            if torch.cuda.is_available():
                current_device = f'cuda:{torch.cuda.current_device()}'
                checkpoint = torch.load(checkpoint_path, map_location=current_device)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            try:
                if self.weight_share:
                    # Load checkpoint into shared PNN
                    self.pnn.load_base_net(checkpoint_path, actors=1)
                    print(f"✓ Successfully loaded checkpoint into shared PNN")
                else:
                    # For separate weights, try to load role-specific weights first
                    # If not found, fall back to loading the same base weights for both
                    if 'model' in checkpoint:
                        # Check if checkpoint has role-specific PNN weights
                        model_keys = list(checkpoint['model'].keys())
                        has_caregiver_pnn = any('caregiver_pnn' in k for k in model_keys)
                        has_recipient_pnn = any('recipient_pnn' in k for k in model_keys)
                        has_pnn = any('pnn' in k and 'caregiver_pnn' not in k and 'recipient_pnn' not in k for k in model_keys)

                        if has_caregiver_pnn and has_recipient_pnn:
                            # Load role-specific weights
                            print(f"Found role-specific PNN weights in checkpoint")
                            self._load_role_specific_pnn_weights(checkpoint)
                            print(f"✓ Successfully loaded role-specific PNN weights")
                        elif has_pnn:
                            # Load shared PNN weights into both caregiver and recipient
                            print(f"Found shared PNN weights in checkpoint, loading into both caregiver and recipient PNNs")
                            self._load_shared_pnn_weights(checkpoint)
                            print(f"✓ Successfully loaded shared PNN weights into both caregiver and recipient PNNs")
                        else:
                            # Load same base weights for both (training from scratch or fine-tuning)
                            print(f"No PNN weights found in checkpoint, loading base network weights for both")
                            self.caregiver_pnn.load_base_net(checkpoint_path, actors=1)
                            self.recipient_pnn.load_base_net(checkpoint_path, actors=1)
                            print(f"✓ Successfully loaded base checkpoint into both caregiver and recipient PNNs")
                    else:
                        # Old checkpoint format, load base weights
                        self.caregiver_pnn.load_base_net(checkpoint_path, actors=1)
                        self.recipient_pnn.load_base_net(checkpoint_path, actors=1)
                        print(f"✓ Successfully loaded checkpoint into both caregiver and recipient PNNs")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                import traceback
                traceback.print_exc()

            # Load running_mean and running_var from checkpoint
            if 'running_mean_std' in checkpoint:
                print("\n=== Loading normalization statistics (running_mean/running_var) ===")
                running_mean_std = checkpoint['running_mean_std']
                checkpoint_size = running_mean_std['running_mean'].shape[0]
                current_size = self.running_mean.shape[0]

                print(f"Checkpoint normalization size: {checkpoint_size}")
                print(f"Current model normalization size: {current_size}")

                if checkpoint_size == current_size:
                    # Perfect match - copy directly
                    self.running_mean.copy_(running_mean_std['running_mean'])
                    self.running_var.copy_(running_mean_std['running_var'])
                    # Load count buffer for stable statistics
                    if 'count' in running_mean_std:
                        self.running_count.copy_(running_mean_std['count'])
                        print(f"✓ Loaded normalization statistics (exact match) with count={self.running_count.item():.0f}")
                    else:
                        print(f"✓ Loaded normalization statistics (exact match, no count in checkpoint)")
                else:
                    # Size mismatch - try to load partial statistics
                    # The checkpoint has different observation size (e.g., 2026 vs 2809)
                    # Load what we can and initialize the rest
                    min_size = min(checkpoint_size, current_size)
                    self.running_mean[:min_size].copy_(running_mean_std['running_mean'][:min_size])
                    self.running_var[:min_size].copy_(running_mean_std['running_var'][:min_size])

                    # Initialize remaining dimensions with default values
                    if current_size > checkpoint_size:
                        self.running_mean[checkpoint_size:].zero_()
                        self.running_var[checkpoint_size:].fill_(1.0)

                    # Load count buffer for stable statistics
                    if 'count' in running_mean_std:
                        self.running_count.copy_(running_mean_std['count'])
                        print(f"✓ Loaded normalization statistics (partial: {min_size}/{current_size} dimensions) with count={self.running_count.item():.0f}")
                    else:
                        print(f"✓ Loaded normalization statistics (partial: {min_size}/{current_size} dimensions, no count in checkpoint)")
                    print(f"  Initialized remaining {current_size - min_size} dimensions with default values")
            else:
                print("⚠ Warning: No running_mean_std found in checkpoint, using default initialization")
                print("  This may cause numerical instability. Consider training from scratch or using a compatible checkpoint.")

        def _load_role_specific_pnn_weights(self, checkpoint):
            """Load role-specific PNN weights from checkpoint"""
            if 'model' not in checkpoint:
                raise ValueError("No model found in checkpoint")

            model_state = checkpoint['model']

            # Extract caregiver PNN weights
            caregiver_state = {}
            for key, value in model_state.items():
                if key.startswith('a2c_network.caregiver_pnn.'):
                    # Remove 'a2c_network.caregiver_pnn.' prefix to get PNN state dict key
                    pnn_key = key.replace('a2c_network.caregiver_pnn.', '')
                    caregiver_state[pnn_key] = value

            # Extract recipient PNN weights
            recipient_state = {}
            for key, value in model_state.items():
                if key.startswith('a2c_network.recipient_pnn.'):
                    # Remove 'a2c_network.recipient_pnn.' prefix to get PNN state dict key
                    pnn_key = key.replace('a2c_network.recipient_pnn.', '')
                    recipient_state[pnn_key] = value

            # Load weights into respective PNNs
            if caregiver_state:
                self.caregiver_pnn.load_state_dict(caregiver_state, strict=False)
                print(f"  Loaded {len(caregiver_state)} caregiver PNN parameters")
            else:
                print(f"  Warning: No caregiver PNN weights found in checkpoint")

            if recipient_state:
                self.recipient_pnn.load_state_dict(recipient_state, strict=False)
                print(f"  Loaded {len(recipient_state)} recipient PNN parameters")
            else:
                print(f"  Warning: No recipient PNN weights found in checkpoint")

        def _load_shared_pnn_weights(self, checkpoint):
            """Load shared PNN weights into both caregiver and recipient PNNs

            Skip the first layer (actors.0.0) due to input dimension mismatch,
            and load all other layers from the checkpoint.
            """
            if 'model' not in checkpoint:
                raise ValueError("No model found in checkpoint")

            model_state = checkpoint['model']

            # Extract shared PNN weights (a2c_network.pnn.*)
            pnn_state = {}
            skipped_keys = []
            first_layer_weights = {}
            first_layer_biases = {}

            for key, value in model_state.items():
                if key.startswith('a2c_network.pnn.'):
                    # Remove 'a2c_network.pnn.' prefix to get PNN state dict key
                    pnn_key = key.replace('a2c_network.pnn.', '')

                    # Handle first layer weight specially due to input dimension mismatch
                    # Weight: [output_dim, input_dim] -> input_dim changed from 2026 to 2809
                    # We'll copy the first 2026 dims and zero-initialize the rest
                    if pnn_key.startswith('actors.0.0.weight'):
                        first_layer_weights[pnn_key] = value
                        print(f"  Found first layer weight {pnn_key}: shape {value.shape}")
                        continue

                    pnn_state[pnn_key] = value

            # Load the same weights into both caregiver and recipient PNNs (strict=False to allow missing keys)
            if pnn_state:
                caregiver_result = self.caregiver_pnn.load_state_dict(pnn_state, strict=False)
                recipient_result = self.recipient_pnn.load_state_dict(pnn_state, strict=False)

                print(f"  Loaded {len(pnn_state)} shared PNN parameters into both caregiver and recipient PNNs")

                # Handle first layer weights: copy common dimensions, zero-initialize new dimensions
                if first_layer_weights:
                    for pnn_key, checkpoint_weight in first_layer_weights.items():
                        # Get the current model's first layer weight parameter
                        caregiver_param = self.caregiver_pnn.state_dict()[pnn_key]
                        recipient_param = self.recipient_pnn.state_dict()[pnn_key]

                        checkpoint_shape = checkpoint_weight.shape  # e.g., [2048, 2026]
                        current_shape = caregiver_param.shape       # e.g., [2048, 2809]

                        print(f"  Partially loading {pnn_key}:")
                        print(f"    Checkpoint shape: {checkpoint_shape}")
                        print(f"    Current shape: {current_shape}")

                        # Copy the common part (first 2026 input dimensions)
                        output_dim = checkpoint_shape[0]
                        common_input_dim = checkpoint_shape[1]

                        # Get current weight tensors (already randomly initialized by PyTorch)
                        new_caregiver_weight = caregiver_param.clone()
                        new_recipient_weight = recipient_param.clone()

                        # Copy checkpoint weights for common dimensions (overwrite random init)
                        new_caregiver_weight[:output_dim, :common_input_dim] = checkpoint_weight
                        new_recipient_weight[:output_dim, :common_input_dim] = checkpoint_weight

                        # For new dimensions (2026:2809), apply small Xavier/He initialization
                        # This ensures gradients can flow and breaks symmetry
                        if current_shape[1] > common_input_dim:
                            new_input_dim = current_shape[1] - common_input_dim
                            # simply zero initialize the new dimensions
                            new_caregiver_weight[:, common_input_dim:].zero_()
                            new_recipient_weight[:, common_input_dim:].zero_()
                            # He initialization: std = sqrt(2 / fan_in)
                            # std = torch.sqrt(torch.tensor(2.0 / current_shape[1]))
                            # new_caregiver_weight[:, common_input_dim:].normal_(0, std * 0.1)  # Scale down by 0.1 for stability
                            # new_recipient_weight[:, common_input_dim:].normal_(0, std * 0.1)

                        # Update the parameters in place
                        caregiver_param.copy_(new_caregiver_weight)
                        recipient_param.copy_(new_recipient_weight)

                        print(f"    ✓ Copied first {common_input_dim} input dimensions from checkpoint")
                        if current_shape[1] > common_input_dim:
                            print(f"    ✓ Zero initialization for remaining {current_shape[1] - common_input_dim} input dimensions")

                # Report any unexpected missing or mismatched keys (excluding the first layer)
                if caregiver_result.missing_keys:
                    non_first_layer_missing = [k for k in caregiver_result.missing_keys if not k.startswith('actors.0.0')]
                    if non_first_layer_missing:
                        print(f"  Warning: Missing keys (unexpected): {non_first_layer_missing}")

                if caregiver_result.unexpected_keys:
                    print(f"  Warning: Unexpected keys: {caregiver_result.unexpected_keys}")
            else:
                print(f"  Warning: No shared PNN weights found in checkpoint")

        def _create_trajectory_cnn(self, input_feature_dim):
            """
            Create the CNN architecture given the input feature dimension.

            Args:
                input_feature_dim: Input feature dimension (2 * single_traj_feature_dim)
            """
            # Design CNN architecture to reduce 20 time steps to 128 dimensions
            # Using multiple conv layers with appropriate stride and kernel sizes

            # Calculate appropriate channel sizes based on input feature dimension
            # Expected input_feature_dim: (rg_pos + rb_rot + body_vel + body_ang_vel) * 2 (self+partner)
            # For 22 bodies: (22*3 + 22*4 + 22*3 + 22*3) * 2 = (66 + 88 + 66 + 66) * 2 = 572 dimensions

            # Use adaptive channel sizing based on input dimension
            if input_feature_dim <= 400:
                first_channels = 256
                second_channels = 256
                third_channels = 256
                fourth_channels = 256
            elif input_feature_dim <= 800:
                first_channels = 512
                second_channels = 512
                third_channels = 512
                fourth_channels = 256
            elif input_feature_dim <= 1600:  # SMPL-X range
                first_channels = 768
                second_channels = 768
                third_channels = 768
                fourth_channels = 384
            else:  # Very high dimensional input
                first_channels = 1024
                second_channels = 1024
                third_channels = 1024
                fourth_channels = 512

            self.trajectory_cnn = nn.Sequential(
                # First conv layer: reduce time dimension and increase channels
                nn.Conv1d(input_feature_dim, first_channels, kernel_size=5, stride=2, padding=2),  # 20 -> 10
                nn.SiLU(),
                nn.BatchNorm1d(first_channels),

                # Second conv layer: further reduce time dimension
                nn.Conv1d(first_channels, second_channels, kernel_size=5, stride=2, padding=2),  # 10 -> 5
                nn.SiLU(),
                nn.BatchNorm1d(second_channels),

                # Third conv layer: more reduction
                nn.Conv1d(second_channels, third_channels, kernel_size=3, stride=2, padding=1),  # 5 -> 3
                nn.SiLU(),
                nn.BatchNorm1d(third_channels),

                # Fourth conv layer: final temporal reduction
                nn.Conv1d(third_channels, fourth_channels, kernel_size=3, stride=2, padding=1),  # 3 -> 2
                nn.SiLU(),
                nn.BatchNorm1d(fourth_channels),

                # Global average pooling to get fixed-size output
                nn.AdaptiveAvgPool1d(1),  # 2 -> 1
                nn.Flatten(),  # [batch_size, fourth_channels]

                # Final fully connected layer to get 128 dimensions
                nn.Linear(fourth_channels, 128),
                nn.SiLU()
            )

            # Initialize weights
            for module in self.trajectory_cnn.modules():
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

            # Move to correct device
            device = next(self.parameters()).device
            self.trajectory_cnn = self.trajectory_cnn.to(device)
            self.trajectory_cnn_initialized = True

        def _encode_future_trajectories(self, combined_trajectories):
            """
            Encode combined trajectory sequences using 1D CNN.

            Args:
                combined_trajectories: [batch_size, future_frames, feature_dim] trajectory data

            Returns:
                torch.Tensor: [batch_size, 128] CNN-encoded features
            """
            # CNN should already be initialized in __init__, but check for safety
            if not self.trajectory_cnn_initialized or self.trajectory_cnn is None:
                print(f"WARNING: trajectory_cnn not initialized, creating now with feature_dim={combined_trajectories.shape[2]}")
                self._create_trajectory_cnn(combined_trajectories.shape[2])

            # Transpose for 1D convolution: [batch_size, feature_dim, future_frames]
            traj_input = combined_trajectories.transpose(1, 2)

            # Apply CNN encoding
            encoded_features = self.trajectory_cnn(traj_input)
            del traj_input
            return encoded_features

        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']
            a_out = self.actor_cnn(obs)  # This is empty
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            
            aux_features = a_out[:, self.self_obs_size + self.task_obs_size:].float()
            
            traj_start_idx = self.partner_obs_size + self.global_root_pos_size
            self_traj_end_idx = traj_start_idx + self.future_trajectory_size//2
            partner_traj_end_idx = self_traj_end_idx + self.future_trajectory_size//2
            
            self_future_traj_flat = aux_features[:, traj_start_idx:self_traj_end_idx]
            partner_future_traj_flat = aux_features[:, self_traj_end_idx:partner_traj_end_idx]
            
            assert a_out.shape[1] == self.self_obs_size + self.task_obs_size + traj_start_idx + self.future_trajectory_size + 1, f"difference: {a_out.shape[1] - (self.self_obs_size + self.task_obs_size + traj_start_idx + self.future_trajectory_size + 1)}"

            # Create dummy aux_intermediates (empty list satisfies the check in PNN)
            # PNN expects aux_intermediates to be not None when zero_fc exists
            aux_intermediates = []
            # Reshape to [batch, 30, 286] format for CNN processing (30 frames, 22 bodies * 13 features = 286)
            batch_size = obs.shape[0]
            features_per_frame = 22 * 13  # 22 non-hand bodies * 13 features each = 286
            self_future_traj = self_future_traj_flat.view(batch_size, 20, features_per_frame)   # [batch, 30, 286]
            partner_future_traj = partner_future_traj_flat.view(batch_size, 20, features_per_frame)  # [batch, 30, 286]

            # Combine self and partner trajectories along channel dimension
            combined_trajectories = torch.cat([self_future_traj, partner_future_traj], dim=2)  # [batch, 30, 572]
            future_trajectory_features = self._encode_future_trajectories(combined_trajectories)
            del combined_trajectories
            del self_future_traj
            del partner_future_traj
            del self_future_traj_flat
            del partner_future_traj_flat

            # Separate weights - determine role and use appropriate PNN
            batch_size = obs.shape[0]

            # Extract role indicator from the last dimension of observation
            # Role indicator: 1.0 for caregiver (even env_id), 0.0 for recipient (odd env_id)
            role_indicator = obs[:, -1]  # Last element contains role indicator
            
            # roleのユニークな値が2つであることを確認
            assert role_indicator.unique().shape[0] == 2, f"role_indicator should have exactly two unique values, got {role_indicator.unique()}"

            # Determine role: caregiver (role_indicator == 1.0) or recipient (role_indicator == 0.0)
            is_caregiver = (role_indicator > 0.5)  # Use threshold for numerical stability
            is_recipient = ~is_caregiver
            
            # Process caregiver and recipient separately
            caregiver_mask = is_caregiver
            recipient_mask = is_recipient

            # Prepare caregiver input (always includes future trajectory features)
            a_out_caregiver = torch.cat([a_out[caregiver_mask][:, :-(1+self.future_trajectory_size)], future_trajectory_features[caregiver_mask]], dim=1)

            # Prepare recipient input (depends on freeze_recipient setting)
            if self.freeze_recipient:
                # Frozen: lightweight input without future trajectory features
                a_out_recipient = a_out[recipient_mask][:, :self.self_obs_size + self.task_obs_size]
            else:
                # Trainable: same input structure as caregiver
                a_out_recipient = torch.cat([a_out[recipient_mask][:, :-(1+self.future_trajectory_size)], future_trajectory_features[recipient_mask]], dim=1)

            # Initialize output with correct shape
            output_shape = list(a_out.shape)
            output_shape[-1] = 153
            a_out_combined = torch.zeros(output_shape, device=obs.device, dtype=a_out.dtype)

            if caregiver_mask.any():
                caregiver_out, _ = self.caregiver_pnn(a_out_caregiver, idx=self.training_prim, aux_intermediates=aux_intermediates)
                a_out_combined[caregiver_mask] = caregiver_out
                del caregiver_out
            if recipient_mask.any():
                if self.freeze_recipient:
                    with torch.no_grad():
                        recipient_out, _ = self.recipient_pnn(a_out_recipient, idx=self.training_prim, aux_intermediates=aux_intermediates)
                        a_out_combined[recipient_mask] = recipient_out
                else:
                    recipient_out, _ = self.recipient_pnn(a_out_recipient, idx=self.training_prim, aux_intermediates=aux_intermediates)
                    a_out_combined[recipient_mask] = recipient_out
                del recipient_out

            del a_out
            a_out = a_out_combined

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = a_out
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return
        
        def _apply_critic_trajectory_encoding(self, obs):
            """Apply CNN trajectory encoding to critic observations (same as actor logic)"""
            # Extract base self observation (self_obs + task_obs)
            base_obs_size = self.self_obs_size + self.task_obs_size
            base_obs = obs[:, :base_obs_size]
            aux_features = obs[:, base_obs_size:]

            # obs is already normalized by agent's _preproc_obs, so use directly
            aux_features_normalized = aux_features

            # Extract and encode future trajectories (same logic as actor forward)
            future_trajectory_features = torch.zeros(obs.shape[0], self.future_trajectory_cnn_output_size,
                                                   device=obs.device, dtype=torch.float32)

            # Extract flattened future trajectories
            traj_start_idx = self.partner_obs_size + self.global_root_pos_size
            self_traj_end_idx = traj_start_idx + self.future_trajectory_size//2
            partner_traj_end_idx = self_traj_end_idx + self.future_trajectory_size//2

            self_future_traj_flat = aux_features_normalized[:, traj_start_idx:self_traj_end_idx]
            partner_future_traj_flat = aux_features_normalized[:, self_traj_end_idx:partner_traj_end_idx]

            # Reshape and combine trajectories
            batch_size = obs.shape[0]
            features_per_frame = 22 * 13
            self_future_traj = self_future_traj_flat.view(batch_size, 20, features_per_frame)
            partner_future_traj = partner_future_traj_flat.view(batch_size, 20, features_per_frame)

            # Combine trajectories
            combined_trajectories = torch.cat([self_future_traj, partner_future_traj], dim=2)

            # Encode with CNN (shared with actor)
            future_trajectory_features = self._encode_future_trajectories(combined_trajectories)
            # Reconstruct observation without raw trajectory data + CNN features
            aux_features_without_future_traj = torch.cat([
                aux_features_normalized[:, :traj_start_idx],  # [self_contact, self_force, base_partner_obs, partner_contact]
                aux_features_normalized[:, partner_traj_end_idx:],  # [role_label]
                future_trajectory_features,  # [128-dim CNN encoded trajectories]
            ], dim=1)
            
            del combined_trajectories
            del self_future_traj
            del partner_future_traj
            del self_future_traj_flat
            del partner_future_traj_flat
            # Return processed observation: [base_obs, processed_aux_features]
            return torch.cat([base_obs, aux_features_without_future_traj], dim=1)
        
        def _rebuild_critic_mlp(self):
            """Rebuild critic MLP to handle critic observation size (asymmetric or symmetric)"""
            # Calculate actual critic observation size after CNN trajectory encoding
            # Symmetric critic: Calculate size after CNN trajectory encoding
            # base_obs + aux_features_without_future_traj + CNN_encoded(128) + role_label
            aux_features_without_future_traj_size = self.partner_obs_size + self.global_root_pos_size 
            cnn_encoded_size = 128

            actual_critic_obs_size = (self.self_obs_size + self.task_obs_size + 
                                    aux_features_without_future_traj_size + cnn_encoded_size + 1) # +1 for role_label

            # Rebuild critic MLP with correct input size
            critic_mlp_args = {
                'input_size': actual_critic_obs_size,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
            }
            
            # Replace the existing critic MLP
            self.critic_mlp = self._build_mlp(**critic_mlp_args)
        
        def eval_critic(self, obs_dict):
            """Override eval_critic to implement trajectory CNN encoding + asymmetric critic"""
            obs = obs_dict['obs']

            # Apply CNN trajectory encoding (same logic as actor forward)
            critic_obs = self._apply_critic_trajectory_encoding(obs)
            # Symmetric critic: obs already normalized by agent
            critic_obs_normalized = critic_obs

            # Ensure consistent dtype (float32)
            critic_obs_normalized = critic_obs_normalized.float()

            # Process through critic network
            c_out = self.critic_cnn(critic_obs_normalized)
            c_out = c_out.contiguous().view(-1, c_out.size(-1))

            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value
        
    
