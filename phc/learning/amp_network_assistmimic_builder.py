
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np
import copy
from phc.learning.pnn import PNN
from rl_games.algos_torch import torch_ext

DISC_LOGIT_INIT_SCALE = 1.0


class AssistMimicPNNBuilder(AMPBuilder):
    """
    Simplified PNN builder for humanx task with single primitive (num_prim=1).
    Input: concat(self_obs, task_obs, partner_self_obs, partner_task_obs)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AssistMimicPNNBuilder.Network(self.params, **kwargs)
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

            # Input shape: self_obs + task_obs + partner_obs (partner_self + partner_task) + role_indicator
            # Partner obs size = self_obs_size + task_obs_size
            global_root_pos_size = 2
            partner_obs_size = self.self_obs_size + self.task_obs_size + global_root_pos_size
            original_pnn_obs_size = self.self_obs_size + self.task_obs_size
            # Total observation size including role indicator (for environment and normalization)
            total_obs_size_with_role = self.self_obs_size + self.task_obs_size + partner_obs_size + 1 + global_root_pos_size
            # PNN input size (without role indicator, as we remove it before passing to PNN)
            pnn_input_size = self.self_obs_size + self.task_obs_size + partner_obs_size + global_root_pos_size

            # Set input shape to include role indicator for normalization statistics
            kwargs['input_shape'] = (total_obs_size_with_role,)

            super().__init__(params, **kwargs)

            # PNN uses input without role indicator
            base_mlp_args = {
                'input_size': self._calc_input_size((original_pnn_obs_size,), self.actor_cnn),
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
            }
            # Caregiver and recipient PNNs receive full observation including partner info
            assist_mlp_args = {
                'input_size': self._calc_input_size((pnn_input_size,), self.actor_cnn),
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

            self.pnn = PNN(
                base_mlp_args,
                output_size=kwargs['actions_num'],
                numCols=self.num_prim,
                has_lateral=self.has_lateral,
                aux_mlp_units=aux_mlp_units
            )
            # Freeze nothing since we only have one primitive and it's trainable
            self.pnn.freeze_pnn(self.training_prim)
            self.caregiver_pnn = PNN(
                assist_mlp_args,
                output_size=kwargs['actions_num'],
                numCols=self.num_prim,
                has_lateral=self.has_lateral,
                aux_mlp_units=aux_mlp_units
            )
            self.recipient_pnn = PNN(
                assist_mlp_args,
                output_size=kwargs['actions_num'],
                numCols=self.num_prim,
                has_lateral=self.has_lateral,
                aux_mlp_units=aux_mlp_units
            )

            # Freeze nothing since we only have one primitive and it's trainable
            self.caregiver_pnn.freeze_pnn(self.training_prim)
            self.recipient_pnn.freeze_pnn(self.training_prim)

            # Create final_fc layers to combine base_pnn and role_pnn outputs
            # Initialize as identity transformation: output = base_pnn_output + 0 * role_pnn_output
            # This ensures initial behavior matches base_pnn
            actions_num = kwargs['actions_num']
            self.caregiver_final_fc = nn.Linear(actions_num * 2, actions_num)
            self.recipient_final_fc = nn.Linear(actions_num * 2, actions_num)

            # Initialize as identity + zero matrix
            with torch.no_grad():
                for fc in [self.caregiver_final_fc, self.recipient_final_fc]:
                    fc.weight.zero_()
                    fc.bias.zero_()
                    # First actions_num columns: identity (copy base_pnn output)
                    fc.weight[:, :actions_num] = torch.eye(actions_num)
                    # Last actions_num columns: zero (ignore role_pnn output initially)

            print(f"✓ Created final_fc layers with identity + zero initialization")

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
            self.pnn.load_base_net(checkpoint_path, actors=1)
            print(f"✓ Successfully loaded checkpoint into base PNN")

            # Copy base_pnn weights to caregiver_pnn and recipient_pnn
            # This creates trainable copies with special handling for 1st layer input dimension mismatch
            self._copy_base_pnn_to_role_pnns()
            print(f"✓ Successfully copied base PNN weights to caregiver and recipient PNNs")

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
                    # Size mismatch - load partial statistics
                    # Checkpoint has different observation size
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

        def _copy_base_pnn_to_role_pnns(self):
            """Copy base_pnn weights to caregiver_pnn and recipient_pnn as trainable copies.

            For the first layer (actors.0.0), special handling is needed due to input dimension mismatch:
            - base_pnn first layer input: original_pnn_obs_size (e.g., 2026)
            - caregiver/recipient_pnn first layer input: pnn_input_size (e.g., 2809)

            Strategy:
            1. Copy all layers except the first layer directly
            2. For the first layer weight: copy common dimensions, zero-initialize new dimensions
            3. For the first layer bias: copy directly (output dimension is the same)
            """
            base_state = self.pnn.state_dict()

            # Separate first layer parameters from others
            first_layer_weights = {}
            first_layer_biases = {}
            other_params = {}

            for key, value in base_state.items():
                if key.startswith('actors.0.0.weight'):
                    first_layer_weights[key] = value
                    print(f"  Found base_pnn first layer weight {key}: shape {value.shape}")
                elif key.startswith('actors.0.0.bias'):
                    first_layer_biases[key] = value
                    print(f"  Found base_pnn first layer bias {key}: shape {value.shape}")
                else:
                    other_params[key] = value

            # Load non-first-layer parameters directly into both caregiver and recipient PNNs
            if other_params:
                caregiver_result = self.caregiver_pnn.load_state_dict(other_params, strict=False)
                recipient_result = self.recipient_pnn.load_state_dict(other_params, strict=False)
                print(f"  Copied {len(other_params)} parameters (excluding first layer) from base_pnn to both role PNNs")

            # Handle first layer weights: copy common dimensions, zero-initialize new dimensions
            if first_layer_weights:
                for key, base_weight in first_layer_weights.items():
                    # Get the current model's first layer weight parameters
                    caregiver_param = self.caregiver_pnn.state_dict()[key]
                    recipient_param = self.recipient_pnn.state_dict()[key]

                    base_shape = base_weight.shape        # e.g., [2048, 2026]
                    current_shape = caregiver_param.shape  # e.g., [2048, 2809]

                    print(f"\n  Processing first layer weight {key}:")
                    print(f"    Base PNN shape: {base_shape}")
                    print(f"    Role PNN shape: {current_shape}")

                    # Copy the common part
                    output_dim = base_shape[0]
                    base_input_dim = base_shape[1]
                    current_input_dim = current_shape[1]

                    # Clone current weights (already randomly initialized by PyTorch)
                    new_caregiver_weight = caregiver_param.clone()
                    new_recipient_weight = recipient_param.clone()

                    # Copy base_pnn weights for common dimensions (overwrite random init)
                    new_caregiver_weight[:output_dim, :base_input_dim] = base_weight
                    new_recipient_weight[:output_dim, :base_input_dim] = base_weight

                    # Zero-initialize new dimensions (partner observation dimensions)
                    if current_input_dim > base_input_dim:
                        new_caregiver_weight[:, base_input_dim:].zero_()
                        new_recipient_weight[:, base_input_dim:].zero_()

                    # Update the parameters in place
                    caregiver_param.copy_(new_caregiver_weight)
                    recipient_param.copy_(new_recipient_weight)

                    print(f"    ✓ Copied first {base_input_dim} input dimensions from base_pnn")
                    if current_input_dim > base_input_dim:
                        print(f"    ✓ Zero initialization for remaining {current_input_dim - base_input_dim} input dimensions")

            # Handle first layer biases: copy directly (output dimension is the same)
            if first_layer_biases:
                for key, base_bias in first_layer_biases.items():
                    # Get the current model's first layer bias parameters
                    caregiver_param = self.caregiver_pnn.state_dict()[key]
                    recipient_param = self.recipient_pnn.state_dict()[key]

                    print(f"\n  Processing first layer bias {key}:")
                    print(f"    Base PNN shape: {base_bias.shape}")
                    print(f"    Role PNN shape: {caregiver_param.shape}")

                    # Bias shape should match output dimension, so direct copy
                    caregiver_param.copy_(base_bias)
                    recipient_param.copy_(base_bias)

                    print(f"    ✓ Copied bias from base_pnn")

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

        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']

            # Extract role indicator from the last dimension
            role_indicator = obs[:, -1]
            assert role_indicator.unique().shape[0] == 2, f"role_indicator should have exactly two unique values, got {role_indicator.unique()}"

            # Determine role: caregiver (role_indicator == 1.0) or recipient (role_indicator == 0.0)
            is_caregiver = (role_indicator > 0)  # Use threshold for numerical stability
            is_recipient = ~is_caregiver
            caregiver_mask = is_caregiver
            recipient_mask = is_recipient

            batch_size = obs.shape[0]
            actions_num = 153  # Output action dimension

            # Create dummy aux_intermediates (empty list satisfies the check in PNN)
            aux_intermediates = []

            # Split observation: [self_obs, task_obs, partner_obs, global_root_pos, role_indicator]
            # We need to extract base_obs (self_obs + task_obs) BEFORE normalization issues
            original_pnn_obs_size = self.self_obs_size + self.task_obs_size

            # Extract base observation (self_obs + task_obs) for base_pnn
            base_obs = obs[:, :original_pnn_obs_size]

            # Pass through actor_cnn to ensure proper processing (even though it's empty)
            base_obs_processed = self.actor_cnn(base_obs)
            base_obs_processed = base_obs_processed.contiguous().view(base_obs_processed.size(0), -1)

            # Step 1: Get base_pnn output (same for all agents, uses only self_obs + task_obs)
            base_pnn_output, _ = self.pnn(base_obs_processed, idx=self.training_prim, aux_intermediates=aux_intermediates)

            # Step 2: Get role-specific pnn output (uses full observation including partner info)
            # Remove role indicator from observation (last dimension)
            obs_no_role = obs[:, :-1]

            # Pass full observation through actor_cnn for proper processing
            full_obs_processed = self.actor_cnn(obs_no_role)
            full_obs_processed = full_obs_processed.contiguous().view(full_obs_processed.size(0), -1)

            role_pnn_output = torch.zeros(batch_size, actions_num, device=obs.device, dtype=full_obs_processed.dtype)

            if caregiver_mask.any():
                caregiver_out, _ = self.caregiver_pnn(full_obs_processed[caregiver_mask], idx=self.training_prim, aux_intermediates=aux_intermediates)
                role_pnn_output[caregiver_mask] = caregiver_out

            if recipient_mask.any():
                recipient_out, _ = self.recipient_pnn(full_obs_processed[recipient_mask], idx=self.training_prim, aux_intermediates=aux_intermediates)
                role_pnn_output[recipient_mask] = recipient_out

            # Step 3: Combine base_pnn and role_pnn outputs via final_fc
            # Concatenate: [base_pnn_output, role_pnn_output]
            combined_features = torch.cat([base_pnn_output, role_pnn_output], dim=1)

            # Apply role-specific final_fc
            a_out = torch.zeros(batch_size, actions_num, device=obs.device, dtype=combined_features.dtype)
            if caregiver_mask.any():
                a_out[caregiver_mask] = self.caregiver_final_fc(combined_features[caregiver_mask])
            if recipient_mask.any():
                a_out[recipient_mask] = self.recipient_final_fc(combined_features[recipient_mask])


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
