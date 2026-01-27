from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from phc.learning.amp_network_builder import AMPBuilder
import torch
import torch.nn as nn
import numpy as np
import copy
from phc.learning.pnn import PNN
from rl_games.algos_torch import torch_ext

DISC_LOGIT_INIT_SCALE = 1.0


class AMPPNNMultiBuilder(AMPBuilder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def build(self, name, **kwargs):
        net = AMPPNNMultiBuilder.Network(self.params, **kwargs)
        return net

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            self.self_obs_size = kwargs['self_obs_size']
            self.task_obs_size = kwargs['task_obs_size']
            self.task_obs_size_detail = kwargs['task_obs_size_detail']
            self.fut_tracks = self.task_obs_size_detail['fut_tracks']
            self.obs_v = self.task_obs_size_detail['obs_v']
            self.num_traj_samples = self.task_obs_size_detail['num_traj_samples']
            self.track_bodies = self.task_obs_size_detail['track_bodies']
            self.num_prim = self.task_obs_size_detail['num_prim']
            self.training_prim = self.task_obs_size_detail['training_prim']
            self.model_base = self.task_obs_size_detail['models_path'][0]
            self.actors_to_load = self.task_obs_size_detail['actors_to_load']
            self.has_lateral = self.task_obs_size_detail['has_lateral']
            
            self.future_size = 1

            # Weight sharing configuration
            self.weight_share = params.get("weight_share", True)  # Default to True for backward compatibility
            print(f"DEBUG: weight_share enabled: {self.weight_share}")

            # Asymmetric critic configuration
            self.asymmetric_critic = params.get("asymmetric_critic", False)  # Default to asymmetric for backward compatibility

            # Calculate contact observation sizes
            self.self_contact_obs_size = self._calculate_self_contact_obs_size()
            self.partner_contact_obs_size = self._calculate_partner_contact_obs_size()
            self.reference_contact_obs_size = self._calculate_reference_contact_obs_size()
            self.action_obs_size = 153
            
            # Freeze recipient configuration
            self.freeze_recipient = params.get("freeze_recipient", False)  # Default to False for backward compatibility
            print(f"DEBUG: freeze_recipient enabled: {self.freeze_recipient}")

            # Residual mode configuration: aux_mlp random init, output = pnn + aux (no final_fc)
            self.residual_mode = params.get("residual_mode", False)
            print(f"DEBUG: residual_mode enabled: {self.residual_mode}")
            
            # Calculate base sizes
            # self.self_obs_size is already the base proprioception (778 dims)
            # Contact info is additional, not part of base proprioception
            self.base_self_obs_size = self.self_obs_size  # 778 dims proprioception
            
            # Calculate partner obs size based on partner_obs_v
            partner_obs_v = getattr(self.task_obs_size_detail, 'partner_obs_v', 2)
            num_bodies = 52
            if partner_obs_v == 3:
                # partner_obs_v=3: separate base partner obs and partner contact
                self.base_partner_obs_size = num_bodies * 3 + num_bodies * 3 + num_bodies * 6 + 6 + 52 * 3 * 2 # Base partner obs size without contact
                # partner_contact_obs_size already calculated above
            else:
                # partner_obs_v=2 is base partner obs without contact
                self.base_partner_obs_size = num_bodies * 3 + num_bodies * 3 + num_bodies * 6 + 6 + 52 * 3 * 2 # Base partner obs size without contact
                self.partner_contact_obs_size = 12  # No contact info
            
            self_force_size = 42

            # Future trajectory size: 30 frames * 22 bodies * 13 features * 2 trajectories
            future_trajectory_size = self.future_size * 22 * 13 * 2

            self.aux_features_size = self.self_contact_obs_size + self_force_size + self.base_partner_obs_size + self.partner_contact_obs_size + future_trajectory_size + 1 + self.action_obs_size  # +1 for role label
            
            # Total observation size for Actor: [self_obs_feat, task_obs, aux_features]
            total_actor_obs_size = self.base_self_obs_size + self.task_obs_size + self.aux_features_size
            
            # Finger joint filtering setup - MUST be initialized first
            self.filter_finger_joints = params.get("filter_finger_joints", False)
            
            # Total observation size for Critic
            if self.asymmetric_critic:
                # Asymmetric critic: Actor obs + partner_task_obs + partner_force_obs
                # Default sizes for partner features (will be set properly when environment reference is available)
                partner_task_obs_size = getattr(self.task_obs_size_detail, 'task_obs_size', 1248)  # Same as task_obs_size
                
                # Apply finger filtering to partner_task_obs if enabled
                if self.filter_finger_joints:
                    # partner_task_obs has same structure as task_obs, so reduce by 720 dims
                    self.filtered_partner_task_obs_size = partner_task_obs_size - 720  # 1248 -> 528
                    print(f"DEBUG: Partner task_obs size reduced from {partner_task_obs_size} to {self.filtered_partner_task_obs_size}")
                else:
                    self.filtered_partner_task_obs_size = partner_task_obs_size
                
                partner_force_obs_size = 42  # Force observation size
                # IMPORTANT: Critic uses UNFILTERED actor obs + filtered partner_task_obs + partner_force_obs
                total_critic_obs_size = total_actor_obs_size + self.filtered_partner_task_obs_size + partner_force_obs_size
                print(f"DEBUG: Using asymmetric critic with partner observations")
            else:
                # Symmetric critic: Same observation as actor
                total_critic_obs_size = total_actor_obs_size
                self.filtered_partner_task_obs_size = 0
                print(f"DEBUG: Using symmetric critic (same as actor observations)")
            
            print(f"DEBUG: base_self_obs_size={self.base_self_obs_size}, task_obs_size={self.task_obs_size}")
            print(f"DEBUG: partner_obs_v={partner_obs_v}, base_partner_obs_size={self.base_partner_obs_size}")
            print(f"DEBUG: aux_features_size={self.aux_features_size} (self_contact={self.self_contact_obs_size}, base_partner={self.base_partner_obs_size}, partner_contact={self.partner_contact_obs_size}, role_label=1) - MultiPulse compatible + role")
            print(f"DEBUG: total_actor_obs_size={total_actor_obs_size}")
            if self.asymmetric_critic:
                partner_force_obs_size = 42
                print(f"DEBUG: filtered_partner_task_obs_size={self.filtered_partner_task_obs_size}, partner_force_obs_size={partner_force_obs_size}")
                print(f"DEBUG: total_critic_obs_size={total_critic_obs_size} (actor_obs + partner_task_obs + partner_force_obs)")
                print(f"DEBUG: total_critic_obs_size calculation: {total_actor_obs_size} + {self.filtered_partner_task_obs_size} + {partner_force_obs_size} = {total_critic_obs_size}")
            else:
                print(f"DEBUG: total_critic_obs_size={total_critic_obs_size} (same as actor)")
                print(f"DEBUG: asymmetric_critic=False, critic uses same observations as actor")
            
            if self.filter_finger_joints:
                self._setup_finger_filtering_indices()
                self.pnn_output_size = 1306  # 2026 - 720
                print(f"DEBUG: Finger filtering enabled, PNN output size reduced to {self.pnn_output_size}")
            else:
                self.pnn_output_size = 2026
                print(f"DEBUG: Finger filtering disabled, PNN output size: {self.pnn_output_size}")
            
            # Future trajectory CNN encoding size
            self.future_trajectory_cnn_output_size = 0 #128

            # Calculate aux_mlp input size
            # aux_features_without_future_traj + filtered_pnn_output + cnn_encoded_future_traj
            future_traj_size = self.future_size * 22 * 13 * 2  # 2 trajectories (self+partner) * 30 frames * 22 bodies * 13 features
            aux_features_without_future_traj_size = self.aux_features_size - future_traj_size
            self.aux_mlp_input_size = aux_features_without_future_traj_size + self.pnn_output_size + self.future_trajectory_cnn_output_size
            print(f"DEBUG: aux_mlp input size: aux_features_without_future_traj({aux_features_without_future_traj_size}) + pnn_output({self.pnn_output_size}) + cnn_encoded_future_traj({self.future_trajectory_cnn_output_size}) = {self.aux_mlp_input_size}")
            
            # Actor uses original observation size
            kwargs['input_shape'] = (total_actor_obs_size,)
            
            # Store critic observation size for later use
            self.critic_obs_size = total_critic_obs_size

            super().__init__(params, **kwargs)

            # Initialize caregiver and recipient sigma values
            self._init_role_based_sigma(params)
            
            # Override critic MLP to handle critic observations (CNN trajectory encoding changes observation size)
            self._rebuild_critic_mlp()
            
            # Build the main PNN for self + task observations (im_pnn_big)
            # Use _calc_input_size like the original PNN builder to get the correct input size after CNN processing
            # Try to get experiment name from various sources
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
            
            print(f"DEBUG: Detected exp_name: {exp_name}")
            
                
            calculated_input_size = self.base_self_obs_size + self.task_obs_size

            actor_mlp_args = {
                'input_size': calculated_input_size,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
            }

            del self.actor_mlp
            self.discrete = params.get("discrete", False)

            # Create the main PNN (im_pnn_big) - output should be feature size, not action size
            pnn_output_size = self.units[-1]  # 512 - feature size for combining with partner embedding

            # Get aux_mlp units for zero_fc layer initialization
            aux_mlp_config = params.get('aux_mlp', {
                'units': [256, 256, 256, 128],
                'activation': 'silu',
                'output_size': kwargs['actions_num'],
                'zero_init': False,
                'debug_interval': 10
            })
            aux_mlp_units = aux_mlp_config.get('units', [256, 256, 256, 128])
            self.aux_mlp_units = aux_mlp_units
            self.pnn = PNN(actor_mlp_args, output_size=kwargs['actions_num'], numCols=self.num_prim, has_lateral=self.has_lateral, aux_mlp_units=aux_mlp_units)

            # If weight_share is False, create separate networks for caregiver and recipient
            if not self.weight_share:
                print("Creating separate networks for caregiver and recipient (weight_share=False)")
                # Create additional MLPs for separate processing
                self.caregiver_aux_mlp = None
                self.recipient_aux_mlp = None
                self.caregiver_final_fc = None
                self.recipient_final_fc = None
            else:
                print("Using shared weights for caregiver and recipient (weight_share=True)")
                
            # Determine checkpoint path
            use_exp_checkpoint = False
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

            # ===== Check if pretrained PHC loading should be skipped (Ablation mode) =====
            load_pretrained_phc = params.get('load_pretrained_phc', True)
            print(f"DEBUG: load_pretrained_phc = {load_pretrained_phc}")

            # Create aux_features embedding MLP (using PNN structure to match main PNN)
            # Following assistmimic_builder approach: use PNN instead of custom MLP

            print(f"DEBUG: aux_mlp_config: {aux_mlp_config}")
            # Set debug interval
            self.aux_mlp_debug_interval = aux_mlp_config.get('debug_interval', 1000)

            # Build aux_mlp as PNN (same structure as main PNN)
            aux_mlp_args = {
                'input_size': self._calc_input_size((self.aux_mlp_input_size,), self.actor_cnn),
                'units': aux_mlp_config['units'],
                'activation': aux_mlp_config['activation'],
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
            }

            if self.weight_share:
                # Shared weights - create single PNN for aux_mlp
                self.aux_mlp = PNN(
                    aux_mlp_args,
                    output_size=kwargs['actions_num'],
                    numCols=self.num_prim,
                    has_lateral=self.has_lateral,
                    aux_mlp_units=aux_mlp_units
                )
                self.aux_mlp.freeze_pnn(self.training_prim)

                if self.residual_mode:
                    # Residual mode: no final_fc, output = pnn + aux
                    self.final_fc = None
                    # Zero-initialize aux_mlp output layer so initial output = pnn_output + 0
                    with torch.no_grad():
                        output_layer = self.aux_mlp.actors[0][-1]  # Last layer of first actor
                        nn.init.zeros_(output_layer.weight)
                        nn.init.zeros_(output_layer.bias)
                    print("✓ Residual mode: final_fc is None, aux_mlp output layer zero-initialized")
                else:
                    # Final fully connected layer (identity initialized)
                    self.final_fc = nn.Linear(kwargs['actions_num'] * 2, kwargs['actions_num'])
                    # Initialize as identity transformation
                    with torch.no_grad():
                        self.final_fc.weight.zero_(); self.final_fc.bias.zero_()
                        self.final_fc.weight[:, :kwargs['actions_num']] = torch.eye(kwargs['actions_num'])
            else:
                # Separate weights - create separate PNNs for caregiver and recipient
                self.caregiver_aux_mlp = PNN(
                    aux_mlp_args,
                    output_size=kwargs['actions_num'],
                    numCols=self.num_prim,
                    has_lateral=self.has_lateral,
                    aux_mlp_units=aux_mlp_units
                )
                self.recipient_aux_mlp = PNN(
                    aux_mlp_args,
                    output_size=kwargs['actions_num'],
                    numCols=self.num_prim,
                    has_lateral=self.has_lateral,
                    aux_mlp_units=aux_mlp_units
                )
                
                if self.freeze_recipient:
                    for param in self.recipient_aux_mlp.parameters():
                        param.requires_grad = False
                    self.recipient_aux_mlp.eval()  # Set to eval mode
                    print("✓ Recipient PNN frozen: requires_grad=False, eval mode enabled")
                else:
                    print("✓ Recipient PNN trainable: requires_grad=True")

                # Freeze both PNNs
                self.caregiver_aux_mlp.freeze_pnn(self.training_prim)
                self.recipient_aux_mlp.freeze_pnn(self.training_prim)

                if self.residual_mode:
                    # Residual mode: no final_fc, output = pnn + aux
                    self.caregiver_final_fc = None
                    self.recipient_final_fc = None
                    # Zero-initialize aux_mlp output layers so initial output = pnn_output + 0
                    with torch.no_grad():
                        for aux_mlp in [self.caregiver_aux_mlp, self.recipient_aux_mlp]:
                            output_layer = aux_mlp.actors[0][-1]  # Last layer of first actor
                            nn.init.zeros_(output_layer.weight)
                            nn.init.zeros_(output_layer.bias)
                    print("✓ Residual mode: final_fc layers are None, aux_mlp output layers zero-initialized")
                else:
                    # Final fully connected layers (identity initialized)
                    self.caregiver_final_fc = nn.Linear(kwargs['actions_num'] * 2, kwargs['actions_num'])
                    self.recipient_final_fc = nn.Linear(kwargs['actions_num'] * 2, kwargs['actions_num'])

                    # Initialize as identity transformation
                    with torch.no_grad():
                        for fc in [self.caregiver_final_fc, self.recipient_final_fc]:
                            # fc.weight.zero_(); fc.bias.zero_()
                            # fc.weight[:, :kwargs['actions_num']] = torch.eye(kwargs['actions_num'])
                            fc.weight[:,:kwargs['actions_num']] = torch.eye(kwargs['actions_num']) * 0.5
                            fc.weight[:,kwargs['actions_num']:] = torch.eye(kwargs['actions_num']) * 0.5
                            fc.bias.zero_()

                # Keep references to shared aux_mlp and final_fc for compatibility
                self.aux_mlp = self.caregiver_aux_mlp  # Default to caregiver for compatibility
                self.final_fc = self.caregiver_final_fc

            # Initialize future trajectory CNN
            # Pre-initialize CNN with expected input feature dimension to allow checkpoint loading
            # Expected input: (self_traj + partner_traj) = 2 * (20 frames * 22 bodies * 13 features) = 2 * 5720 features per frame
            # After reshape: [batch, 20, 22*13*2] = [batch, 20, 572]
            expected_trajectory_feature_dim = 22 * 13 * 2  # 572 features per frame (self + partner)
            self._create_trajectory_cnn(expected_trajectory_feature_dim)
            self.trajectory_cnn_initialized = True

            self.running_mean = kwargs['mean_std'].running_mean
            self.running_var = kwargs['mean_std'].running_var
            self.running_count = kwargs['mean_std'].count

            # Initialize separate normalization statistics for critic (only if asymmetric)
            if self.asymmetric_critic:
                self.critic_running_mean = torch.zeros(self.critic_obs_size, dtype=torch.float32, device=self.running_mean.device)
                self.critic_running_var = torch.ones(self.critic_obs_size, dtype=torch.float32, device=self.running_var.device)
            else:
                # For symmetric critic, reuse actor normalization statistics
                self.critic_running_mean = self.running_mean
                self.critic_running_var = self.running_var

            # ===== Check if pretrained PHC loading should be skipped (Ablation mode) =====
            if not load_pretrained_phc:
                print("=" * 80)
                print("ABLATION MODE: Skipping pretrained PHC loading")
                print("  - PNN networks will use random initialization (PyTorch default)")
                print("  - Normalization statistics will start from scratch")
                print("  - All weights will be trained from random initialization")
                print("=" * 80)
                # Reset normalization statistics to initial state
                self.running_mean.zero_()
                self.running_var.fill_(1.0)
                self.running_count.fill_(1.0)  # CRITICAL: Reset count to 1.0 for proper statistics updates
                print(f"✓ Normalization statistics reset: mean=0, var=1, count=1")

                # Reset critic normalization statistics as well
                if self.asymmetric_critic:
                    self.critic_running_mean.zero_()
                    self.critic_running_var.fill_(1.0)
                    print(f"✓ Critic normalization statistics reset: mean=0, var=1")

                # Initialize aux_features observation count
                self.aux_features_count = 0
                print(f"DEBUG: Initialized aux_features tracking. Count: {self.aux_features_count}")
                return  # Skip all checkpoint loading below
            # ===== End of ablation mode check =====

            # Load checkpoint with proper device mapping to avoid CUDA device mismatch
            if torch.cuda.is_available():
                current_device = f'cuda:{torch.cuda.current_device()}'
                checkpoint = torch.load(checkpoint_path, map_location=current_device)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load pretrained PHC weights
            # For inference, always load at least 1 actor (training_prim + 1)
            actors_to_load = 1 #max(1, self.actors_to_load) if self.actors_to_load > 0 else self.training_prim + 1
            # Load PNN from checkpoint with proper logging
            print(f"Loading {actors_to_load} PNN actors from {checkpoint_path}")
            try:
                self.pnn.load_base_net(checkpoint_path, actors_to_load)
                print(f"✓ Successfully loaded PNN weights from checkpoint")
            except Exception as e:
                print(f"✗ Failed to load PNN weights: {e}")
                raise e
            self.pnn.freeze_pnn(self.training_prim)
            print(f"✓ PNN frozen for training_prim={self.training_prim}")

            # CRITICAL: Load normalization statistics from checkpoint after initialization
            if 'running_mean_std' in checkpoint:
                running_mean_std = checkpoint['running_mean_std']
                checkpoint_size = running_mean_std['running_mean'].shape[0]
                current_size = self.running_mean.shape[0]
                expected_main_size = calculated_input_size  # Use calculated input size instead of fixed size
                
                print(f"DEBUG: Loading normalization stats - current_size={current_size}, checkpoint_size={checkpoint_size}, expected_main_size={expected_main_size}")
                
                if use_exp_checkpoint and checkpoint_size == current_size:
                    # When loading from experiment checkpoint and sizes match completely, copy everything
                    self.running_mean.copy_(running_mean_std['running_mean'])
                    self.running_var.copy_(running_mean_std['running_var'])
                    self.running_count.copy_(running_mean_std['count'])
                    print(f"✓ Successfully loaded complete normalization statistics for {current_size} observation dimensions from experiment checkpoint")
                elif checkpoint_size == expected_main_size:
                    # When loading from base checkpoint, only copy main observations part
                    self.running_mean[:expected_main_size].copy_(running_mean_std['running_mean'])
                    self.running_var[:expected_main_size].copy_(running_mean_std['running_var'])
                    self.running_count.copy_(running_mean_std['count'])
                    print(f"✓ Successfully loaded main normalization statistics for {expected_main_size} observation dimensions")
                    
                    # Initialize aux_features normalization statistics following PHC standard
                    aux_start_idx = expected_main_size  # After self_obs_feat + task_obs
                    aux_end_idx = aux_start_idx + self.aux_features_size
                    # Initialize aux_features normalization with PHC standard (zeros for mean, ones for variance)
                    self.running_mean[aux_start_idx:aux_end_idx].zero_()  # Start with zero mean
                    self.running_var[aux_start_idx:aux_end_idx].fill_(1.0)  # Start with unit variance
                    print(f"Initialized aux_features normalization statistics for indices {aux_start_idx}:{aux_end_idx}")
                    
                    # Initialize critic normalization statistics (only if asymmetric)
                    if self.asymmetric_critic:
                        # Copy actor normalization for shared observation parts
                        actor_obs_size = total_actor_obs_size
                        self.critic_running_mean[:actor_obs_size].copy_(self.running_mean[:actor_obs_size])
                        self.critic_running_var[:actor_obs_size].copy_(self.running_var[:actor_obs_size])
                        # Initialize partner features with zeros and ones
                        self.critic_running_mean[actor_obs_size:].zero_()
                        self.critic_running_var[actor_obs_size:].fill_(1.0)
                        print(f"Initialized asymmetric critic normalization statistics - actor_part: {actor_obs_size}, partner_part: {self.critic_obs_size - actor_obs_size}")
                    else:
                        print(f"Symmetric critic using actor normalization statistics")
                else:
                    print(f"✗ Warning: Normalization statistics size mismatch! Current model: {current_size}, checkpoint: {checkpoint_size}, expected main: {expected_main_size}")
                    print(f"✗ This will cause incorrect normalization and poor performance!")
                    
                    # Fallback: Initialize aux_features normalization statistics
                    aux_start_idx = expected_main_size
                    aux_end_idx = aux_start_idx + self.aux_features_size
                    self.running_mean[aux_start_idx:aux_end_idx].zero_()
                    self.running_var[aux_start_idx:aux_end_idx].fill_(1.0)
                    print(f"Fallback: Initialized aux_features normalization statistics for indices {aux_start_idx}:{aux_end_idx}")
                    
                    # Fallback: Initialize critic normalization statistics (only if asymmetric)
                    if self.asymmetric_critic:
                        actor_obs_size = total_actor_obs_size
                        self.critic_running_mean[:actor_obs_size].copy_(self.running_mean[:actor_obs_size])
                        self.critic_running_var[:actor_obs_size].copy_(self.running_var[:actor_obs_size])
                        self.critic_running_mean[actor_obs_size:].zero_()
                        self.critic_running_var[actor_obs_size:].fill_(1.0)
                        print(f"Fallback: Initialized asymmetric critic normalization statistics")
                    else:
                        print(f"Fallback: Symmetric critic using actor normalization statistics")
            else:
                print("✗ Warning: No running_mean_std found in checkpoint!")
                # Initialize aux_features normalization statistics following PHC standard
                aux_start_idx = calculated_input_size  # After self_obs_feat + task_obs
                aux_end_idx = aux_start_idx + self.aux_features_size
                self.running_mean[aux_start_idx:aux_end_idx].zero_()  # Start with zero mean
                self.running_var[aux_start_idx:aux_end_idx].fill_(1.0)  # Start with unit variance
                self.running_count[aux_start_idx:aux_end_idx].zero_()
                print(f"No checkpoint: Initialized aux_features normalization statistics for indices {aux_start_idx}:{aux_end_idx}")
                
                # Initialize critic normalization statistics without checkpoint (only if asymmetric)
                if self.asymmetric_critic:
                    actor_obs_size = total_actor_obs_size
                    self.critic_running_mean[:actor_obs_size].copy_(self.running_mean[:actor_obs_size])
                    self.critic_running_var[:actor_obs_size].copy_(self.running_var[:actor_obs_size])
                    self.critic_running_mean[actor_obs_size:].zero_()
                    self.critic_running_var[actor_obs_size:].fill_(1.0)
                    print(f"No checkpoint: Initialized asymmetric critic normalization statistics")
                else:
                    print(f"No checkpoint: Symmetric critic using actor normalization statistics")
            
            # Initialize aux_features observation count for proper statistics tracking
            self.aux_features_count = 0
            print(f"DEBUG: Initialized aux_features tracking. Count: {self.aux_features_count}")
            
            # Load additional modules from experiment checkpoint if available
            if use_exp_checkpoint and 'model' in checkpoint:
                self._load_additional_modules_from_checkpoint(checkpoint)
            
            # Clean up checkpoint from memory to prevent memory leak
            del checkpoint
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ Checkpoint memory cleaned up")
            
            # Print initial aux_mlp weights for verification
            print("\n=== INITIAL AUX_MLP WEIGHTS ===")
            self._debug_step_count = 0
            # self._print_aux_mlp_weights_debug()

        def _init_role_based_sigma(self, params):
            """Initialize separate sigma values for caregiver and recipient"""
            if self.is_continuous and not self.space_config['learn_sigma']:
                # Get role-based sigma configuration
                role_sigma_config = self.space_config.get('role_sigma', {})
                actions_num = self.sigma.shape[0]

                # Create separate sigma parameters for caregiver and recipient
                self.caregiver_sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                self.recipient_sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)

                # Get sigma values from config
                caregiver_val = role_sigma_config.get('caregiver_val', -2.9)
                recipient_val = role_sigma_config.get('recipient_val', -2.9)

                # Initialize both sigma parameters
                nn.init.constant_(self.caregiver_sigma, caregiver_val)
                nn.init.constant_(self.recipient_sigma, recipient_val)

                print(f"✓ Initialized role-based sigma: caregiver={caregiver_val}, recipient={recipient_val}")
            else:
                print("✓ Role-based sigma not applicable (learnable sigma enabled)")

        def _get_role_based_sigma(self, obs_dict):
            """Get appropriate sigma based on environment roles"""
            if not (self.is_continuous and not self.space_config['learn_sigma']):
                return self.sigma

            # Get batch size from observations
            obs = obs_dict['obs']
            batch_size = obs.shape[0]

            # Extract role from observation (last element of aux_features)
            aux_start_idx = self.base_self_obs_size + self.task_obs_size
            role_idx = aux_start_idx + self.aux_features_size - 1  # Last element is role label
            role_labels = obs[:, role_idx]  # Normalized: -1 for recipient (originally 0), 1 for caregiver (originally 1)

            is_recipient = (role_labels <= 0)  # Handle both original [0,1] and normalized [-1,1]: <=0 for recipient, >0 for caregiver

            # Create sigma tensor for the batch
            batch_sigma = torch.zeros(batch_size, self.caregiver_sigma.shape[0], device=obs.device)

            # Assign sigma values based on roles
            batch_sigma[~is_recipient] = self.caregiver_sigma.unsqueeze(0).expand((~is_recipient).sum(), -1)
            batch_sigma[is_recipient] = self.recipient_sigma.unsqueeze(0).expand(is_recipient.sum(), -1)

            return batch_sigma


        def _setup_finger_filtering_indices(self):
            """Setup indices for filtering finger joints from PNN output"""
            # Finger joint indices in SMPLH_MUJOCO_NAMES (30 finger joints)
            finger_indices = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # L_Index~L_Thumb  
                            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # R_Index~R_Thumb
            
            # task_obs内での削除インデックス計算
            task_obs_remove_indices = self._calculate_task_obs_finger_indices(finger_indices)
            
            # PNN出力での削除インデックス = 778(self_obs) + task_obs内削除インデックス
            self.pnn_output_remove_indices = [778 + idx for idx in task_obs_remove_indices]
            
            # 保持するインデックス
            all_indices = set(range(2026))
            remove_indices = set(self.pnn_output_remove_indices) 
            self.pnn_output_keep_indices = sorted(list(all_indices - remove_indices))
            
            print(f"DEBUG: Finger filtering setup - removing {len(self.pnn_output_remove_indices)} dimensions from PNN output")

        def _calculate_task_obs_finger_indices(self, finger_joint_indices):
            """Calculate task_obs indices to remove for finger joints"""
            remove_indices = []
            
            # obs_v=6構成: [pos_diff(3), rot_diff(6), vel_diff(3), ang_vel_diff(3), ref_pos(3), ref_rot(6)] × 52joints
            components = [
                ('pos_diff', 3),      # 0-155
                ('rot_diff', 6),      # 156-467
                ('vel_diff', 3),      # 468-623  
                ('ang_vel_diff', 3),  # 624-779
                ('ref_pos', 3),       # 780-935
                ('ref_rot', 6)        # 936-1247
            ]
            
            start_idx = 0
            for comp_name, dims_per_joint in components:
                for finger_joint_idx in finger_joint_indices:
                    joint_start = start_idx + finger_joint_idx * dims_per_joint
                    remove_indices.extend(range(joint_start, joint_start + dims_per_joint))
                start_idx += 52 * dims_per_joint
            
            return sorted(remove_indices)

        def _get_task_obs_keep_indices(self):
            """Get keep indices for task_obs only (not full PNN input)"""
            if not hasattr(self, '_task_obs_keep_indices'):
                # Calculate task_obs remove indices  
                finger_indices = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  # L_Index~L_Thumb  
                                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # R_Index~R_Thumb
                task_obs_remove_indices = self._calculate_task_obs_finger_indices(finger_indices)
                
                # task_obs keep indices (1248 total -> 528 after filtering)
                all_task_indices = set(range(1248))
                remove_task_indices = set(task_obs_remove_indices)
                self._task_obs_keep_indices = sorted(list(all_task_indices - remove_task_indices))
            
            return self._task_obs_keep_indices

        def _build_aux_mlp(self, input_size, units, activation, output_size, zero_init=True):
            """Build the aux_features embedding MLP with intermediate output access"""
            class AuxMLPWithIntermediate(nn.Module):
                def __init__(self, input_size, units, activation, output_size, zero_init, normalization):
                    super().__init__()
                    self.layers = nn.ModuleList()
                    self.activation_layers = nn.ModuleList()
                    self.norm_layers = nn.ModuleList()

                    # Input layer
                    self.layers.append(nn.Linear(input_size, units[0]))

                    # Activation
                    if activation == 'silu':
                        self.activation_layers.append(nn.SiLU())
                    elif activation == 'relu':
                        self.activation_layers.append(nn.ReLU())
                    elif activation == 'tanh':
                        self.activation_layers.append(nn.Tanh())

                    # Normalization
                    if normalization == 'layer_norm':
                        self.norm_layers.append(nn.LayerNorm(units[0]))
                    elif normalization == 'batch_norm':
                        self.norm_layers.append(nn.BatchNorm1d(units[0]))
                    else:
                        self.norm_layers.append(nn.Identity())

                    # Hidden layers
                    for i in range(len(units) - 1):
                        self.layers.append(nn.Linear(units[i], units[i + 1]))

                        # Activation
                        if activation == 'silu':
                            self.activation_layers.append(nn.SiLU())
                        elif activation == 'relu':
                            self.activation_layers.append(nn.ReLU())
                        elif activation == 'tanh':
                            self.activation_layers.append(nn.Tanh())

                        # Add normalization for hidden layers (except the last one)
                        if i < len(units) - 2:  # Not the last hidden layer
                            if normalization == 'layer_norm':
                                self.norm_layers.append(nn.LayerNorm(units[i + 1]))
                            elif normalization == 'batch_norm':
                                self.norm_layers.append(nn.BatchNorm1d(units[i + 1]))
                            else:
                                self.norm_layers.append(nn.Identity())
                        else:
                            self.norm_layers.append(nn.Identity())

                    # Output layer - ONLY this layer gets zero initialization
                    self.output_layer = nn.Linear(units[-1], output_size)
                    if zero_init:
                        nn.init.zeros_(self.output_layer.weight)
                        nn.init.zeros_(self.output_layer.bias)

                def forward(self, x, return_intermediates=False):
                    if return_intermediates==False:
                        error_msg = "Currently, return_intermediates must be True when return_intermediates is False"
                        raise ValueError(error_msg)
                    intermediates = []

                    # Process through layers
                    for i, (layer, activation, norm) in enumerate(zip(self.layers, self.activation_layers, self.norm_layers)):
                        x = layer(x)
                        x = activation(x)
                        x = norm(x)
                        intermediates.append(x)

                    # Output layer
                    output = self.output_layer(x)

                    return output, intermediates

            # Normalization to match PNN style
            norm_name = getattr(self, 'normalization', None) or 'batch_norm'

            aux_mlp = AuxMLPWithIntermediate(input_size, units, activation, output_size, zero_init, norm_name)

            if zero_init:
                print(f"Applied zero initialization to FINAL layer only (ControlNet-style)")
            else:
                print(f"Using default initialization for all layers")

            return aux_mlp
        

        def _initialize_trajectory_cnn(self):
            """
            Initialize the 1D CNN architecture for trajectory encoding.
            This CNN will process 30-frame future trajectories of self and partner (hand joints removed)
            and output 128-dimensional embeddings for aux_mlp input.
            """
            # Calculate input feature dimension based on SMPL-X body structure
            # num_bodies = 52 total - 30 hand joints = 22 bodies
            # features per body: rigid_body_pos (3) + rigid_body_rot (4) + rigid_body_vel (3) + rigid_body_ang_vel (3) = 13
            # 2 humanoids (self + partner)
            # input_feature_dim = 22 * 13 * 2 = 572
            num_non_hand_bodies = 22
            features_per_body = 13  # pos(3) + rot(4) + vel(3) + ang_vel(3)
            num_humanoids = 2  # self + partner
            input_feature_dim = num_non_hand_bodies * features_per_body * num_humanoids

            print(f"Initializing trajectory CNN with input_feature_dim={input_feature_dim} (22 bodies × 13 features × 2 humanoids)")
            self._create_trajectory_cnn(input_feature_dim)

        def _create_trajectory_cnn(self, input_feature_dim):
            """
            Create the CNN architecture given the input feature dimension.

            Args:
                input_feature_dim: Input feature dimension (2 * single_traj_feature_dim)
            """
            # Design CNN architecture to reduce 30 time steps to 128 dimensions
            # Using multiple conv layers with appropriate stride and kernel sizes

            # Calculate appropriate channel sizes based on input feature dimension
            # Expected input_feature_dim: (rg_pos + rb_rot + body_vel + body_ang_vel) * 2 (self+partner)
            # SMPL-X: (52*3 + 52*4 + 52*3 + 52*3) * 2 = (156 + 208 + 156 + 156) * 2 = 1352 dimensions

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
            elif input_feature_dim <= 1600:  # SMPL-X range (~1352 dims)
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
                nn.Conv1d(input_feature_dim, first_channels, kernel_size=5, stride=2, padding=2),  # 30 -> 15
                nn.SiLU(),
                nn.BatchNorm1d(first_channels),

                # Second conv layer: further reduce time dimension
                nn.Conv1d(first_channels, second_channels, kernel_size=5, stride=2, padding=2),  # 15 -> 8
                nn.SiLU(),
                nn.BatchNorm1d(second_channels),

                # Third conv layer: more reduction
                nn.Conv1d(second_channels, third_channels, kernel_size=3, stride=2, padding=1),  # 8 -> 4
                nn.SiLU(),
                nn.BatchNorm1d(third_channels),

                # Fourth conv layer: final temporal reduction
                nn.Conv1d(third_channels, fourth_channels, kernel_size=3, stride=2, padding=1),  # 4 -> 2
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

            # print(f"Trajectory CNN initialized with input feature dim: {input_feature_dim}")
            # print(f"CNN architecture: {self.trajectory_cnn}")

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
            # encoded_features = self.trajectory_cnn(traj_input)
            encoded_features = torch.zeros(combined_trajectories.shape[0], 128, device=combined_trajectories.device, dtype=torch.float32)
            del traj_input

            return encoded_features

        def _load_additional_modules_from_checkpoint(self, checkpoint):
            """Load aux_mlp, final_fc, and other modules from experiment checkpoint"""
            model_state = checkpoint['model']
            current_state = self.state_dict()
            
            def _extract_local_key(checkpoint_key):
                """Extract local key handling both DataParallel (module.) and normal prefixes"""
                # Handle DataParallel prefix with a2c_network (a2c_network.module.)
                if checkpoint_key.startswith('a2c_network.module.'):
                    return checkpoint_key[len('a2c_network.module.'):]
                # Handle DataParallel prefix (module.a2c_network.)
                elif checkpoint_key.startswith('module.a2c_network.'):
                    return checkpoint_key[len('module.a2c_network.'):]
                # Handle normal prefix (a2c_network.)
                elif checkpoint_key.startswith('a2c_network.'):
                    return checkpoint_key[len('a2c_network.'):]
                # Handle module prefix only
                elif checkpoint_key.startswith('module.'):
                    return checkpoint_key[len('module.'):]
                # Handle keys without prefix
                else:
                    return checkpoint_key
            
            # Load caregiver_aux_mlp weights if available in checkpoint
            caregiver_aux_mlp_keys = [k for k in model_state.keys() if 'caregiver_aux_mlp' in k]
            if caregiver_aux_mlp_keys:
                print(f"Loading caregiver_aux_mlp weights from checkpoint...")
                for key in caregiver_aux_mlp_keys:
                    local_key = _extract_local_key(key)
                    if local_key in current_state:
                        if current_state[local_key].shape == model_state[key].shape:
                            current_state[local_key].copy_(model_state[key])
                            print(f"  ✓ Loaded {local_key}")
                        else:
                            print(f"  ✗ Shape mismatch for {local_key}: {current_state[local_key].shape} vs {model_state[key].shape}")
                    else:
                        print(f"  ✗ Key not found in current model: {local_key}")
            
            # Load recipient_aux_mlp weights if available in checkpoint
            recipient_aux_mlp_keys = [k for k in model_state.keys() if 'recipient_aux_mlp' in k]
            if recipient_aux_mlp_keys:
                print(f"Loading recipient_aux_mlp weights from checkpoint...")
                for key in recipient_aux_mlp_keys:
                    local_key = _extract_local_key(key)
                    if local_key in current_state:
                        if current_state[local_key].shape == model_state[key].shape:
                            current_state[local_key].copy_(model_state[key])
                            print(f"  ✓ Loaded {local_key}")
                        else:
                            print(f"  ✗ Shape mismatch for {local_key}: {current_state[local_key].shape} vs {model_state[key].shape}")
                    else:
                        print(f"  ✗ Key not found in current model: {local_key}")
            
            # Load aux_mlp weights if available (and copy to role-based format if needed)
            aux_mlp_keys = [k for k in model_state.keys() if 'aux_mlp' in k and 'caregiver_aux_mlp' not in k and 'recipient_aux_mlp' not in k]
            if aux_mlp_keys:
                print(f"Loading aux_mlp weights from checkpoint...")
                for key in aux_mlp_keys:
                    local_key = _extract_local_key(key)

                    # First try to load to the original key
                    if local_key in current_state:
                        if current_state[local_key].shape == model_state[key].shape:
                            current_state[local_key].copy_(model_state[key])
                            print(f"  ✓ Loaded {local_key}")
                        else:
                            print(f"  ✗ Shape mismatch for {local_key}: {current_state[local_key].shape} vs {model_state[key].shape}")
                    else:
                        print(f"  ✗ Key not found in current model: {local_key}")

                    # If weight_share=False, also copy to caregiver and recipient versions (only if they don't exist in checkpoint)
                    if not self.weight_share and not caregiver_aux_mlp_keys and not recipient_aux_mlp_keys:
                        caregiver_key = local_key.replace('aux_mlp', 'caregiver_aux_mlp')
                        recipient_key = local_key.replace('aux_mlp', 'recipient_aux_mlp')

                        # Copy to caregiver_aux_mlp
                        if caregiver_key in current_state:
                            if current_state[caregiver_key].shape == model_state[key].shape:
                                current_state[caregiver_key].copy_(model_state[key])
                                print(f"  ✓ Copied {local_key} → {caregiver_key}")
                            else:
                                print(f"  ✗ Shape mismatch for {caregiver_key}: {current_state[caregiver_key].shape} vs {model_state[key].shape}")

                        # Copy to recipient_aux_mlp
                        if recipient_key in current_state:
                            if current_state[recipient_key].shape == model_state[key].shape:
                                current_state[recipient_key].copy_(model_state[key])
                                print(f"  ✓ Copied {local_key} → {recipient_key}")
                            else:
                                print(f"  ✗ Shape mismatch for {recipient_key}: {current_state[recipient_key].shape} vs {model_state[key].shape}")
            
            # Load caregiver_final_fc weights if available in checkpoint
            caregiver_final_fc_keys = [k for k in model_state.keys() if 'caregiver_final_fc' in k]
            if caregiver_final_fc_keys:
                print(f"Loading caregiver_final_fc weights from checkpoint...")
                for key in caregiver_final_fc_keys:
                    local_key = _extract_local_key(key)
                    if local_key in current_state:
                        if current_state[local_key].shape == model_state[key].shape:
                            current_state[local_key].copy_(model_state[key])
                            print(f"  ✓ Loaded {local_key}")
                        else:
                            print(f"  ✗ Shape mismatch for {local_key}: {current_state[local_key].shape} vs {model_state[key].shape}")
                    else:
                        print(f"  ✗ Key not found in current model: {local_key}")
            
            # Load recipient_final_fc weights if available in checkpoint
            recipient_final_fc_keys = [k for k in model_state.keys() if 'recipient_final_fc' in k]
            if recipient_final_fc_keys:
                print(f"Loading recipient_final_fc weights from checkpoint...")
                for key in recipient_final_fc_keys:
                    local_key = _extract_local_key(key)
                    if local_key in current_state:
                        if current_state[local_key].shape == model_state[key].shape:
                            current_state[local_key].copy_(model_state[key])
                            print(f"  ✓ Loaded {local_key}")
                        else:
                            print(f"  ✗ Shape mismatch for {local_key}: {current_state[local_key].shape} vs {model_state[key].shape}")
                    else:
                        print(f"  ✗ Key not found in current model: {local_key}")
            
            # Load final_fc weights if available (and copy to role-based format if needed)
            final_fc_keys = [k for k in model_state.keys() if 'final_fc' in k and 'caregiver_final_fc' not in k and 'recipient_final_fc' not in k]
            if final_fc_keys:
                print(f"Loading final_fc weights from checkpoint...")
                for key in final_fc_keys:
                    local_key = _extract_local_key(key)

                    # First try to load to the original key
                    if local_key in current_state:
                        if current_state[local_key].shape == model_state[key].shape:
                            current_state[local_key].copy_(model_state[key])
                            print(f"  ✓ Loaded {local_key}")
                        else:
                            print(f"  ✗ Shape mismatch for {local_key}: {current_state[local_key].shape} vs {model_state[key].shape}")
                    else:
                        print(f"  ✗ Key not found in current model: {local_key}")

                    # If weight_share=False, also copy to caregiver and recipient versions (only if they don't exist in checkpoint)
                    if not self.weight_share and not caregiver_final_fc_keys and not recipient_final_fc_keys:
                        caregiver_key = local_key.replace('final_fc', 'caregiver_final_fc')
                        recipient_key = local_key.replace('final_fc', 'recipient_final_fc')

                        # Copy to caregiver_final_fc
                        if caregiver_key in current_state:
                            if current_state[caregiver_key].shape == model_state[key].shape:
                                current_state[caregiver_key].copy_(model_state[key])
                                print(f"  ✓ Copied {local_key} → {caregiver_key}")
                            else:
                                print(f"  ✗ Shape mismatch for {caregiver_key}: {current_state[caregiver_key].shape} vs {model_state[key].shape}")

                        # Copy to recipient_final_fc
                        if recipient_key in current_state:
                            if current_state[recipient_key].shape == model_state[key].shape:
                                current_state[recipient_key].copy_(model_state[key])
                                print(f"  ✓ Copied {local_key} → {recipient_key}")
                            else:
                                print(f"  ✗ Shape mismatch for {recipient_key}: {current_state[recipient_key].shape} vs {model_state[key].shape}")
            
            # Load sigma parameters if available
            sigma_keys = ['sigma']  # Common sigma parameter name
            for sigma_name in sigma_keys:
                # Try different key formats
                possible_keys = [
                    f'a2c_network.module.{sigma_name}',
                    f'a2c_network.{sigma_name}',
                    f'module.a2c_network.{sigma_name}',
                    f'module.{sigma_name}',
                    sigma_name
                ]
                
                found = False
                for key in possible_keys:
                    if key in model_state:
                        if sigma_name in current_state:
                            if current_state[sigma_name].shape == model_state[key].shape:
                                current_state[sigma_name].copy_(model_state[key])
                                print(f"  ✓ Loaded {sigma_name}")
                                found = True
                                break
                            else:
                                print(f"  ✗ Shape mismatch for {sigma_name}: {current_state[sigma_name].shape} vs {model_state[key].shape}")
                        else:
                            print(f"  ✗ Key not found in current model: {sigma_name}")
                        found = True
                        break
                
                if not found:
                    print(f"  ✗ {sigma_name} not found in checkpoint")
            
            # Load trajectory_cnn weights if available in checkpoint
            trajectory_cnn_keys = [k for k in model_state.keys() if 'trajectory_cnn' in k]
            if trajectory_cnn_keys:
                print(f"Loading trajectory_cnn weights from checkpoint...")

                # trajectory_cnn should already be initialized in _initialize_trajectory_cnn()
                if not self.trajectory_cnn_initialized:
                    error_msg = "trajectory_cnn should be initialized in _initialize_trajectory_cnn()"
                    print(error_msg)
                    raise ValueError(error_msg)

                # Load the weights
                current_state = self.state_dict()
                loaded_count = 0
                failed_count = 0

                for key in trajectory_cnn_keys:
                    local_key = _extract_local_key(key)
                    if local_key in current_state:
                        if current_state[local_key].shape == model_state[key].shape:
                            current_state[local_key].copy_(model_state[key])
                            loaded_count += 1
                        else:
                            print(f"  ✗ Shape mismatch for {local_key}: {current_state[local_key].shape} vs {model_state[key].shape}")
                            failed_count += 1
                    else:
                        print(f"  ✗ Key not found in current model: {local_key}")
                        failed_count += 1

                print(f"  trajectory_cnn loading summary: {loaded_count} loaded, {failed_count} failed out of {len(trajectory_cnn_keys)} total")
                if loaded_count > 0:
                    print(f"  ✓ trajectory_cnn weights successfully loaded from checkpoint")
                else:
                    print(f"  ✗ WARNING: No trajectory_cnn weights were loaded!")
            else:
                print(f"  ✗ No trajectory_cnn weights found in checkpoint")

            # ===== Initialize aux_mlp weights from PNN (following humanx_pnn_builder approach) =====
            # Skip aux_mlp weight loading in residual_mode (keep random initialization)
            if self.residual_mode:
                print("\n=== RESIDUAL MODE: Skipping aux_mlp weight initialization from PNN ===")
                print("  aux_mlp will use random initialization (PyTorch default)")
                print(f"✓ Additional modules loaded from experiment checkpoint (aux_mlp kept random)")
                self.load_state_dict(current_state, strict=True)
                return  # Skip aux_mlp weight loading

            print("\n=== Initializing aux_mlp weights from PNN ===")

            # Check if caregiver_aux_mlp/recipient_aux_mlp exist in checkpoint
            # If not, load from shared PNN weights
            if not self.weight_share:
                has_caregiver_aux = any('caregiver_aux_mlp' in k for k in model_state.keys())
                has_recipient_aux = any('recipient_aux_mlp' in k for k in model_state.keys())

                if not has_caregiver_aux and not has_recipient_aux:
                    print("No role-specific aux_mlp weights found in checkpoint")
                    print("Loading shared PNN weights into both caregiver_aux_mlp and recipient_aux_mlp")

                    # Extract shared PNN weights (a2c_network.pnn.*)
                    # Skip first layer due to input dimension mismatch (following humanx_pnn_builder)
                    pnn_state = {}
                    first_layer_weights = {}
                    for key, value in model_state.items():
                        if 'a2c_network.pnn.actors.0' in key:
                            # Remove 'a2c_network.pnn.' prefix to get PNN state dict key
                            pnn_key = key.replace('a2c_network.pnn.', '')

                            # Handle first layer weight/bias specially due to input dimension mismatch
                            # Weight: [output_dim, input_dim] -> input_dim changed from 2026 to aux_mlp_input_size
                            # We'll copy the first 2026 dims and zero-initialize the rest
                            if pnn_key.startswith('actors.0.0.weight') or pnn_key.startswith('actors.0.0.bias'):
                                first_layer_weights[pnn_key] = value
                                print(f"  Found first layer parameter {pnn_key}: shape {value.shape}")
                                continue  # Skip adding to pnn_state

                            pnn_state[pnn_key] = value

                    if pnn_state:
                        print(f"Found {len(pnn_state)} shared PNN parameters (excluding first layer)")

                        # Load the same weights into both caregiver and recipient aux_mlp (strict=False to allow missing keys)
                        caregiver_result = self.caregiver_aux_mlp.load_state_dict(pnn_state, strict=False)
                        recipient_result = self.recipient_aux_mlp.load_state_dict(pnn_state, strict=False)

                        print(f"  Loaded {len(pnn_state) - len(caregiver_result.missing_keys)} parameters into caregiver_aux_mlp")
                        print(f"  Loaded {len(pnn_state) - len(recipient_result.missing_keys)} parameters into recipient_aux_mlp")

            # PNN input size (from checkpoint) - common dimensions to copy
            common_input_dim = self.self_obs_size + self.task_obs_size  # 2026 dims

            print(f"\nPartially loading first layer (input size mismatch):")
            print(f"Common input dimensions to copy from PNN: {common_input_dim}")
            print(f"Current aux_mlp input size: {self.aux_mlp_input_size}")

            # Get PNN's first layer weights from checkpoint
            # Look for 'pnn.actors.0.0.weight' in checkpoint
            pnn_first_layer_weight = None
            for key in model_state.keys():
                if 'pnn.actors.0.0.weight' in key:
                    pnn_first_layer_weight = model_state[key]
                    print(f"Found PNN first layer weight: {key}, shape: {pnn_first_layer_weight.shape}")
                    break

            if pnn_first_layer_weight is not None:
                checkpoint_shape = pnn_first_layer_weight.shape  # e.g., [2048, 2026]

                # Initialize aux_mlp first layer weights for both caregiver and recipient
                aux_mlp_modules = []
                if self.weight_share:
                    aux_mlp_modules.append(('aux_mlp', self.aux_mlp))
                else:
                    aux_mlp_modules.append(('caregiver_aux_mlp', self.caregiver_aux_mlp))
                    aux_mlp_modules.append(('recipient_aux_mlp', self.recipient_aux_mlp))

                for mlp_name, mlp_module in aux_mlp_modules:
                    # Get first layer of aux_mlp (PNN structure: actors[0][0])
                    # aux_mlp is now a PNN, so we access actors[0][0] for the first Linear layer
                    first_actor = mlp_module.actors[0]  # Sequential network
                    first_layer = first_actor[0]  # First Linear layer

                    # Verify it's a Linear layer
                    if not isinstance(first_layer, nn.Linear):
                        error_msg = f"Unexpected first layer type: {type(first_layer)}"
                        raise ValueError(error_msg)

                    current_weight_param = first_layer.weight  # Shape: [output_dim, input_dim]
                    current_shape = current_weight_param.shape  # e.g., [256, aux_mlp_input_size]

                    print(f"\n  Partially loading {mlp_name} first layer:")
                    print(f"    Checkpoint shape: {checkpoint_shape}")
                    print(f"    Current shape: {current_shape}")

                    # Get current weight tensor (already randomly initialized by PyTorch)
                    new_weight = current_weight_param.clone()

                    # Copy checkpoint weights for common dimensions (first self_obs_size+task_obs_size dims)
                    # Copy only the overlapping output dimensions
                    output_dim_to_copy = min(checkpoint_shape[0], current_shape[0])
                    new_weight[:output_dim_to_copy, :common_input_dim] = pnn_first_layer_weight[:output_dim_to_copy, :]

                    print(f"    ✓ Copied first {common_input_dim} input dimensions from checkpoint")
                    print(f"    ✓ Copied output dimensions: {output_dim_to_copy}/{current_shape[0]}")

                    # Zero-initialize the remaining input dimensions (common_input_dim and beyond)
                    if current_shape[1] > common_input_dim:
                        new_weight[:, common_input_dim:].zero_()
                        print(f"    ✓ Zero-initialized remaining {current_shape[1] - common_input_dim} input dimensions")

                    # Update the parameter in place (use no_grad to avoid autograd issues)
                    with torch.no_grad():
                        current_weight_param.copy_(new_weight)

                    print(f"    ✓ {mlp_name} first layer weights initialized")
            else:
                print(f"  ✗ PNN first layer weight not found in checkpoint")
                print(f"  ✗ Skipping aux_mlp weight initialization from PNN")

            print(f"✓ Additional modules loaded from experiment checkpoint")
            self.load_state_dict(current_state, strict=True)
        
        def _update_aux_features_stats(self, aux_features):
            """Update aux_features normalization statistics using Welford's algorithm"""
            # Implementation follows PHC's RunningMeanStd update mechanism
            aux_start_idx = self.base_self_obs_size + self.task_obs_size
            aux_end_idx = aux_start_idx + self.aux_features_size
            
            # Calculate batch statistics
            batch_mean = aux_features.mean(dim=0)  # Mean across batch dimension
            batch_var = aux_features.var(dim=0, unbiased=False)  # Variance across batch dimension
            batch_count = aux_features.size(0)  # Batch size
            
            # Get current statistics
            current_mean = self.running_mean[aux_start_idx:aux_end_idx]
            current_var = self.running_var[aux_start_idx:aux_end_idx]
            current_count = self.aux_features_count

            # print(f"DEBUG: batch_count={batch_count}, current_count={current_count}")
            # print(f"DEBUG: batch_mean mean={batch_mean.mean().item():.4f}, batch_var mean={batch_var.mean().item():.4f}")
            
            # Update count
            new_count = current_count + batch_count
            
            # Update mean using Welford's algorithm
            delta = batch_mean - current_mean
            new_mean = current_mean + delta * batch_count / new_count
            
            # Update variance using Welford's algorithm
            delta2 = batch_mean - new_mean
            new_var = (current_var * current_count + batch_var * batch_count + 
                      delta * delta2 * current_count * batch_count / new_count) / new_count
            
            # Update the running statistics
            self.running_mean[aux_start_idx:aux_end_idx] = new_mean
            self.running_var[aux_start_idx:aux_end_idx] = new_var
            self.aux_features_count = new_count
            
            # print(f"DEBUG: After update - new_count={new_count}")
            # print(f"DEBUG: new_mean mean={new_mean.mean().item():.4f}, new_var mean={new_var.mean().item():.4f}")

        def _calculate_self_contact_obs_size(self):
            """Calculate self contact observation size based on config"""
            # Check if contact features are enabled based on obs_v
            # obs_v=4 includes contact features, obs_v=1 does not
            # if hasattr(self, 'obs_v') and self.self_obs_v == 4:
            #     # Contact features enabled - based on env_im_interx_multirl2.yaml hand_contact_bodies
            hand_contact_bodies = [
                "R_Wrist", "L_Wrist", "R_Index3", "L_Index3", 
                "R_Middle3", "L_Middle3", "R_Ring3", "L_Ring3", 
                "R_Pinky3", "L_Pinky3", "R_Thumb3", "L_Thumb3"
            ]
            # Each contact body: only contact flag = 1 value
            contact_obs_size = len(hand_contact_bodies) * 1
            print(f"DEBUG: _calculate_self_contact_obs_size (obs_v={self.obs_v}) = {contact_obs_size}")
            return contact_obs_size
            # else:
            #     # No contact features
            #     print(f"DEBUG: _calculate_self_contact_obs_size (obs_v={getattr(self, 'self_obs_v', 'unknown')}) = 0 (no contact)")
            #     return 0
            
        def _calculate_partner_contact_obs_size(self):
            """Calculate partner contact observation size based on config"""
            # Check partner observation version - partner_obs_v=3 includes contact, partner_obs_v=2 does not
            partner_obs_v = getattr(self.task_obs_size_detail, 'partner_obs_v', 3)  # Default to no contact
            if partner_obs_v == 3:
                # Contact features enabled - same calculation as self contact
                hand_contact_bodies = [
                    "R_Wrist", "L_Wrist", "R_Index3", "L_Index3", 
                    "R_Middle3", "L_Middle3", "R_Ring3", "L_Ring3", 
                    "R_Pinky3", "L_Pinky3", "R_Thumb3", "L_Thumb3"
                ]
                # Each contact body: only contact flag = 1 value
                contact_obs_size = len(hand_contact_bodies) * 1
                print(f"DEBUG: _calculate_partner_contact_obs_size (partner_obs_v={partner_obs_v}) = {contact_obs_size}")
                return contact_obs_size
            else:
                # No contact features
                print(f"DEBUG: _calculate_partner_contact_obs_size (partner_obs_v={partner_obs_v}) = 0 (no contact)")
                return 0

        def _calculate_reference_contact_obs_size(self):
            """Calculate reference contact observation size (DISABLED for MultiPulse compatibility)"""
            # MultiPulse excludes reference contact diff
            reference_obs_size = 0
            print(f"DEBUG: _calculate_reference_contact_obs_size = {reference_obs_size} (disabled for MultiPulse compatibility)")
            return reference_obs_size

        # def _build_contact_mlp(self, input_size, output_size, units, zero_init=True):
        #     """Build contact information processing MLP"""
        #     if input_size == 0:
        #         return nn.Identity()
                
        #     layers = []
            
        #     # Input layer
        #     layers.append(nn.Linear(input_size, units[0]))
        #     if zero_init:
        #         nn.init.zeros_(layers[-1].weight)
        #         nn.init.zeros_(layers[-1].bias)
        #     layers.append(nn.SiLU())
            
        #     # Hidden layers
        #     for i in range(len(units) - 1):
        #         layers.append(nn.Linear(units[i], units[i + 1]))
        #         if zero_init:
        #             nn.init.zeros_(layers[-1].weight)
        #             nn.init.zeros_(layers[-1].bias)
        #         layers.append(nn.SiLU())
            
        #     # Output layer
        #     layers.append(nn.Linear(units[-1], output_size))
        #     if zero_init:
        #         nn.init.zeros_(layers[-1].weight)
        #         nn.init.zeros_(layers[-1].bias)
                
        #     return nn.Sequential(*layers)
            
        def _extract_aux_features(self, obs):
            """Extract aux_features from observation: [self_contact, partner_obs, partner_contact, role_label] (MultiPulse compatible + role)"""
            # obs format: [self_obs_feat, task_obs, aux_features]
            # aux_features = [self_contact, partner_obs, partner_contact, role_label] (includes role for recipient/caregiver determination)
            start_idx = self.base_self_obs_size + self.task_obs_size
            aux_features = obs[:, start_idx:start_idx + self.aux_features_size]
            return aux_features
                
        # def _update_contact_obs_stats(self, contact_obs):
        #     """Update contact observation normalization statistics"""
        #     if contact_obs.shape[1] == 0:
        #         return
                
        #     # Calculate batch statistics
        #     batch_mean = contact_obs.mean(dim=0)
        #     batch_var = contact_obs.var(dim=0, unbiased=False)
        #     batch_count = contact_obs.size(0)
            
        #     # Get current statistics
        #     current_mean = self.running_mean[self.contact_obs_start_idx:self.contact_obs_end_idx]
        #     current_var = self.running_var[self.contact_obs_start_idx:self.contact_obs_end_idx]
        #     current_count = self.contact_obs_count
            
        #     # Update count
        #     new_count = current_count + batch_count
            
        #     # Update mean using Welford's algorithm
        #     delta = batch_mean - current_mean
        #     new_mean = current_mean + delta * batch_count / new_count
            
        #     # Update variance using Welford's algorithm
        #     delta2 = batch_mean - new_mean
        #     new_var = (current_var * current_count + batch_var * batch_count + 
        #               delta * delta2 * current_count * batch_count / new_count) / new_count
            
        #     # Update the running statistics
        #     self.running_mean[self.contact_obs_start_idx:self.contact_obs_end_idx] = new_mean
        #     self.running_var[self.contact_obs_start_idx:self.contact_obs_end_idx] = new_var
        #     self.contact_obs_count = new_count

        def eval_actor(self, obs_dict):
            obs = obs_dict['obs']
            # print(f"DEBUG: eval_actor called with obs shape: {obs.shape}")

            # Split observations into components: [self_obs_feat, task_obs, aux_features]
            # Extract main observations (self_obs_feat + task_obs) for PNN
            main_obs_size = self.base_self_obs_size + self.task_obs_size
            main_obs = obs[:, :main_obs_size]

            # Extract aux_features: [self_contact, partner_obs, partner_contact, role_label]
            aux_features = self._extract_aux_features(obs)

            # Debug: Print observation statistics and aux_mlp weights periodically
            if hasattr(self, '_debug_step_count'):
                self._debug_step_count += 1
            else:
                self._debug_step_count = 0

            # Print aux_mlp weight statistics periodically
            debug_interval = getattr(self, 'aux_mlp_debug_interval', 1000)  # Default every 1000 steps
            # if self._debug_step_count % debug_interval == 0:
            #     self._print_aux_mlp_weights_debug()

            # NOTE: aux_features normalization is handled automatically by RunningMeanStd in _preproc_obs
            # Do NOT manually update statistics here to avoid double-update conflicts
            # if self.training:
            #     self._update_aux_features_stats(aux_features)

            # Convert to float32 to match aux_mlp dtype
            aux_features_normalized = aux_features.float()

            # Extract observations for PNN: [base_self_obs, task_obs]
            # main_obs contains: [base_self_obs, task_obs] (first part of full observation)
            pnn_input = main_obs  # Should be exactly base_self_obs + task_obs = 2026 dims
            # Pass main observations through CNN (empty for this case but keeps consistency)
            a_out = self.actor_cnn(pnn_input)
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            # Extract future trajectory features from aux_features and encode with CNN
            future_trajectory_features = torch.zeros(obs.shape[0], self.future_trajectory_cnn_output_size,
                                                   device=obs.device, dtype=torch.float32)

            try:
                # Extract future trajectory data from aux_features
                # aux_features structure: [self_contact, self_force, base_partner_obs, partner_contact, self_future_traj_flat, partner_future_traj_flat, role_label]

                # Calculate start indices for future trajectory data - properly calculate sizes
                # Hand contact features: 12 (2 wrists + 10 fingertips)
                self_contact_size = 12

                self_force_size = 42
                num_bodies = 52  # SMPL-X body count
                base_partner_obs_size = num_bodies * 3 + num_bodies * 3 + num_bodies * 6 + 6  # root_pos + joint_pos + joint_vel + root_rot_gravity_6d = 315

                base_partner_obs_size += num_bodies * 6  # wrist-relative pos + vel = 312
                    # Total: 315 + 312 = 627

                partner_contact_size = 12   # Partner contact features (same as self_contact_size)
                # Future trajectory sizes: 30 frames * 22 bodies * 13 features = 8580 per trajectory (hand joints removed)
                future_traj_size = self.future_size * 22 * 13
                action_size = 153

                # Debug: Print calculated sizes
                # print(f"DEBUG: Size calculations - self_contact: {self_contact_size}, self_force: {self_force_size}, base_partner: {base_partner_obs_size}, partner_contact: {partner_contact_size}, future_traj: {future_traj_size}")
                # print(f"DEBUG: Total aux_features expected: {self_contact_size + self_force_size + base_partner_obs_size + partner_contact_size + future_traj_size * 2 + 1} (including role_label)")
                # print(f"DEBUG: Actual aux_features size: {aux_features_normalized.shape[1]}")
                # Extract flattened future trajectories from aux_features
                traj_start_idx = self_contact_size + self_force_size + base_partner_obs_size + partner_contact_size + action_size
                self_traj_end_idx = traj_start_idx + future_traj_size
                partner_traj_end_idx = self_traj_end_idx + future_traj_size
                
                assert aux_features_normalized.shape[1] == traj_start_idx + future_traj_size * 2 + 1

                self_future_traj_flat = aux_features_normalized[:, traj_start_idx:self_traj_end_idx]  # [batch, 8580]
                partner_future_traj_flat = aux_features_normalized[:, self_traj_end_idx:partner_traj_end_idx]  # [batch, 8580]

                # Reshape to [batch, 30, 286] format for CNN processing (30 frames, 22 bodies * 13 features = 286)
                batch_size = obs.shape[0]
                features_per_frame = 22 * 13  # 22 non-hand bodies * 13 features each = 286
                self_future_traj = self_future_traj_flat.view(batch_size, self.future_size, features_per_frame)   # [batch, 30, 286]
                partner_future_traj = partner_future_traj_flat.view(batch_size, self.future_size, features_per_frame)  # [batch, 30, 286]

                # Combine self and partner trajectories along channel dimension
                combined_trajectories = torch.cat([self_future_traj, partner_future_traj], dim=2)  # [batch, 30, 572]

                # Encode with CNN to get 128-dimensional features
                future_trajectory_features = self._encode_future_trajectories(combined_trajectories)
            except Exception as e:
                error_msg = f"Warning: Failed to extract future trajectory features: {e}"
                print(error_msg)
                raise Exception(error_msg)
                # Use zero features as fallback

            # Prepare aux_mlp input: [aux_features_without_future_traj, filtered_pnn_input, cnn_encoded_future_traj]
            # Remove future trajectory data from aux_features to avoid double processing
            aux_features_without_future_traj = torch.cat([
                aux_features_normalized[:, :traj_start_idx],  # [self_contact, self_force, base_partner_obs, partner_contact]
                aux_features_normalized[:, partner_traj_end_idx:]  # [role_label]
            ], dim=1)

            aux_mlp_input = torch.cat([a_out, aux_features_without_future_traj], dim=1) #future_trajectory_features

            # Compute aux_mlp output BEFORE PNN forward pass
            # aux_mlp is now a PNN, so we call it with dummy aux_intermediates=[]
            aux_mlp_output = None
            # Extract role from observation (last element of aux_features)
            aux_start_idx = self.base_self_obs_size + self.task_obs_size
            role_idx = aux_start_idx + self.aux_features_size - 1  # Last element is role label
            role_labels = obs[:, role_idx]  # Last element is role label

            assert role_labels.unique().shape[0] == 2, f"role_labels should have exactly two unique values, got {role_labels.unique()}"
            if self.weight_share:
                # Shared weights - compute aux_mlp (PNN) output
                aux_mlp_output, _ = self.aux_mlp(aux_mlp_input, idx=self.training_prim, aux_intermediates=[])
            else:
                # Separate weights - determine role and use appropriate PNNs
                batch_size = obs.shape[0]

                is_recipient = (role_labels <= 0)  # Handle both original [0,1] and normalized [-1,1]: <=0 for recipient, >0 for caregiver

                # Process caregiver and recipient separately
                caregiver_mask = ~is_recipient
                recipient_mask = is_recipient

                # Initialize outputs - get output_size from aux_mlp config
                output_size = 153  # Default action size
                aux_mlp_output = torch.zeros(batch_size, output_size, device=obs.device)

                if caregiver_mask.any():
                    caregiver_aux_output, _ = self.caregiver_aux_mlp(aux_mlp_input[caregiver_mask], idx=self.training_prim, aux_intermediates=[])
                    aux_mlp_output[caregiver_mask] = caregiver_aux_output

                if recipient_mask.any():
                    if self.freeze_recipient:
                        # Recipient is frozen - use no_grad to avoid unnecessary computation
                        with torch.no_grad():
                            recipient_aux_output, _ = self.recipient_aux_mlp(aux_mlp_input[recipient_mask], idx=self.training_prim, aux_intermediates=[])
                            aux_mlp_output[recipient_mask] = recipient_aux_output
                    else:
                        recipient_aux_output, _ = self.recipient_aux_mlp(aux_mlp_input[recipient_mask], idx=self.training_prim, aux_intermediates=[])
                        aux_mlp_output[recipient_mask] = recipient_aux_output

            # Pass through main PNN (im_pnn_big) with dummy aux_intermediates
            main_features, _ = self.pnn(a_out, idx=self.training_prim, aux_intermediates=[])

            # Debug: Print PNN statistics
            # if self._debug_step_count % 1000 == 0:
            #     print(f"DEBUG: PNN input shape: {a_out.shape}, output shape: {main_features.shape}")
            #     print(f"DEBUG: PNN input stats - mean: {a_out.mean().item():.4f}, std: {a_out.std().item():.4f}")
            #     print(f"DEBUG: PNN output stats - mean: {main_features.mean().item():.4f}, std: {main_features.std().item():.4f}")

            if self.weight_share:
                if self.residual_mode:
                    # Residual mode: simple addition of PNN and aux_mlp outputs
                    a_out = main_features + aux_mlp_output
                else:
                    # Combine PNN output with aux_mlp output via final_fc
                    combined_features = torch.cat([main_features, aux_mlp_output], dim=1)
                    a_out = self.final_fc(combined_features)
            else:
                # For separate weights, use role-specific processing
                batch_size = obs.shape[0]
                a_out = torch.zeros(batch_size, main_features.shape[1], device=obs.device)

                if self.residual_mode:
                    # Residual mode: simple addition of PNN and aux_mlp outputs
                    if caregiver_mask.any():
                        a_out[caregiver_mask] = main_features[caregiver_mask] + aux_mlp_output[caregiver_mask]
                    if recipient_mask.any():
                        a_out[recipient_mask] = main_features[recipient_mask] + aux_mlp_output[recipient_mask]
                else:
                    if caregiver_mask.any():
                        # caregiver_combined = torch.cat([main_features[caregiver_mask], aux_mlp_output[caregiver_mask]], dim=1)
                        # a_out[caregiver_mask] = self.caregiver_final_fc(caregiver_combined)
                        a_out[caregiver_mask] = aux_mlp_output[caregiver_mask]

                    if recipient_mask.any():
                        a_out[recipient_mask] = aux_mlp_output[recipient_mask]
                        # else:
                            # recipient_combined = torch.cat([main_features[recipient_mask], aux_mlp_output[recipient_mask]], dim=1)
                            # a_out[recipient_mask] = self.recipient_final_fc(recipient_combined)

            # Debug: Check for NaN/Inf values
            if torch.isnan(main_features).any() or torch.isinf(main_features).any():
                print(f"WARNING: NaN/Inf detected in main_features")

            if torch.isnan(a_out).any() or torch.isinf(a_out).any():
                print(f"WARNING: NaN/Inf detected in final output")
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = a_out
                if self.space_config['fixed_sigma']:
                    # Use role-based sigma selection
                    role_sigma = self._get_role_based_sigma(obs_dict)
                    sigma = mu * 0.0 + self.sigma_act(role_sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def eval_critic(self, obs_dict):
            """Override eval_critic to implement trajectory CNN encoding + asymmetric critic"""
            obs = obs_dict['obs']
            seq_length = obs_dict.get('seq_length', 1)
            states = obs_dict.get('rnn_states', None)

            # Apply CNN trajectory encoding (same logic as actor forward)
            critic_obs = self._apply_critic_trajectory_encoding(obs)

            if self.asymmetric_critic:
                # Asymmetric critic: Get additional partner features
                batch_size = obs.shape[0]
                env_ids = torch.arange(batch_size, dtype=torch.long, device=obs.device)

                # Compute additional partner features - only available during training from environment
                if hasattr(self, '_env_ref') and self._env_ref is not None:
                    partner_task_obs = self._env_ref._compute_partner_task_obs(env_ids)
                    partner_force_obs = self._env_ref._compute_partner_force_obs(env_ids)

                    # Apply finger filtering to partner_task_obs if enabled
                    if self.filter_finger_joints and partner_task_obs.shape[1] > self.filtered_partner_task_obs_size:
                        task_obs_keep_indices = self._get_task_obs_keep_indices()
                        partner_task_obs = partner_task_obs[:, task_obs_keep_indices]
                else:
                    # During inference or when environment reference is not available
                    task_obs_size = self.filtered_partner_task_obs_size
                    force_obs_size = getattr(self, '_partner_force_obs_size', 42)
                    partner_task_obs = torch.zeros(batch_size, task_obs_size, device=obs.device)
                    partner_force_obs = torch.zeros(batch_size, force_obs_size, device=obs.device)

                # Append partner features to CNN-processed observation
                critic_obs = torch.cat([critic_obs, partner_task_obs, partner_force_obs], dim=1)
                # Note: Partner features are not normalized since they come directly from environment
                critic_obs_normalized = critic_obs
            else:
                # Symmetric critic: obs already normalized by agent
                critic_obs_normalized = critic_obs

            # Ensure consistent dtype (float32)
            critic_obs_normalized = critic_obs_normalized.float()

            # Process through critic network
            c_out = self.critic_cnn(critic_obs_normalized)
            c_out = c_out.contiguous().view(-1, c_out.size(-1))

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    c_out_in = c_out
                    c_out = self.critic_mlp(c_out_in)

                    if self.rnn_concat_input:
                        c_out = torch.cat([c_out, c_out_in], dim=1)

                batch_size_rnn = c_out.size()[0]
                num_seqs = batch_size_rnn // seq_length
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)

                # RNN processing for critic states
                if len(states) == 2:
                    c_states = states[1].reshape(num_seqs, seq_length, -1)
                else:
                    c_states = states[2:].reshape(num_seqs, seq_length, -1)
                c_out, c_states = self.c_rnn(c_out, c_states[:, 0:1].transpose(0, 1).contiguous())

                if self.rnn_name == 'sru':
                    c_out = c_out.transpose(0, 1)
                else:
                    if self.rnn_ln:
                        c_out = self.c_layer_norm(c_out)
                c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                if type(c_states) is not tuple:
                    c_states = (c_states,)

                if self.is_rnn_before_mlp:
                    c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))
                return value, c_states

            else:
                c_out = self.critic_mlp(c_out)
                value = self.value_act(self.value(c_out))
                return value

        def set_env_ref(self, env_ref):
            """Set environment reference for accessing partner observations during training (if asymmetric critic enabled)"""
            if self.asymmetric_critic:
                self._env_ref = env_ref
                # Store observation sizes for inference fallback
                if hasattr(env_ref, 'get_task_obs_size'):
                    self._partner_task_obs_size = env_ref.get_task_obs_size()
                else:
                    self._partner_task_obs_size = 1248  # Default task_obs size
                
                if hasattr(env_ref, '_get_hand_force_obs_size'):
                    self._partner_force_obs_size = env_ref._get_hand_force_obs_size()
                else:
                    self._partner_force_obs_size = 42  # Default force_obs size
                
                print(f"Set environment reference for asymmetric critic - partner_task_obs_size: {self._partner_task_obs_size}, partner_force_obs_size: {self._partner_force_obs_size}")
            else:
                print(f"Symmetric critic enabled, no environment reference needed")

        def _rebuild_critic_mlp(self):
            """Rebuild critic MLP to handle critic observation size (asymmetric or symmetric)"""
            # Calculate actual critic observation size after CNN trajectory encoding
            if self.asymmetric_critic:
                # Asymmetric critic: CNN-processed actor obs + partner features
                actual_critic_obs_size = self.critic_obs_size
            else:
                # Symmetric critic: Calculate size after CNN trajectory encoding
                # base_obs + aux_features_without_future_traj + CNN_encoded(128) + role_label
                future_trajectory_size = self.future_size * 22 * 13 * 2  # Original trajectory size
                aux_features_without_future_traj_size = self.aux_features_size - future_trajectory_size  # Remove trajectories
                cnn_encoded_size = 0

                actual_critic_obs_size = (self.base_self_obs_size + self.task_obs_size +
                                        aux_features_without_future_traj_size + cnn_encoded_size)

           
            if self.asymmetric_critic:
                print(f"DEBUG: Rebuilding critic MLP for asymmetric observations")
            else:
                print(f"DEBUG: Rebuilding critic MLP for symmetric observations with CNN trajectory encoding")
            
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


        def _apply_critic_trajectory_encoding(self, obs):
            """Apply CNN trajectory encoding to critic observations (same as actor logic)"""
            # Extract base self observation (self_obs + task_obs)
            base_obs_size = self.base_self_obs_size + self.task_obs_size
            base_obs = obs[:, :base_obs_size]
            aux_features = obs[:, base_obs_size:]

            # obs is already normalized by agent's _preproc_obs, so use directly
            aux_features_normalized = aux_features

            # Extract and encode future trajectories (same logic as actor forward)
            future_trajectory_features = torch.zeros(obs.shape[0], self.future_trajectory_cnn_output_size,
                                                   device=obs.device, dtype=torch.float32)
            # Extract future trajectory data from aux_features
            self_contact_size = 12
            self_force_size = 42
            num_bodies = 52
            base_partner_obs_size = num_bodies * 3 + num_bodies * 3 + num_bodies * 6 + 6
            if hasattr(self, 'obs_v') and self.obs_v >= 4:
                base_partner_obs_size += num_bodies * 6
            partner_contact_size = 12
            future_traj_size = self.future_size * 22 * 13
            action_size = 153

            # Extract flattened future trajectories
            traj_start_idx = self_contact_size + self_force_size + base_partner_obs_size + partner_contact_size + action_size
            self_traj_end_idx = traj_start_idx + future_traj_size
            partner_traj_end_idx = self_traj_end_idx + future_traj_size
            role_idx_in_aux = traj_start_idx + future_traj_size * 2  # Index within aux_features
            role_labels = aux_features_normalized[:, role_idx_in_aux]  # Access from aux_features, not obs
            assert role_labels.unique().shape[0] == 2, "role_labels should have exactly two unique values"

            self_future_traj_flat = aux_features_normalized[:, traj_start_idx:self_traj_end_idx]
            partner_future_traj_flat = aux_features_normalized[:, self_traj_end_idx:partner_traj_end_idx]

            # Reshape and combine trajectories
            batch_size = obs.shape[0]
            features_per_frame = 22 * 13
            self_future_traj = self_future_traj_flat.view(batch_size, self.future_size, features_per_frame)
            partner_future_traj = partner_future_traj_flat.view(batch_size, self.future_size, features_per_frame)

            # Combine trajectories
            combined_trajectories = torch.cat([self_future_traj, partner_future_traj], dim=2)

            # Encode with CNN (shared with actor)
            future_trajectory_features = self._encode_future_trajectories(combined_trajectories)
            # Reconstruct observation without raw trajectory data + CNN features
            aux_features_without_future_traj = torch.cat([
                aux_features_normalized[:, :traj_start_idx],  # [self_contact, self_force, base_partner_obs, partner_contact]
                aux_features_normalized[:, partner_traj_end_idx:],  # [role_label]
                #future_trajectory_features,  # [128-dim CNN encoded trajectories]
            ], dim=1)

            # Return processed observation: [base_obs, processed_aux_features]
            return torch.cat([base_obs, aux_features_without_future_traj], dim=1)

        def train(self, mode=True):
            """Override to keep recipient_aux_mlp in eval mode when frozen"""
            super().train(mode)
            if self.freeze_recipient and hasattr(self, 'recipient_aux_mlp') and self.recipient_aux_mlp is not None:
                self.recipient_aux_mlp.eval()
            return self