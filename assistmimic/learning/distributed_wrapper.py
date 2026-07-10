import os
import torch
import torch.distributed as dist


class DistributedWrapper:
    """
    A wrapper to handle PyTorch distributed training without horovod.
    This replaces the horovod functionality in rl_games.
    """
    
    def __init__(self):
        self.rank = 0
        self.rank_size = 1
        self.local_rank = 0
        self.is_distributed = False
        
        # Check if we're in a distributed environment
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.is_distributed = True
            self.setup_distributed()
    
    def setup_distributed(self):
        """Initialize PyTorch distributed training"""
        if not dist.is_initialized():
            self.rank = int(os.environ.get('RANK', '0'))
            self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            self.rank_size = int(os.environ.get('WORLD_SIZE', '1'))
            
            # Initialize the process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=self.rank,
                world_size=self.rank_size
            )
            
            # Set the device for this process
            torch.cuda.set_device(self.local_rank)
            
            print(f"Distributed training initialized - Rank: {self.rank}, Local Rank: {self.local_rank}, World Size: {self.rank_size}")
        else:
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            self.rank_size = dist.get_world_size()
    
    def update_algo_config(self, config):
        """Update algorithm configuration for distributed training"""
        updated_config = config.copy()
        
        if self.is_distributed:
            # Adjust seed for each rank - handle different config structures
            if 'params' in updated_config and 'seed' in updated_config['params']:
                updated_config['params']['seed'] += self.rank
            elif 'seed' in updated_config:
                updated_config['seed'] += self.rank
            
            # Adjust batch sizes for distributed training - handle different config structures
            config_dict = None
            if 'params' in updated_config and 'config' in updated_config['params']:
                config_dict = updated_config['params']['config']
            elif 'config' in updated_config:
                config_dict = updated_config['config']
            else:
                config_dict = updated_config
            
            # For DDP, keep the same batch size per GPU to increase effective batch size
            # The effective batch size becomes: batch_size * num_gpus
            # DDP automatically handles gradient averaging across GPUs
            print(f"DDP: Keeping original batch sizes per GPU. Effective batch size will be multiplied by {self.rank_size}")
            
            # Adjust num_actors (environment count) to be distributed across GPUs
            # This is necessary to prevent memory access errors
            # if config_dict and 'num_actors' in config_dict:
            #     original_num_actors = config_dict['num_actors']
            #     config_dict['num_actors'] = max(1, original_num_actors // self.rank_size)
            #     config_dict['minibatch_size'] = max(1, config_dict['minibatch_size'] // self.rank_size)
            #     config_dict['amp_minibatch_size'] = max(1, config_dict['amp_minibatch_size'] // self.rank_size)
            #     config_dict['amp_batch_size'] = max(1, config_dict['amp_batch_size'] // self.rank_size)
            #     print(f"DDP: Adjusted num_actors from {original_num_actors} to {config_dict['num_actors']} per GPU")
            #     print(f"DDP: Total effective environments = {config_dict['num_actors']} * {self.rank_size} = {config_dict['num_actors'] * self.rank_size}")
        
        return updated_config
    
    def setup_algo(self, algo):
        """Setup algorithm for distributed training"""
        if self.is_distributed and hasattr(algo, 'model'):
            # Ensure the model is on the correct device first
            device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.local_rank)
            # torch.cuda.empty_cache()  # Clear cache to prevent GPU0 memory imbalance
            
            if hasattr(algo.model, 'a2c_network'):
                # Store original network before wrapping
                original_network = algo.model.a2c_network
                
                # Fix GPU memory imbalance by loading state dict through CPU first
                if hasattr(algo.model, 'state_dict'):
                    state_dict = algo.model.state_dict()
                    # Map all tensors to CPU first to avoid GPU0 address references
                    cpu_state_dict = {}
                    for key, tensor in state_dict.items():
                        cpu_state_dict[key] = tensor.cpu()
                    # Load CPU state dict then move to target device
                    algo.model.load_state_dict(cpu_state_dict)
                
                # Move ENTIRE model to correct device first (this includes all submodules)
                algo.model = algo.model.to(device)
                
                # Explicitly ensure ALL submodules are on correct device
                for name, module in algo.model.named_modules():
                    module.to(device)
                
                # Clear cache again before DDP wrapping
                torch.cuda.set_device(self.local_rank)
                # torch.cuda.empty_cache()
                
                # Then wrap with DistributedDataParallel
                ddp_model = torch.nn.parallel.DistributedDataParallel(
                    algo.model.a2c_network,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,   # Enable unused parameter detection for role-based training
                    broadcast_buffers=False,       # Disable buffer broadcasting to reduce GPU0 overhead
                    bucket_cap_mb=25               # Reduce bucket size to minimize memory overhead
                )

                # IMPORTANT: Do NOT set static_graph for distillation/curriculum learning
                # Static graph is incompatible with curriculum learning where parameters
                # are selectively updated in different stages (e.g., actor disabled in early stages)
                # Only enable for standard training where all parameters are always updated
                ddp_model._set_static_graph()
                
                # After DDP wrapping, ensure the wrapped module's submodules are also on correct device
                for name, module in ddp_model.module.named_modules():
                    module.to(device)
                
                # Add all missing methods to the DDP wrapper by forwarding to module
                methods_to_forward = [
                    'is_rnn', 'eval_critic', 'eval_actor', 'eval_disc', 'eval_z',
                    'compute_prior', 'get_disc_logit_weights', 'get_disc_weights',
                    'embedding_size'
                ]
                
                def create_method_wrapper(method_name):
                    def wrapper(*args, **kwargs):
                        return getattr(ddp_model.module, method_name)(*args, **kwargs)
                    return wrapper
                
                for method_name in methods_to_forward:
                    if hasattr(original_network, method_name):
                        attr = getattr(original_network, method_name)
                        if callable(attr):
                            # For methods, create a proper wrapper that calls the underlying method
                            setattr(ddp_model, method_name, create_method_wrapper(method_name))
                        else:
                            # For properties/attributes, forward directly
                            setattr(ddp_model, method_name, attr)
                
                algo.model.a2c_network = ddp_model
            
            # Set the device attributes and move other components
            algo.ppo_device = device
            
            # Move running_mean_std to correct device if it exists
            if hasattr(algo, 'running_mean_std') and algo.running_mean_std is not None:
                algo.running_mean_std = algo.running_mean_std.to(device)
            
            # Move value_mean_std to correct device if it exists
            if hasattr(algo, 'value_mean_std') and algo.value_mean_std is not None:
                algo.value_mean_std = algo.value_mean_std.to(device)
            
            # Move amp_input_mean_std to correct device if it exists
            if hasattr(algo, '_amp_input_mean_std') and algo._amp_input_mean_std is not None:
                algo._amp_input_mean_std = algo._amp_input_mean_std.to(device)
            
            # Move disc_reward_mean_std to correct device if it exists
            if hasattr(algo, '_disc_reward_mean_std') and algo._disc_reward_mean_std is not None:
                algo._disc_reward_mean_std = algo._disc_reward_mean_std.to(device)
            
            # Move other components that might be on wrong device
            if hasattr(algo, 'current_rewards'):
                algo.current_rewards = algo.current_rewards.to(device)
            if hasattr(algo, 'current_lengths'):
                algo.current_lengths = algo.current_lengths.to(device)
            if hasattr(algo, 'dones'):
                algo.dones = algo.dones.to(device)
            
            # Move action bounds to correct device
            if hasattr(algo, 'actions_low'):
                algo.actions_low = algo.actions_low.to(device)
            if hasattr(algo, 'actions_high'):
                algo.actions_high = algo.actions_high.to(device)
            
            # Move game tracking components to correct device
            if hasattr(algo, 'game_rewards') and algo.game_rewards is not None:
                algo.game_rewards = algo.game_rewards.to(device)
            if hasattr(algo, 'game_lengths') and algo.game_lengths is not None:
                algo.game_lengths = algo.game_lengths.to(device)
            
            # Move RNN states if they exist
            if hasattr(algo, 'rnn_states') and algo.rnn_states is not None:
                if isinstance(algo.rnn_states, list):
                    algo.rnn_states = [state.to(device) for state in algo.rnn_states]
                else:
                    algo.rnn_states = algo.rnn_states.to(device)
            
            if hasattr(algo, 'mb_rnn_states') and algo.mb_rnn_states is not None:
                if isinstance(algo.mb_rnn_states, list):
                    algo.mb_rnn_states = [state.to(device) for state in algo.mb_rnn_states]
                else:
                    algo.mb_rnn_states = algo.mb_rnn_states.to(device)
            
            print(f"Rank {self.rank}: Model and components moved to {device}")
            
            # Sync environment tensors to correct device after set_device calls
            self._sync_env_tensors(algo, device)
    
    def _sync_env_tensors(self, algo, device):
        """Sync environment tensors to correct device after set_device"""
        if hasattr(algo, 'vec_env') and hasattr(algo.vec_env, 'env') and hasattr(algo.vec_env.env, 'task'):
            task = algo.vec_env.env.task
            
            # Update task device attribute
            task.device = torch.device(device)
            
            # List of tensors that need to be synced
            tensor_attrs = [
                'progress_buf', 'reset_buf', '_terminate_buf', '_contact_forces',
                'self_obs_buf', 'reward_raw', '_humanoid_root_states', 
                '_dof_pos', '_dof_vel', '_rigid_body_pos', '_rigid_body_rot',
                '_rigid_body_vel', '_rigid_body_ang_vel', '_humanoid_actor_ids',
                '_initial_dof_pos', '_initial_dof_vel'
            ]
            
            for attr_name in tensor_attrs:
                if hasattr(task, attr_name):
                    tensor = getattr(task, attr_name)
                    if isinstance(tensor, torch.Tensor) and tensor.device != torch.device(device):
                        setattr(task, attr_name, tensor.to(device))
                        print(f"Rank {self.rank}: Moved {attr_name} to {device}")
            
            # Sync specific additional tensor attributes
            additional_tensor_attrs = [
                '_termination_heights', '_contact_body_ids', '_key_body_ids',
                'motor_efforts', 'dof_limits_lower', 'dof_limits_upper',
                'kp_gains', 'kd_gains', '_pd_action_offset', '_pd_action_scale'
            ]
            
            for attr_name in additional_tensor_attrs:
                if hasattr(task, attr_name):
                    tensor = getattr(task, attr_name)
                    if isinstance(tensor, torch.Tensor) and tensor.device != torch.device(device):
                        setattr(task, attr_name, tensor.to(device))
                        print(f"Rank {self.rank}: Moved {attr_name} to {device}")
    
    def sync_stats(self, algo):
        """Synchronize statistics across processes"""
        if self.is_distributed:
            # For now, we'll just ensure all processes are synchronized
            dist.barrier()
    
    def broadcast_parameters(self, model):
        """Broadcast model parameters from rank 0 to all other ranks"""
        if self.is_distributed:
            # Use async broadcast to reduce GPU0 memory pressure
            broadcast_handles = []
            for param in model.parameters():
                handle = dist.broadcast(param.data, src=0, async_op=True)
                broadcast_handles.append(handle)
            # Wait for all broadcasts to complete
            for handle in broadcast_handles:
                handle.wait()
    
    def allreduce_gradients(self, model):
        """All-reduce gradients across all processes"""
        if self.is_distributed:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.rank_size
    
    def average_value(self, value, name=None):
        """Average a value across all processes (placeholder for compatibility)"""
        # For multi_gpu=False case, we don't need actual averaging since DDP handles it
        # This method exists for compatibility with existing code
        return value
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()