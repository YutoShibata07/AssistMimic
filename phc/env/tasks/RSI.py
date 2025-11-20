"""
Reference State Initialization (RSI) Module
Provides trajectory buffer functionality for storing and sampling successful episodes.
"""

import torch
import numpy as np

class TrajectoryBuffer:
    """Buffer for storing successful trajectories for Reference State Initialization"""
    
    def __init__(self, buffer_size=10, reward_threshold=5.0, device='cuda'):
        self.buffer_size = buffer_size
        self.reward_threshold = reward_threshold
        self.device = device
        
        self.effective_size = buffer_size  # 9 slots for successful trajectories
        
        # Motion-level storage: motion_uniq_id -> list of trajectory entries
        self.motion_trajectories = {}
        self.motion_write_indices = {}  # Track FIFO write position for each motion
        
        print(f"TrajectoryBuffer initialized: {buffer_size} slots (1 reserved for reference)")
        
    def store_trajectory(self, motion_uniq_id, trajectory_data, episode_reward):
        """Store a successful trajectory with FIFO replacement"""

        if episode_reward < self.reward_threshold:
            return False  # Don't store trajectories below threshold
            
        if motion_uniq_id not in self.motion_trajectories:
            self.motion_trajectories[motion_uniq_id] = []
            self.motion_write_indices[motion_uniq_id] = 0
        
        # Don't clone again - tensors are already detached and cloned in _capture_trajectory_step
        # This prevents double memory usage
        trajectory_entry = {
            'timestamps': trajectory_data['timestamps'],  # Already a fresh tensor from torch.tensor()
            'caregiver_states': trajectory_data['caregiver_states'] if trajectory_data['caregiver_states'] and any(len(v) > 0 for v in trajectory_data['caregiver_states'].values()) else {},
            'recipient_states': trajectory_data['recipient_states'] if trajectory_data['recipient_states'] and any(len(v) > 0 for v in trajectory_data['recipient_states'].values()) else {},
            'episode_reward': float(episode_reward)
        }
        
        # FIFO replacement
        if len(self.motion_trajectories[motion_uniq_id]) < self.effective_size:
            self.motion_trajectories[motion_uniq_id].append(trajectory_entry)
            print(f"Stored trajectory for {motion_uniq_id}. Reward: {episode_reward:.2f}. Buffer size: {len(self.motion_trajectories[motion_uniq_id])}/{self.effective_size}")
        else:
            # Replace oldest entry
            write_idx = self.motion_write_indices[motion_uniq_id]
            self.motion_trajectories[motion_uniq_id][write_idx] = trajectory_entry
            self.motion_write_indices[motion_uniq_id] = (write_idx + 1) % self.effective_size
        return True
        
    def has_trajectories(self, motion_uniq_id):
        """Check if buffer has any trajectories for given motion_uniq_id"""
        return motion_uniq_id in self.motion_trajectories and len(self.motion_trajectories[motion_uniq_id]) > 0
        
    def sample_random_state(self, motion_uniq_id, env_id):
        """Randomly sample initialization state from buffer for given motion and environment"""
        
        if not self.has_trajectories(motion_uniq_id):
            return None
            
        # Randomly select a trajectory
        trajectories = self.motion_trajectories[motion_uniq_id]
        traj_idx = torch.randint(0, len(trajectories), (1,)).item()
        trajectory = trajectories[traj_idx]
        
        # Randomly select a frame index from the trajectory
        frame_indices = trajectory['timestamps']
        frame_idx = torch.randint(0, len(frame_indices), (1,)).item()
        selected_frame = frame_indices[frame_idx]
        
        # Determine if this env_id is caregiver or recipient
        is_recipient = (env_id % 2 == 1)
        role = "recipient" if is_recipient else "caregiver"
        
        # Select appropriate states, fallback to caregiver if recipient states are empty
        if is_recipient:
            states = trajectory['recipient_states']
        elif not is_recipient:
            states = trajectory['caregiver_states']
        else:
            print(trajectory)
            error_msg = f"No valid states found for env_id {env_id}"
            raise ValueError(error_msg)
        
        # Extract state at selected frame index
        sampled_state = {
            'motion_time': selected_frame,
            'root_pos': states['root_pos'][frame_idx],
            'root_rot': states['root_rot'][frame_idx], 
            'dof_pos': states['dof_pos'][frame_idx],
            'root_vel': states['root_vel'][frame_idx],
            'root_ang_vel': states['root_ang_vel'][frame_idx],
            'dof_vel': states['dof_vel'][frame_idx]
        }
        
        return sampled_state
        
    def get_buffer_stats(self):
        """Get buffer statistics for logging"""
        total_trajectories = sum(len(trajs) for trajs in self.motion_trajectories.values())
        motion_count = len(self.motion_trajectories)
        avg_reward = 0.0
        
        if total_trajectories > 0:
            total_reward = sum(
                sum(traj['episode_reward'] for traj in trajs) 
                for trajs in self.motion_trajectories.values()
            )
            avg_reward = total_reward / total_trajectories
            
        return {
            'total_trajectories': total_trajectories,
            'motion_count': motion_count,
            'avg_reward': avg_reward,
            'buffer_utilization': total_trajectories / (len(self.motion_trajectories) * self.effective_size) if motion_count > 0 else 0.0
        }


class RSIMixin:
    """Mixin class providing Reference State Initialization functionality"""

    def _init_rsi(self, cfg):
        """Initialize RSI components"""
        # Initialize Reference State Initialization buffer
        self.trajectory_buffer = TrajectoryBuffer(
            buffer_size=cfg["env"].get("trajectory_buffer_size", 10),
            reward_threshold=cfg["env"].get("trajectory_reward_threshold", 300),
            device=self.device
        )
        print(f"Trajectory buffer initialized: size={self.trajectory_buffer.buffer_size}, threshold={self.trajectory_buffer.reward_threshold}")

        # Track trajectory data during episodes for buffer storage
        self.episode_trajectory_data = {}  # env_id -> trajectory data
        self.episode_start_times = torch.zeros(self.num_envs, device=self.device)

        # Map env_id to motion unique ID (G009T006A035R010, etc.)
        self.env_to_motion_uniq_id = {}  # env_id -> motion_unique_id

        # Select random subset of environment pairs for trajectory tracking
        # This reduces memory usage while maintaining diversity
        num_pairs = self.num_envs // 2
        num_tracking_pairs = cfg["env"].get("trajectory_tracking_pairs", min(50, num_pairs))

        # Randomly select pair indices and convert to env_ids (caregiver, recipient)
        selected_pair_indices = torch.randperm(num_pairs)[:num_tracking_pairs]
        tracking_env_ids = []
        for pair_idx in selected_pair_indices:
            caregiver_id = pair_idx * 2
            recipient_id = pair_idx * 2 + 1
            tracking_env_ids.extend([caregiver_id.item(), recipient_id.item()])

        self.trajectory_tracking_env_ids = set(tracking_env_ids)
        print(f"Trajectory tracking: {num_tracking_pairs} pairs ({len(self.trajectory_tracking_env_ids)} envs) out of {num_pairs} pairs ({self.num_envs} envs total)")
        print(f"Memory reduction: {(1 - len(self.trajectory_tracking_env_ids)/self.num_envs)*100:.1f}%")
    
    def _update_env_motion_mapping(self, env_ids):
        """Update the mapping between env_id and motion_unique_id at episode start"""
        # Access parent class motion library to get current motion keys
        if hasattr(self, '_motion_lib') and hasattr(self._motion_lib, 'curr_motion_keys'):
            curr_motion_keys = self._motion_lib.curr_motion_keys
            
            for env_id in env_ids:
                env_id_val = env_id.item() if torch.is_tensor(env_id) else env_id
                
                # Get motion key for this environment
                motion_key = curr_motion_keys[env_id_val]
                
                # Extract unique motion ID (remove '_caregiver' or '_recipient' suffix)
                if '_caregiver' in motion_key:
                    motion_uniq_id = motion_key.replace('_caregiver', '')
                elif '_recipient' in motion_key:
                    motion_uniq_id = motion_key.replace('_recipient', '')
                else:
                    error_msg = f"Role not found in motion key: {motion_key}"
                    raise ValueError(error_msg)
                
                # Store mapping
                self.env_to_motion_uniq_id[env_id_val] = motion_uniq_id
    
    def _get_motion_uniq_id_for_env(self, env_id):
        """Get the motion_unique_id for a given environment"""
        
        # Use stored mapping to get motion unique ID (e.g., G009T006A035R010)
        if hasattr(self, 'env_to_motion_uniq_id') and env_id in self.env_to_motion_uniq_id:
            motion_id = self.env_to_motion_uniq_id[env_id]
            return motion_id
        else:
            # Try to update mapping if motion library is available
            if hasattr(self, '_motion_lib') and hasattr(self._motion_lib, 'curr_motion_keys'):
                self._update_env_motion_mapping([env_id])
                if hasattr(self, 'env_to_motion_uniq_id') and env_id in self.env_to_motion_uniq_id:
                    motion_id = self.env_to_motion_uniq_id[env_id]
                    return motion_id
            
            error_msg = f"No motion_unique_id found for env_id: {env_id}. Available mappings: {getattr(self, 'env_to_motion_uniq_id', {})}"
            if env_id < 5:
                print(f"[RSI ERROR] {error_msg}")
            raise ValueError(error_msg)
    
    def _start_trajectory_tracking(self, env_ids):
        """Initialize trajectory tracking for new episodes (only for selected envs)"""
        for env_id in env_ids:
            env_id_val = env_id.item() if torch.is_tensor(env_id) else env_id

            # Only track selected environments
            if env_id_val not in self.trajectory_tracking_env_ids:
                continue

            # Initialize trajectory data storage for this environment
            self.episode_trajectory_data[env_id_val] = {
                'timestamps': [],
                'caregiver_states': {
                    'root_pos': [],
                    'root_rot': [],
                    'dof_pos': [],
                    'root_vel': [],
                    'root_ang_vel': [],
                    'dof_vel': []
                },
                'recipient_states': {
                    'root_pos': [],
                    'root_rot': [],
                    'dof_pos': [],
                    'root_vel': [],
                    'root_ang_vel': [],
                    'dof_vel': []
                }
            }

            # Record episode start time
            self.episode_start_times[env_id_val] = 0  # Will be set during first physics step
    
    def _capture_trajectory_step(self, env_ids):
        """Capture trajectory data during episode (only for selected envs)"""
        # Use no_grad to prevent gradient tracking and memory leaks
        with torch.no_grad():
            # Filter to only tracking environments
            if torch.is_tensor(env_ids):
                tracking_mask = torch.tensor([env_id.item() in self.trajectory_tracking_env_ids
                                             for env_id in env_ids], device=env_ids.device)
                filtered_env_ids = env_ids[tracking_mask]
            else:
                filtered_env_ids = [env_id for env_id in env_ids
                                   if env_id in self.trajectory_tracking_env_ids]

            # Early return if no tracking environments
            if len(filtered_env_ids) == 0:
                return

            # Get current states for tracking environments only
            root_pos = self._rigid_body_pos[filtered_env_ids, 0, :3]  # Root position
            root_rot = self._rigid_body_rot[filtered_env_ids, 0, :]   # Root rotation
            dof_pos = self._dof_pos[filtered_env_ids]                 # DOF positions
            root_vel = self._rigid_body_vel[filtered_env_ids, 0, :3]  # Root velocity
            root_ang_vel = self._rigid_body_ang_vel[filtered_env_ids, 0, :3]  # Root angular velocity
            dof_vel = self._dof_vel[filtered_env_ids]                 # DOF velocities

            # Get episode_length from config (default 300 if not set)
            episode_length = getattr(self, 'max_episode_length', 300)

            # Capture states for each environment
            for i, env_id in enumerate(filtered_env_ids):
                env_id_val = env_id.item() if torch.is_tensor(env_id) else env_id

                # Skip if trajectory tracking not initialized for this environment
                if env_id_val not in self.episode_trajectory_data:
                    continue

                # Current frame index - use item() for scalar to avoid tensor accumulation
                current_frame = self.progress_buf[env_id_val].item()

                # Determine if this is caregiver or recipient
                is_recipient = (env_id_val % 2 == 1)

                # Store trajectory data
                trajectory_data = self.episode_trajectory_data[env_id_val]

                # FIFO: If trajectory length reaches episode_length, remove oldest frame before adding new one
                # This prevents unbounded memory growth while keeping the most recent episode_length frames
                current_trajectory_length = len(trajectory_data['timestamps'])
                if current_trajectory_length >= episode_length:
                    # Remove oldest frame (index 0) from all arrays
                    trajectory_data['timestamps'].pop(0)

                    if is_recipient:
                        # Remove oldest recipient data
                        for key in trajectory_data['recipient_states'].keys():
                            if len(trajectory_data['recipient_states'][key]) > 0:
                                trajectory_data['recipient_states'][key].pop(0)
                    else:
                        # Remove oldest caregiver data
                        for key in trajectory_data['caregiver_states'].keys():
                            if len(trajectory_data['caregiver_states'][key]) > 0:
                                trajectory_data['caregiver_states'][key].pop(0)

                # Add new frame data
                trajectory_data['timestamps'].append(current_frame)

                if is_recipient:
                    # Store recipient states with .detach().clone() to break computational graph
                    trajectory_data['recipient_states']['root_pos'].append(root_pos[i].detach().clone())
                    trajectory_data['recipient_states']['root_rot'].append(root_rot[i].detach().clone())
                    trajectory_data['recipient_states']['dof_pos'].append(dof_pos[i].detach().clone())
                    trajectory_data['recipient_states']['root_vel'].append(root_vel[i].detach().clone())
                    trajectory_data['recipient_states']['root_ang_vel'].append(root_ang_vel[i].detach().clone())
                    trajectory_data['recipient_states']['dof_vel'].append(dof_vel[i].detach().clone())
                else:
                    # Store caregiver states with .detach().clone() to break computational graph
                    trajectory_data['caregiver_states']['root_pos'].append(root_pos[i].detach().clone())
                    trajectory_data['caregiver_states']['root_rot'].append(root_rot[i].detach().clone())
                    trajectory_data['caregiver_states']['dof_pos'].append(dof_pos[i].detach().clone())
                    trajectory_data['caregiver_states']['root_vel'].append(root_vel[i].detach().clone())
                    trajectory_data['caregiver_states']['root_ang_vel'].append(root_ang_vel[i].detach().clone())
                    trajectory_data['caregiver_states']['dof_vel'].append(dof_vel[i].detach().clone())
    
    def _save_successful_trajectories(self, env_ids, episode_rewards):
        """Save successful episode trajectories to buffer by merging caregiver/recipient pairs

        Args:
            env_ids: Tensor or list of environment IDs that finished episodes
                    Only contains caregiver IDs (even numbers: 0, 2, 4...) from done_indices
            episode_rewards: Full rewards tensor [num_envs] with rewards for ALL environments
        """
        stored_count = 0

        # Convert to list if tensor
        if torch.is_tensor(env_ids):
            env_ids = env_ids.tolist() if env_ids.dim() > 0 else [env_ids.item()]

        # Convert episode_rewards to tensor if it isn't already
        if not torch.is_tensor(episode_rewards):
            episode_rewards = torch.tensor(episode_rewards, device=self.device)

        # Process each env_id (these are caregiver IDs from done_indices)
        for env_id in env_ids:
            caregiver_id = env_id if env_id % 2 == 0 else env_id - 1
            recipient_id = caregiver_id + 1

            # Skip if no trajectory data for either environment
            if caregiver_id not in self.episode_trajectory_data or recipient_id not in self.episode_trajectory_data:
                # Clean up any partial trajectory data
                if caregiver_id in self.episode_trajectory_data:
                    del self.episode_trajectory_data[caregiver_id]
                if recipient_id in self.episode_trajectory_data:
                    del self.episode_trajectory_data[recipient_id]
                continue

            # Get rewards for both caregiver and recipient from full rewards tensor
            caregiver_reward = episode_rewards[caregiver_id].item()
            recipient_reward = episode_rewards[recipient_id].item()
            max_reward = max(caregiver_reward, recipient_reward)

            # Get motion_unique_id (should be same for both)
            motion_uniq_id = self._get_motion_uniq_id_for_env(caregiver_id)

            # IMPORTANT: Always clean up trajectory data, regardless of reward threshold
            # to prevent memory leak from accumulated failed episodes
            if max_reward < self.trajectory_buffer.reward_threshold:
                # Clean up failed episode data before continuing
                del self.episode_trajectory_data[caregiver_id]
                del self.episode_trajectory_data[recipient_id]
                continue


            # Merge trajectory data from both environments
            caregiver_data = self.episode_trajectory_data[caregiver_id]
            recipient_data = self.episode_trajectory_data[recipient_id]

            # Use timestamps from caregiver (they should be similar)
            # Convert Python list to tensor - timestamps are already scalars (not tensors)
            processed_data = {
                'timestamps': torch.tensor(caregiver_data['timestamps'], device=self.device),
                'caregiver_states': {key: torch.stack(vals) for key, vals in caregiver_data['caregiver_states'].items() if len(vals) > 0},
                'recipient_states': {key: torch.stack(vals) for key, vals in recipient_data['recipient_states'].items() if len(vals) > 0}
            }

            # Store merged trajectory with max reward
            if self.trajectory_buffer.store_trajectory(motion_uniq_id, processed_data, max_reward):
                stored_count += 1

            # Clean up trajectory data for both environments
            del self.episode_trajectory_data[caregiver_id]
            del self.episode_trajectory_data[recipient_id]
            
        if stored_count > 0:
            stats = self.trajectory_buffer.get_buffer_stats()
            import numpy as np
            if np.random.rand() < 0.01:
                print(f"Stored {stored_count} successful trajectories. Buffer: {stats['total_trajectories']} trajs, avg_reward={stats['avg_reward']:.1f}")
    
    def _sample_from_trajectory_buffer(self, env_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        """Sample states from trajectory buffer for initialization.
        Returns a torch.bool mask of shape [len(env_ids)] indicating per-env buffer usage,
        or None if no trajectory_buffer is available.
        """
        if not hasattr(self, 'trajectory_buffer'):
            return None
            
        buffer_usage_ratio = 0.5  # 50% chance to use buffer, 50% reference motion
        used_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        # Process pairs: env_ids come as caregiver/recipient pairs
        for i in range(0, len(env_ids), 2):
            if i+1 >= len(env_ids):
                break  # Skip if odd number of environments
                
            caregiver_id = env_ids[i].item() if torch.is_tensor(env_ids[i]) else env_ids[i]
            recipient_id = env_ids[i+1].item() if torch.is_tensor(env_ids[i+1]) else env_ids[i+1]
            
            motion_uniq_id = self._get_motion_uniq_id_for_env(caregiver_id)
            
            # Decide whether to use buffer or reference motion for this PAIR
            has_trajs = self.trajectory_buffer.has_trajectories(motion_uniq_id)
            use_buffer = (torch.rand(1).item() < buffer_usage_ratio and has_trajs)

            if use_buffer:
                # Sample SAME trajectory and frame for both caregiver and recipient
                trajectories = self.trajectory_buffer.motion_trajectories[motion_uniq_id]
                traj_idx = torch.randint(0, len(trajectories), (1,)).item()
                trajectory = trajectories[traj_idx]
                
                # Randomly select a frame index from the trajectory
                frame_indices = trajectory['timestamps']
                frame_idx = torch.randint(0, len(frame_indices), (1,)).item()
                selected_frame = frame_indices[frame_idx]
                
                if caregiver_id < 5:
                    print(f"[RSI DEBUG] Environments {caregiver_id} and {recipient_id} are initialized from the buffer (motion_uniq_id: {motion_uniq_id}, traj_idx: {traj_idx}, frame_idx: {frame_idx}, selected_frame: {selected_frame})")
    
                caregiver_states = trajectory['caregiver_states']
                motion_times[i] = selected_frame
                root_pos[i] = caregiver_states['root_pos'][frame_idx]
                root_rot[i] = caregiver_states['root_rot'][frame_idx]
                dof_pos[i] = caregiver_states['dof_pos'][frame_idx]
                root_vel[i] = caregiver_states['root_vel'][frame_idx]
                root_ang_vel[i] = caregiver_states['root_ang_vel'][frame_idx]
                dof_vel[i] = caregiver_states['dof_vel'][frame_idx]
                used_mask[i] = True
                    
                recipient_states = trajectory['recipient_states']
                motion_times[i+1] = selected_frame
                root_pos[i+1] = recipient_states['root_pos'][frame_idx]
                root_rot[i+1] = recipient_states['root_rot'][frame_idx]
                dof_pos[i+1] = recipient_states['dof_pos'][frame_idx]
                root_vel[i+1] = recipient_states['root_vel'][frame_idx]
                root_ang_vel[i+1] = recipient_states['root_ang_vel'][frame_idx]
                dof_vel[i+1] = recipient_states['dof_vel'][frame_idx]
                used_mask[i+1] = True
                    
                
                # Check distance between caregiver and recipient
                if used_mask[i] and used_mask[i+1]:
                    distance = torch.norm(root_pos[i] - root_pos[i+1])
                    if distance < 5.0:
                        error_msg = f"Distance between caregiver and recipient is less than 5.0m: {distance:.2f}m"
                        raise ValueError(error_msg)
            
        return used_mask