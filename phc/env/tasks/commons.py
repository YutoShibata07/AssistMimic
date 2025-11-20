"""
Common utilities and constants for humanoid interaction tasks.

This module contains shared functions and constants that can be reused
across different task implementations.
"""

import torch
import os

# ==================== Body Name Constants ====================

SMPL_BODY_NAMES = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
    'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
    'Torso', 'Spine', 'Chest', 'Neck', 'Head',
    'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
]

LEFT_FINGER_BODIES = [
    'L_Index1', 'L_Index2', 'L_Index3',
    'L_Middle1', 'L_Middle2', 'L_Middle3',
    'L_Ring1', 'L_Ring2', 'L_Ring3',
    'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
    'L_Thumb1', 'L_Thumb2', 'L_Thumb3'
]

RIGHT_FINGER_BODIES = [
    'R_Index1', 'R_Index2', 'R_Index3',
    'R_Middle1', 'R_Middle2', 'R_Middle3',
    'R_Ring1', 'R_Ring2', 'R_Ring3',
    'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
    'R_Thumb1', 'R_Thumb2', 'R_Thumb3'
]

FORCE_BODY_NAMES = [
    'L_Wrist', 'L_Index3', 'L_Middle3', 'L_Ring3', 'L_Pinky3', 'L_Thumb3',
    'R_Wrist', 'R_Index3', 'R_Middle3', 'R_Ring3', 'R_Pinky3', 'R_Thumb3',
    'L_Elbow', 'R_Elbow'
]

FINGER_WRIST_NAMES = ['L_Wrist', 'R_Wrist']

# ==================== Observation Size Constants ====================

HAND_FORCE_OBS_SIZE = 42  # 14 bodies * 3D force vector

# ==================== Pure Mathematical Functions ====================

def normalize_quaternion_batch(quat_batch):
    """Normalize batch of quaternions and handle edge cases.

    This function normalizes a batch of quaternions to unit quaternions,
    handling edge cases where quaternions are near-zero or exactly zero
    by replacing them with identity quaternions.

    Args:
        quat_batch (torch.Tensor): [N, 4] tensor of quaternions in (x, y, z, w) format

    Returns:
        torch.Tensor: [N, 4] tensor of normalized quaternions

    Example:
        >>> quats = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
        >>> normalized = normalize_quaternion_batch(quats)
    """
    # Calculate norms for each quaternion
    quat_norms = torch.norm(quat_batch, dim=-1, keepdim=True)  # [N, 1]

    # Identify quaternions that are too small (near-zero)
    small_quat_mask = (quat_norms < 1e-6).squeeze(-1)  # [N]

    # Identity quaternion (no rotation)
    identity_quat = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        device=quat_batch.device,
        dtype=quat_batch.dtype
    )

    # Safely normalize quaternions (avoid division by zero)
    normalized_quats = quat_batch / torch.clamp(quat_norms, min=1e-8)

    # Replace near-zero quaternions with identity quaternion
    normalized_quats[small_quat_mask] = identity_quat

    return normalized_quats

# ==================== Partner Interaction Utilities ====================

def get_partner_env_ids(env_ids):
    """Calculate partner environment IDs for paired environments.

    In paired humanoid interaction tasks, environments are arranged in pairs
    (caregiver-recipient or agent-agent). This function calculates the partner's
    environment ID using the pairing rule:
    - Even env_id (0, 2, 4, ...) -> partner is env_id + 1
    - Odd env_id (1, 3, 5, ...) -> partner is env_id - 1

    Args:
        env_ids (torch.Tensor): Tensor of environment IDs (can be any shape)

    Returns:
        torch.Tensor: Tensor of partner environment IDs (same shape as input)

    Example:
        >>> env_ids = torch.tensor([0, 1, 2, 3, 4, 5])
        >>> partners = get_partner_env_ids(env_ids)
        >>> print(partners)  # Output: tensor([1, 0, 3, 2, 5, 4])
    """
    partner_env_ids = torch.where(env_ids % 2 == 0, env_ids + 1, env_ids - 1)
    return partner_env_ids

def get_pair_offset_value(env_spacing=5.0):
    """Calculate pair offset value from environment spacing.

    In paired environments, the physical spacing between paired humanoids
    is typically double the base environment spacing.

    Args:
        env_spacing (float): Base spacing between environments (default: 5.0)

    Returns:
        float: Pair offset value (env_spacing * 2)

    Example:
        >>> offset = get_pair_offset_value(5.0)
        >>> print(offset)  # Output: 10.0
    """
    return env_spacing * 2

# ==================== Data Conversion Utilities ====================

def get_motion_filename_from_data(sample_data, env_id):
    """Get motion filename from stored trajectory/torque data.

    Generates a formatted filename string from motion data, useful for
    saving or loading motion-specific information.

    Args:
        sample_data (dict): Dictionary containing motion data with keys:
            - 'motion_key' (str or int): Motion key identifier, or
            - 'motion_id' (int): Motion ID number
        env_id (torch.Tensor or int): Environment ID (can be tensor or scalar)

    Returns:
        str: Formatted motion filename string in the format:
            - "env_{id}_{motion_filename}" if motion_key is a string path
            - "env_{id}_motion_{motion_id}" if motion_key/motion_id is an int
            - "env_{id}_motion" as fallback

    Example:
        >>> data = {'motion_key': 'walk_forward.pkl'}
        >>> filename = get_motion_filename_from_data(data, torch.tensor(5))
        >>> print(filename)  # Output: "env_5_walk_forward"
    """
    # Handle tensor or scalar env_id
    if hasattr(env_id, 'item'):
        env_id_val = env_id.item()
    else:
        env_id_val = env_id

    if sample_data.get('motion_key') is not None:
        motion_key = sample_data['motion_key']
        if isinstance(motion_key, str):
            # Extract filename without extension from path
            motion_filename = os.path.splitext(os.path.basename(motion_key))[0]
            return f"env_{env_id_val}_{motion_filename}"
        else:
            # Motion key is an integer
            return f"env_{env_id_val}_motion_{motion_key}"
    elif sample_data.get('motion_id') is not None:
        motion_id = sample_data['motion_id']
        return f"env_{env_id_val}_motion_{motion_id}"
    else:
        # Fallback to env_id only
        return f"env_{env_id_val}_motion"
