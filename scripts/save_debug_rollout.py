"""Save debug data from AssistMimic rollout for comparison with Eden.

Run with: conda activate mulci_isaac
    cd /home/shibatie/SSD/AssistMimic
    python scripts/save_debug_rollout.py

Saves per-frame data: body_pos, body_rot, observations (raw + normalized), actions, pd_targets
"""
import sys
import os
import torch
import numpy as np
import pickle

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from phc.utils.running_mean_std import RunningMeanStd
    import torch.nn as nn

    device = "cpu"
    ckpt_path = "/home/shibatie/SSD/Eden/examples/assistmimic/checkpoints/Humanoid.pth"
    motion_path = "/home/shibatie/SSD/Eden/examples/assistmimic/motions/interx_processed_fixed_v6_A019_G001_G002.pkl"

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load running_mean_std
    rms_data = ckpt['running_mean_std']
    obs_size = int(rms_data['running_mean'].shape[0])
    rms = RunningMeanStd((obs_size,))
    rms.running_mean.copy_(rms_data['running_mean'])
    rms.running_var.copy_(rms_data['running_var'])
    rms.count.copy_(rms_data['count'])
    rms.eval()  # Don't update stats
    print(f"RMS: size={obs_size}, mean range=[{rms.running_mean[:2026].min():.4f}, {rms.running_mean[:2026].max():.4f}]")

    # Build PNN actor
    model_state = ckpt['model']
    layers = []
    for i in range(0, 13, 2):
        w = model_state[f'a2c_network.pnn.actors.0.{i}.weight']
        b = model_state[f'a2c_network.pnn.actors.0.{i}.bias']
        linear = nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(b)
        layers.append(linear)
        if i < 12:
            layers.append(nn.SiLU())
    pnn = nn.Sequential(*layers)
    pnn.eval()

    # Build pd_action_offset and pd_action_scale (matching _build_pd_action_offset_scale)
    # Use PHC joint limits
    import math
    SMPLH_MUJOCO_NAMES = [
        "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
        "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
        "Torso", "Spine", "Chest", "Neck", "Head",
        "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
        "L_Index1", "L_Index2", "L_Index3",
        "L_Middle1", "L_Middle2", "L_Middle3",
        "L_Pinky1", "L_Pinky2", "L_Pinky3",
        "L_Ring1", "L_Ring2", "L_Ring3",
        "L_Thumb1", "L_Thumb2", "L_Thumb3",
        "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
        "R_Index1", "R_Index2", "R_Index3",
        "R_Middle1", "R_Middle2", "R_Middle3",
        "R_Pinky1", "R_Pinky2", "R_Pinky3",
        "R_Ring1", "R_Ring2", "R_Ring3",
        "R_Thumb1", "R_Thumb2", "R_Thumb3",
    ]

    # PHC joint limits (from SMPL humanoid XML)
    _PHC_JOINT_LIMITS_DEG = {
        "Hip": 90.0, "Ankle": 90.0, "Toe": 180.0, "Knee": 180.0,
        "Torso": 60.0, "Spine": 60.0, "Chest": 60.0,
        "Neck": 90.0, "Head": 90.0,
        "Thorax": 5.625, "Shoulder": 720.0, "Wrist": 180.0,
        "Index": 180.0, "Middle": 180.0, "Pinky": 180.0,
        "Ring": 180.0, "Thumb": 180.0, "Elbow": 180.0,
    }

    dof_joint_names = SMPLH_MUJOCO_NAMES[1:]  # 51 joints
    num_dofs = len(dof_joint_names) * 3  # 153

    lim_low = np.full(num_dofs, -math.pi)
    lim_high = np.full(num_dofs, math.pi)

    for j, jname in enumerate(dof_joint_names):
        for pattern, max_deg in _PHC_JOINT_LIMITS_DEG.items():
            if pattern in jname:
                max_rad = math.radians(max_deg)
                for axis in range(3):
                    lim_low[j * 3 + axis] = -max_rad
                    lim_high[j * 3 + axis] = max_rad
                break

    # PHC 3-DOF expansion
    for j in range(len(dof_joint_names)):
        off = j * 3
        curr = max(abs(lim_low[off:off+3]).max(), abs(lim_high[off:off+3]).max())
        curr = min(1.2 * curr, math.pi)
        lim_low[off:off+3] = -curr
        lim_high[off:off+3] = curr

    pd_action_offset = 0.5 * (lim_high + lim_low)
    pd_action_scale = 0.5 * (lim_high - lim_low)

    # Knee_y special
    for j, jname in enumerate(dof_joint_names):
        if "Knee" in jname:
            pd_action_offset[j * 3 + 1] = 0.0
            pd_action_scale[j * 3 + 1] = 5.0

    pd_action_offset = torch.tensor(pd_action_offset, dtype=torch.float32)
    pd_action_scale = torch.tensor(pd_action_scale, dtype=torch.float32)

    print(f"pd_action_scale L_Hip: {pd_action_scale[:3].tolist()}")
    print(f"pd_action_scale L_Knee: {pd_action_scale[3:6].tolist()}")
    print(f"pd_action_scale Torso: {pd_action_scale[24:27].tolist()}")

    # Load motion data
    with open(motion_path, 'rb') as f:
        motion_data = pickle.load(f)

    keys = list(motion_data.keys())
    cg_key = [k for k in keys if k.endswith("_caregiver")][0]
    cg_motion = motion_data[cg_key]

    # Load height fixes
    hf_path = motion_path.replace('.pkl', '_height_fixes.pkl')
    height_fixes = {}
    if os.path.exists(hf_path):
        with open(hf_path, 'rb') as f:
            height_fixes = pickle.load(f)

    motion_name = cg_key.replace("_caregiver", "")

    # Now: compute observation from motion data FK (same as Eden does)
    # This matches what the policy sees when initialized from motion data
    sys.path.insert(0, '/home/shibatie/SSD/Eden/examples/assistmimic/scripts')
    from policy_loader import (
        compute_full_observation,
        compute_body_positions_from_motion_mujoco,
        compute_body_velocities_mujoco,
    )

    cg_trans = torch.tensor(cg_motion["trans"], dtype=torch.float32)
    cg_pose_quat = torch.tensor(cg_motion["pose_quat_global"], dtype=torch.float32)  # XYZW

    # FK
    cg_body_pos = compute_body_positions_from_motion_mujoco(cg_trans, cg_pose_quat)

    # Height fix
    if motion_name in height_fixes:
        cg_fix, _ = height_fixes[motion_name]
        cg_offset = -cg_fix
    else:
        cg_offset = 0.0
    cg_body_pos[:, :, 2] += cg_offset
    cg_trans = cg_trans.clone()
    cg_trans[:, 2] += cg_offset

    # Body rotations XYZW -> WXYZ
    cg_body_rot = cg_pose_quat[..., [3, 0, 1, 2]]

    # Velocities
    fps = cg_motion.get("fps", 30)
    cg_vel = compute_body_velocities_mujoco(cg_body_pos, cg_body_rot, fps)

    num_frames = min(30, len(cg_trans) - 1)
    results = {
        "motion_name": motion_name,
        "num_frames": num_frames,
        "fps": fps,
        "pd_action_offset": pd_action_offset.numpy(),
        "pd_action_scale": pd_action_scale.numpy(),
        "frames": [],
    }

    for t in range(num_frames):
        cur_pos = cg_body_pos[t:t+1]
        cur_rot = cg_body_rot[t:t+1]
        cur_vel = cg_vel["body_lin_vel"][t:t+1]
        cur_ang = cg_vel["body_ang_vel"][t:t+1]

        ref_t = min(t + 1, len(cg_trans) - 1)
        ref_pos = cg_body_pos[ref_t:ref_t+1]
        ref_rot = cg_body_rot[ref_t:ref_t+1]
        ref_vel = cg_vel["body_lin_vel"][ref_t:ref_t+1]
        ref_ang = cg_vel["body_ang_vel"][ref_t:ref_t+1]

        # Raw observation (2026 dims)
        obs_raw = compute_full_observation(
            cur_pos, cur_rot, cur_vel, cur_ang,
            ref_pos, ref_rot, ref_vel, ref_ang,
            upright=False,
        )

        # Normalize with AssistMimic's RMS (pad to 3760 for RMS, then take first 2026)
        obs_padded = torch.zeros(1, obs_size, dtype=torch.float64)
        obs_padded[:, :2026] = obs_raw.double()
        obs_norm_full = rms(obs_padded).float()
        obs_norm_2026 = obs_norm_full[:, :2026]

        # Policy action
        with torch.no_grad():
            action = pnn(obs_norm_2026)

        # PD target
        action_clamped = action.clamp(-1, 1)
        pd_target = pd_action_offset + pd_action_scale * action_clamped.squeeze(0)

        frame_data = {
            "root_pos": cur_pos[0, 0].numpy().copy(),
            "root_rot_wxyz": cur_rot[0, 0].numpy().copy(),
            "obs_raw_first20": obs_raw[0, :20].numpy().copy(),
            "obs_norm_first20": obs_norm_2026[0, :20].numpy().copy(),
            "obs_raw_shape": obs_raw.shape,
            "action": action.squeeze(0).numpy().copy(),
            "action_clamped": action_clamped.squeeze(0).numpy().copy(),
            "pd_target": pd_target.numpy().copy(),
        }
        results["frames"].append(frame_data)

        if t < 5:
            print(f"Frame {t}:")
            print(f"  root_h={obs_raw[0, 0].item():.4f}")
            print(f"  obs_norm[:6]={obs_norm_2026[0, :6].tolist()}")
            print(f"  action range=[{action.min():.4f}, {action.max():.4f}], mean={action.mean():.4f}")
            print(f"  pd_target[:6]={pd_target[:6].tolist()}")

    save_path = "/home/shibatie/SSD/Eden/examples/assistmimic/assistmimic_debug_rollout.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved debug rollout: {save_path}")


if __name__ == "__main__":
    main()
