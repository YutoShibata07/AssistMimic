"""Save A019 reference data from AssistMimic for Eden comparison.

Uses the same motion data and FK as Eden, but with AssistMimic's RMS and
full network architecture to produce reference actions.

Run: conda activate mulci_isaac
     cd /home/shibatie/SSD/AssistMimic
     python scripts/save_a019_reference.py
"""
import sys, os, torch, torch.nn as nn, numpy as np, pickle, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/shibatie/SSD/Eden/examples/assistmimic/scripts')

from policy_loader import (
    compute_full_observation_3760,
    compute_body_positions_from_motion_mujoco,
    compute_body_velocities_mujoco,
    AssistMimicPolicy,
)


def main():
    device = "cpu"
    base = "/home/shibatie/SSD/Eden/examples/assistmimic"

    # Load the SAME policy that Eden uses
    policy = AssistMimicPolicy(f"{base}/checkpoints/Humanoid.pth", device=device)

    # Load motion
    with open(f"{base}/motions/interx_processed_fixed_v6_A019_G001_G002.pkl", 'rb') as f:
        data = pickle.load(f)
    with open(f"{base}/motions/interx_processed_fixed_v6_A019_G001_G002_height_fixes.pkl", 'rb') as f:
        hf = pickle.load(f)

    keys = list(data.keys())
    cg_key = [k for k in keys if k.endswith('_caregiver')][0]
    rc_key = cg_key.replace('_caregiver', '_recipient')
    mn = cg_key.replace('_caregiver', '')

    cg, rc = data[cg_key], data[rc_key]
    cg_trans = torch.tensor(cg['trans'], dtype=torch.float32)
    rc_trans = torch.tensor(rc['trans'], dtype=torch.float32)
    cg_quat = torch.tensor(cg['pose_quat_global'], dtype=torch.float32)
    rc_quat = torch.tensor(rc['pose_quat_global'], dtype=torch.float32)

    # FK
    cg_body_pos = compute_body_positions_from_motion_mujoco(cg_trans, cg_quat)
    rc_body_pos = compute_body_positions_from_motion_mujoco(rc_trans, rc_quat)

    # Height fix
    if mn in hf:
        cg_body_pos[:,:,2] += -hf[mn][0]
        rc_body_pos[:,:,2] += -hf[mn][1]
        cg_trans = cg_trans.clone(); cg_trans[:,2] += -hf[mn][0]
        rc_trans = rc_trans.clone(); rc_trans[:,2] += -hf[mn][1]

    cg_rot = cg_quat[..., [3,0,1,2]]
    rc_rot = rc_quat[..., [3,0,1,2]]

    fps = cg.get('fps', 30)
    cg_vel = compute_body_velocities_mujoco(cg_body_pos, cg_rot, fps)
    rc_vel = compute_body_velocities_mujoco(rc_body_pos, rc_rot, fps)

    num_frames = min(30, len(cg_trans) - 1)
    cg_prev = torch.zeros(1, 153)
    rc_prev = torch.zeros(1, 153)

    results = {"frames": [], "motion": mn}

    for t in range(num_frames):
        ref_t = min(t + 1, len(cg_trans) - 1)

        # CG observation (3760)
        cg_obs = compute_full_observation_3760(
            cg_body_pos[t:t+1], cg_rot[t:t+1],
            cg_vel['body_lin_vel'][t:t+1], cg_vel['body_ang_vel'][t:t+1],
            cg_body_pos[ref_t:ref_t+1], cg_rot[ref_t:ref_t+1],
            cg_vel['body_lin_vel'][ref_t:ref_t+1], cg_vel['body_ang_vel'][ref_t:ref_t+1],
            partner_body_pos=rc_body_pos[t:t+1],
            partner_body_vel=rc_vel['body_lin_vel'][t:t+1],
            partner_body_rot=rc_rot[t:t+1],
            prev_actions=cg_prev, role_label=0.0, upright=False,
        )

        # CG action (using correct network)
        cg_action = policy.get_action(cg_obs, role="caregiver")
        cg_action_c = cg_action.clamp(-1, 1)
        cg_prev = cg_action_c.detach()

        # Also normalize to compare
        cg_obs_norm = policy.normalize_obs(cg_obs)

        results["frames"].append({
            "cg_obs_norm_first50": cg_obs_norm[0, :50].numpy().copy(),
            "cg_obs_norm_partner_first20": cg_obs_norm[0, 2026:2046].numpy().copy(),
            "cg_action": cg_action[0].numpy().copy(),
            "cg_action_first10": cg_action[0, :10].numpy().copy(),
        })

        if t < 5:
            print(f"Frame {t}: CG action range=[{cg_action.min():.4f}, {cg_action.max():.4f}], "
                  f"mean={cg_action.mean():.4f}")
            print(f"  obs_norm[:6]={cg_obs_norm[0,:6].tolist()}")
            print(f"  action[:6]={cg_action[0,:6].tolist()}")

    save_path = f"{base}/a019_reference_full_arch.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
