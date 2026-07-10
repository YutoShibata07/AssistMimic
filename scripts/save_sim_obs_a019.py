"""Run IsaacGym simulation on A019 and save per-frame observations.

Runs phc/run_hydra.py with hooks to save obs_buf at each step.

Usage:
    cd /home/shibatie/SSD/AssistMimic
    conda activate mulci_isaac
    python scripts/save_sim_obs_a019.py
"""
import subprocess
import sys
import os

# Modify humanoid_im_interx_helpup.py to save observations
# We'll use environment variable to trigger saving

def main():
    model = "g-cluster_0_1_2_4_n_clusters_10-cvpr2026-dagger-9k-v2"
    test_data = "sample_data/interx_processed_fixed_v6_A019_G001_G002.pkl"

    # Run with 2 envs (1 pair: CG + RC)
    hydra_args = [
        "env=env_im_interx_helpup",
        "learning=im_simpleliftup_mlp",
        "robot=smplx_humanoid",
        "robot.freeze_hand=False",
        "robot.box_body=False",
        f"exp_name={model}",
        "im_eval=True", "test=True", "headless=True", "no_log=True", "epoch=-1",
        "env.num_envs=2",  # 1 pair = 2 envs (CG + RC)
        f"++env.interx_data_path={test_data}",
        "++env.recipient_mass_scale=0.7",
        "++env.kp_scale=1.0",
        "++env.kd_scale=1.0",
        "++env.save_debug_obs=True",  # Custom flag to trigger obs saving
        "++env.save_debug_obs_path=/home/shibatie/SSD/Eden/examples/assistmimic/isaacgym_sim_obs.pkl",
        "++env.save_debug_obs_frames=30",
    ]

    cmd = f"python phc/run_hydra.py {' '.join(hydra_args)}"
    print(f"Running: {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    main()
