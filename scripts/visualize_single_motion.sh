#!/bin/sh
# Visualize a single motion (G058T010A035R003)
conda activate mulci_isaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Step 1: Prepare the single-motion pkl
python scripts/prepare_single_motion_pkl.py \
    --src sample_data/interx_processed_fixed_v9_cluster_ids_0_1_2_4_n_clusters_10.pkl \
    --motion_id G058T010A035R003 \
    --out sample_data/interx_G058T010A035R003.pkl

# Step 2: Run visualization (num_envs=2: 1 caregiver + 1 recipient)
EXP_NAME="g-cluster_0_1_2_4_n_clusters_10-cvpr2026-dagger-9k-v2"
python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$EXP_NAME \
       test=True im_eval=False headless=False epoch=-1 env.num_envs=2 \
       ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 \
       eval_subdir=rollout_normal \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_G058T010A035R003.pkl \
       ++env.terminationDistance=0.5 \
       +save_rollout=False
