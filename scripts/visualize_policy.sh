#!/bin/sh
# HHI-Assist Evaluation Script
# Runs evaluation under various conditions for the HHI-assist task
conda activate mulci_isaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EXP_NAME="g-cluster_0_1_2_4_n_clusters_10-ours-v14-dagger"

python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$EXP_NAME \
       test=True im_eval=True headless=False epoch=-1 env.num_envs=2 ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_1_2_4_n_clusters_10.pkl \
       ++env.terminationDistance=0.5

