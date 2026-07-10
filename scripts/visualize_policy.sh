#!/bin/sh
# Generalist Policy Visualization Script (Inter-X)
# Rolls out the DAgger-distilled generalist policy on 30 diverse interaction
# clips with GUI rendering. Set headless=True for batch evaluation instead.
conda activate mulci_isaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EXP_NAME="g-cluster_0_1_2_4_n_clusters_10-cvpr2026-dagger-9k-v2"
mkdir -p output/HumanoidIm/${EXP_NAME}
cp checkpoints/${EXP_NAME}/Humanoid.pth output/HumanoidIm/${EXP_NAME}/Humanoid.pth
eval_file="sample_data/interx_processed_fixed_v9_cluster_ids_0_1_2_4_n_clusters_10.pkl"
# eval_file="sample_data/G009T006A035R010.pkl"
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$EXP_NAME \
       test=True im_eval=True headless=False epoch=-1 env.num_envs=2 \
       ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 \
       eval_subdir=rollout_normal \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=$eval_file \
       ++env.terminationDistance=0.5 \
       +save_rollout=False

