#!/bin/bash
# Rollout Data Saving Script - Normal Settings
# Saves successful rollout data (joint angles, root pos/rot) for each motion
# Uses multiple parallel environments to increase chance of success for difficult motions

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mulci_isaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

EXP_NAME="g-cluster_0_1_2_4_n_clusters_10-ours-v14-dagger"
NUM_ENVS=1024  # High parallelism to increase success chance for difficult motions

echo "=========================================="
echo "Rollout Data Saving - Normal Settings"
echo "=========================================="
echo "Experiment: $EXP_NAME"
echo "Number of environments: $NUM_ENVS"
echo "Output directory: output/HumanoidIm/$EXP_NAME/rollout_data/"
echo ""

python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$EXP_NAME \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=$NUM_ENVS \
       ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 \
       eval_subdir=rollout_normal \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_1_2_4_n_clusters_10.pkl \
       ++env.terminationDistance=0.5 \
       +save_rollout=True

echo ""
echo "=========================================="
echo "Rollout data saving complete (Normal settings)"
echo "Check output/HumanoidIm/$EXP_NAME/rollout_data/ for saved files"
echo "=========================================="
