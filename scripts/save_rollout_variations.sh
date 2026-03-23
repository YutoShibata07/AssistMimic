#!/bin/bash
# Rollout Data Saving Script - Two Variation Tests
# Test 1: recipient_mass_scale=0.84 (weakness_scale=1.0)
# Test 2: recipient_weakness_scale=0.25 (mass_scale=0.7)

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mulci_isaac
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

EXP_NAME="g-cluster_0_1_2_4_n_clusters_10-ours-v14-dagger"
NUM_ENVS=1024

echo "=========================================="
echo "Rollout Data Saving - Variation Tests"
echo "=========================================="

# ===========================================
# Test 1: Mass Scale = 0.84
# ===========================================
echo ""
echo "=========================================="
echo "Test 1: Mass Scale Only (mass=0.84)"
echo "=========================================="
echo "Recipient mass scale: 0.84"
echo "Recipient weakness scale: 1.0 (default)"
echo "Output directory: rollout_mass_0.84"
echo ""

python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$EXP_NAME \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=$NUM_ENVS \
       ++env.recipient_mass_scale=0.84 \
       ++env.recipient_weakness_scale=1.0 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 \
       eval_subdir=rollout_mass_0.84 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_1_2_4_n_clusters_10.pkl \
       ++env.terminationDistance=0.5 \
       +save_rollout=True

echo ""
echo "Test 1 complete (mass_scale=0.84)"
echo ""

# ===========================================
# Test 2: Weakness Scale = 0.25
# ===========================================
echo ""
echo "=========================================="
echo "Test 2: Weakness Scale Only (weakness=0.25)"
echo "=========================================="
echo "Recipient mass scale: 0.7 (default)"
echo "Recipient weakness scale: 0.25"
echo "Output directory: rollout_weakness_0.25"
echo ""

python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$EXP_NAME \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=$NUM_ENVS \
       ++env.recipient_mass_scale=0.7 \
       ++env.recipient_weakness_scale=0.25 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 \
       eval_subdir=rollout_weakness_0.25 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_1_2_4_n_clusters_10.pkl \
       ++env.terminationDistance=0.5 \
       +save_rollout=True

echo ""
echo "Test 2 complete (weakness_scale=0.25)"
echo ""

echo "=========================================="
echo "All variation tests complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - rollout_mass_0.84/"
echo "  - rollout_weakness_0.25/"
echo "=========================================="
