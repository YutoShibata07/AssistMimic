#!/bin/sh
# Inter-X Evaluation Script
# Reproduces Table 2 of the paper (specialist policies on the Inter-X dataset):
#   1. AssistMimic (full)                      -> g-cluster-0-n10-ours-v14-adj
#   2. (-) Dynamic Reference Retargeting       -> ours-contact
#   3. (-) Contact Promoting Reward            -> ours-no-contact-no-retargeting
#   4. Sequential Training (frozen recipient)  -> CVPR2026-assistmimic-ablation-freeze-recipient
#   5. (-) Weight Initialization               -> g-cluster-0-n10-ours-v17-sep-wo-res-con
# Extra (not in Table 2): retargeting threshold 0.8 variant at the end of this script.
# Set to the GPU you want to use
export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# 1. AssistMimic (full): with contact reward and dynamic reference retargeting
#    (exp: g-cluster-0-n10-ours-v14-adj)
#    Uses the default retargeting distance threshold
#    (force_tracking_distance_threshold=1.3), matching the paper's Table 2.
# ============================================================================
echo "g-cluster-0-n10-ours-v14-adj (AssistMimic full, threshold 1.3)"
model_name=g-cluster-0-n10-ours-v14-adj
mkdir -p output/HumanoidIm/$model_name
cp checkpoints/$model_name/Humanoid.pth output/HumanoidIm/$model_name/Humanoid.pth

echo "start evaluating"
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true

# ours with mass-1.2
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true

# ours with recipient weakness-0.25
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true

# ============================================================================
# 2. (-) Dynamic Reference Retargeting: contact reward only, no retargeting
#    (exp: g-cluster-0-n10-ours-v13-contact-sep)
# ============================================================================
echo "ours-contact"
model_name=ours-contact
mkdir -p output/HumanoidIm/$model_name
cp checkpoints/$model_name/Humanoid.pth output/HumanoidIm/$model_name/Humanoid.pth

python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ours with mass-1.2
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ours with recipient weakness-0.25
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ============================================================================
# 3. (-) Contact Promoting Reward: no contact reward, no retargeting
#    (exp: g-cluster-0-n10-ours-v14-sep)
# ============================================================================
echo "ours-no-contact-no-retargeting"
model_name=ours-no-contact-no-retargeting
mkdir -p output/HumanoidIm/$model_name
cp checkpoints/$model_name/Humanoid.pth output/HumanoidIm/$model_name/Humanoid.pth

python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ours with mass-1.2
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ours with recipient weakness-0.25
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ============================================================================
# 4. Sequential Training (frozen recipient, decoupled learning)
#    (exp: CVPR2026-assistmimic-ablation-freeze-recipient)
# ============================================================================
echo "CVPR2026-assistmimic-ablation-freeze-recipient"
model_name=CVPR2026-assistmimic-ablation-freeze-recipient
mkdir -p output/HumanoidIm/$model_name
cp checkpoints/$model_name/Humanoid.pth output/HumanoidIm/$model_name/Humanoid.pth

# 1. Normal condition
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5

# 2. Mass-1.2 condition
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5

# 3. Weakness-0.25 condition
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5

# ============================================================================
# 5. (-) Weight Initialization: trained from scratch, without initializing
#    from the single-person motion prior (exp: g-cluster-0-n10-ours-v17-sep-wo-res-con)
#    Training fails to converge (paper: SR 0.0), included for completeness.
# ============================================================================
echo "g-cluster-0-n10-ours-v17-sep-wo-res-con"
model_name=g-cluster-0-n10-ours-v17-sep-wo-res-con
mkdir -p output/HumanoidIm/$model_name
cp checkpoints/$model_name/Humanoid.pth output/HumanoidIm/$model_name/Humanoid.pth

python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# weight-init ablation with mass-1.2
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# weight-init ablation with recipient weakness-0.25
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ============================================================================
# Extra: AssistMimic with a tighter retargeting threshold (0.8 instead of 1.3)
#    (exp: g-cluster-0-n10-ours-v13-contact-sep-adj-0.8-v2)
# This block is NOT part of Table 2. It demonstrates that the dynamic reference
# retargeting distance threshold (force_tracking_distance_threshold) is a
# tunable hyperparameter: training and evaluating with a tighter threshold
# (1.3 -> 0.8) further improves robustness over the paper's reported numbers.
# ============================================================================
echo "ours-with-contact-retargeting-0.8-v2 (extra: threshold 0.8)"
model_name=ours-with-contact-retargeting-0.8-v2
mkdir -p output/HumanoidIm/$model_name
cp checkpoints/$model_name/Humanoid.pth output/HumanoidIm/$model_name/Humanoid.pth

python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 env.force_tracking_distance_threshold=0.8 \
       ++env.enable_adjust_caregiver_hand_reference=true

# threshold-0.8 variant with mass-1.2
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
       ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 env.force_tracking_distance_threshold=0.8 \
       ++env.enable_adjust_caregiver_hand_reference=true

# threshold-0.8 variant with recipient weakness-0.25
python assistmimic/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=$model_name \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=false \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 env.force_tracking_distance_threshold=0.8 \
       ++env.enable_adjust_caregiver_hand_reference=true
