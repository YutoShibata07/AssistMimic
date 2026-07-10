#!/bin/sh
# HHI-Assist Evaluation Script
# Reproduces Table 3 of the paper (specialist policies on the HHI-Assist dataset):
#   1. AssistMimic (full)                -> AA-RM-wo-FullAssist-ours-v16-sep-con-adj (with adjust_caregiver_hand_reference)
#   2. (-) Contact Promoting Reward      -> AA-RM-wo-FullAssist-ours-v14-sep (no contact reward, no retargeting)
#   3. (-) Dynamic Reference Retargeting -> AA-RM-wo-FullAssist-ours-v14-sep-con (contact reward, without adjust_caregiver_hand_reference)
#   4. (-) Weight Initialization         -> AA-RM-wo-FullAssist-ours-v17-wo-res-con-adj-fitting (trained from scratch, no motion prior init)

# Set to the GPU you want to use
export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# Policy 1: AA-RM-wo-FullAssist-ours-v16-sep-con-adj
# Uses _adjust_caregiver_hand_reference (default: enabled)
# ============================================================================
EXP_NAME="AA-RM-wo-FullAssist-ours-v16-sep-con-adj"
motion_path="sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100_short.pkl"

mkdir -p output/HumanoidIm/${EXP_NAME}
cp checkpoints/${EXP_NAME}/Humanoid.pth output/HumanoidIm/${EXP_NAME}/Humanoid.pth
# 1. Normal condition
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       eval_subdir=normal learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true

# # 2. Mass-1.5 condition (recipient mass scale = 0.7 * 1.5 = 1.05)
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=1.05 \
       eval_subdir=mass-1.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true


# # 3. Hip-torque-0.5 condition (reduce hip max torque from 20 to 10)
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_hip_effort=10 \
       eval_subdir=hip-torque-0.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true


# ============================================================================
# Policy 2: AA-RM-wo-FullAssist-ours-v14-sep
# (-) Contact Promoting Reward: no contact reward, no retargeting
# ============================================================================
EXP_NAME="AA-RM-wo-FullAssist-ours-v14-sep"
mkdir -p output/HumanoidIm/${EXP_NAME}
cp checkpoints/${EXP_NAME}/Humanoid.pth output/HumanoidIm/${EXP_NAME}/Humanoid.pth

# 1. Normal condition (WITHOUT adjust_caregiver_hand_reference)
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       eval_subdir=normal learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# # 2. Mass-1.5 condition (WITHOUT adjust_caregiver_hand_reference)
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=1.05 \
       eval_subdir=mass-1.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# # 3. Hip-torque-0.5 condition (WITHOUT adjust_caregiver_hand_reference)
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_hip_effort=10 \
       eval_subdir=hip-torque-0.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ============================================================================
# Policy 3: AA-RM-wo-FullAssist-ours-v14-sep-con
# (-) Dynamic Reference Retargeting: contact reward, without adjust_caregiver_hand_reference
# ============================================================================
EXP_NAME="AA-RM-wo-FullAssist-ours-v14-sep-con"
mkdir -p output/HumanoidIm/${EXP_NAME}
cp checkpoints/${EXP_NAME}/Humanoid.pth output/HumanoidIm/${EXP_NAME}/Humanoid.pth
# 1. Normal condition
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       eval_subdir=normal learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# 2. Mass-1.5 condition (recipient mass scale = 0.7 * 1.5 = 1.05)
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=1.05 \
       eval_subdir=mass-1.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false


# 3. Hip-torque-0.5 condition (reduce hip max torque from 20 to 10)
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_hip_effort=10 \
       eval_subdir=hip-torque-0.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# ============================================================================
# Policy 4: AA-RM-wo-FullAssist-ours-v17-wo-res-con-adj-fitting
# (-) Weight Initialization: trained from scratch, without initializing from
# the single-person motion prior. Training succumbs to reward hacking
# (paper: SR 19.1 marked with a dagger); unseen-dynamics conditions are not
# evaluated for this row, so only the normal condition is run.
# ============================================================================
EXP_NAME="AA-RM-wo-FullAssist-ours-v17-wo-res-con-adj-fitting"
mkdir -p output/HumanoidIm/${EXP_NAME}
cp checkpoints/${EXP_NAME}/Humanoid.pth output/HumanoidIm/${EXP_NAME}/Humanoid.pth
# 1. Normal condition
python assistmimic/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       eval_subdir=normal learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=$motion_path \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true


