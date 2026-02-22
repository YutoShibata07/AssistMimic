#!/bin/sh
# HHI-Assist Evaluation Script
# Runs evaluation under various conditions for the HHI-assist task
# Compares two policies:
#   1. AA-RM-wo-FullAssist-ours-v16-sep-con-adj (with adjust_caregiver_hand_reference)
#   2. AA-RM-wo-FullAssist-ours-v16-sep (without adjust_caregiver_hand_reference)

export CUDA_VISIBLE_DEVICES=1

# ============================================================================
# Policy 1: AA-RM-wo-FullAssist-ours-v16-sep-con-adj
# Uses _adjust_caregiver_hand_reference (default: enabled)
# ============================================================================
EXP_NAME="AA-RM-wo-FullAssist-ours-v16-sep-con-adj"

# Backup current checkpoint and use specific epoch
cp -r output/HumanoidIm/${EXP_NAME}/Humanoid.pth output/HumanoidIm/${EXP_NAME}/Humanoid_backup.pth
cp -r output/HumanoidIm/${EXP_NAME}/Humanoid_00002000.pth output/HumanoidIm/${EXP_NAME}/Humanoid.pth

# 1. Normal condition
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       eval_subdir=normal learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true

# # 2. Mass-1.5 condition (recipient mass scale = 0.7 * 1.5 = 1.05)
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=1.05 \
       eval_subdir=mass-1.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true


# # 3. Hip-torque-0.5 condition (reduce hip max torque from 20 to 10)
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_hip_effort=10 \
       eval_subdir=hip-torque-0.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=true


# ============================================================================
# Policy 2: AA-RM-wo-FullAssist-ours-v16-sep
# Does NOT use _adjust_caregiver_hand_reference (disabled)
# ============================================================================
EXP_NAME="AA-RM-wo-FullAssist-ours-v14-sep"
cp -r output/HumanoidIm/${EXP_NAME}/Humanoid.pth output/HumanoidIm/${EXP_NAME}/Humanoid_backup.pth
cp -r output/HumanoidIm/${EXP_NAME}/Humanoid_00002000.pth output/HumanoidIm/${EXP_NAME}/Humanoid.pth

# 1. Normal condition (WITHOUT adjust_caregiver_hand_reference)
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       eval_subdir=normal learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# # 2. Mass-1.5 condition (WITHOUT adjust_caregiver_hand_reference)
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=1.05 \
       eval_subdir=mass-1.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# # 3. Hip-torque-0.5 condition (WITHOUT adjust_caregiver_hand_reference)
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_hip_effort=10 \
       eval_subdir=hip-torque-0.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

EXP_NAME="AA-RM-wo-FullAssist-ours-v14-sep-con"
cp -r output/HumanoidIm/${EXP_NAME}/Humanoid.pth output/HumanoidIm/${EXP_NAME}/Humanoid_backup.pth
cp -r output/HumanoidIm/${EXP_NAME}/Humanoid_00002000.pth output/HumanoidIm/${EXP_NAME}/Humanoid.pth

# 1. Normal condition
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       eval_subdir=normal learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false

# 2. Mass-1.5 condition (recipient mass scale = 0.7 * 1.5 = 1.05)
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=1.05 \
       eval_subdir=mass-1.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false


# 3. Hip-torque-0.5 condition (reduce hip max torque from 20 to 10)
python phc/run_hydra.py env=env_im_hhi-assist learning=im_hhi-assist_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
       exp_name=$EXP_NAME test=True im_eval=True headless=True epoch=-1 \
       env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_hip_effort=10 \
       eval_subdir=hip-torque-0.5 learning.params.network.freeze_recipient=false \
       ++env.hhi_assist_bed_data_path=sample_data/hhi-assist_processed_v6_AA-RM-wo-FullAssist_100.pkl \
       ++env.terminationDistance=0.5 \
       ++env.enable_adjust_caregiver_hand_reference=false


