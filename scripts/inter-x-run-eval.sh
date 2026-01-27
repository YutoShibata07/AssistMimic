#!/bin/sh
# GPU 0
export CUDA_VISIBLE_DEVICES=0

# ours (exp: g-cluster-0-n10-ours-v13-contact-sep)
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=g-cluster-0-n10-ours-v13-contact-sep \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
#        ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5

# # ours with mass-1.2 (exp: g-cluster-0-n10-ours-v13-contact-sep)
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=g-cluster-0-n10-ours-v13-contact-sep \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
#        ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5

# # ours with recipient weakness-0.25 (exp: g-cluster-0-n10-ours-v13-contact-sep)
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=g-cluster-0-n10-ours-v13-contact-sep \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
#        ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5

# ours with different online retarget params
cp -r output/HumanoidIm/g-cluster-0-n10-ours-v13-contact-sep-adj-0.8/Humanoid.pth output/HumanoidIm/g-cluster-0-n10-ours-v13-contact-sep-adj-0.8/Humanoid_backup.pth
cp -r output/HumanoidIm/g-cluster-0-n10-ours-v13-contact-sep-adj-0.8/Humanoid_00002000.pth output/HumanoidIm/g-cluster-0-n10-ours-v13-contact-sep-adj-0.8/Humanoid.pth
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=g-cluster-0-n10-ours-v13-contact-sep-adj-0.8 \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
#        ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5 env.force_tracking_distance_threshold=0.8

# # ours with mass-1.2 (exp: g-cluster-0-n10-ours-v13-contact-sep)
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=g-cluster-0-n10-ours-v13-contact-sep-adj-0.8 \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
#        ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5 env.force_tracking_distance_threshold=0.8

# ours with recipient weakness-0.25 (exp: g-cluster-0-n10-ours-v13-contact-sep)
python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
       robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=g-cluster-0-n10-ours-v13-contact-sep-adj-0.8 \
       test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
       ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
       learning.params.network.freeze_recipient=true \
       ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
       ++env.terminationDistance=0.5 env.force_tracking_distance_threshold=0.8




# cp -r output/HumanoidIm/CVPR2026-assistmimic-ablation-freeze-recipient/Humanoid.pth output/HumanoidIm/CVPR2026-assistmimic-ablation-freeze-recipient/Humanoid_backup.pth
# cp -r output/HumanoidIm/CVPR2026-assistmimic-ablation-freeze-recipient/Humanoid_00002000.pth output/HumanoidIm/CVPR2026-assistmimic-ablation-freeze-recipient/Humanoid.pth
# # 1. Normal condition
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=CVPR2026-assistmimic-ablation-freeze-recipient \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
#        ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5

# # 2. Mass-1.2 condition
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=CVPR2026-assistmimic-ablation-freeze-recipient \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
#        ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5

# # 3. Weakness-0.25 condition
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=CVPR2026-assistmimic-ablation-freeze-recipient \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
#        ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5

# # vs Residual learning baseline
# cp -r output/HumanoidIm/CVPR2026-assistmimic-ablation-residual/Humanoid.pth output/HumanoidIm/CVPR2026-assistmimic-ablation-residual/Humanoid_backup.pth
# cp -r output/HumanoidIm/CVPR2026-assistmimic-ablation-residual/Humanoid_00002000.pth output/HumanoidIm/CVPR2026-assistmimic-ablation-residual/Humanoid.pth

# # 1. Normal condition
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=CVPR2026-assistmimic-ablation-residual \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
#        ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=normal rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=false learning.params.network.residual_mode=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5

# # 2. Mass-1.2 condition
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=CVPR2026-assistmimic-ablation-residual \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.84 \
#        ++env.kp_scale=1.0 ++env.kd_scale=1.0 eval_subdir=mass-1.2 rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=false learning.params.network.residual_mode=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5

# # 3. Weakness-0.25 condition
# python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp \
#        robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False exp_name=CVPR2026-assistmimic-ablation-residual \
#        test=True im_eval=True headless=True epoch=-1 env.num_envs=1000 ++env.recipient_mass_scale=0.7 \
#        ++env.recipient_weakness_scale=0.25 eval_subdir=weakness-0.25 rl_device=cuda:0 device_id=0 \
#        learning.params.network.freeze_recipient=false learning.params.network.residual_mode=true \
#        ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl \
#        ++env.terminationDistance=0.5