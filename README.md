# Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning 


Official implementation of ~~ paper: "Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning". 

# Environment Setup

This project relies on **PHC (Perpetual Humanoid Control for Real-time Simulated Avatars)**.  
Please follow the instructions below to correctly set up the environment and download all required assets.

---

## 1. Install PHC and Prepare Dependencies

Visit the official PHC GitHub repository and complete **Setup Steps 1–4**:

➡️ **PHC Repository:** https://github.com/ZhengyiLuo/PHC

These steps include:

- Creating the appropriate Python environment  
- Installing required Python and system dependencies  
- Downloading the required **SMPL / SMPL-X model files**  
- Downloading PHC sample datasets

Make sure that all four setup steps are completed successfully.

## 2. Download pretrained GMT policy weight
- Download pretraiend weights at : https://drive.google.com/drive/folders/12DFXtGtSjiHdyqru4FzwYfKg3uMPbVWw?usp=drive_link
- Put this folder under **output/**

# Train tracking policy
```bash
python phc/run_hydra.py env=env_im_interx_helpup learning=im_simpleliftup_mlp exp_name=g-cluster-0-n10-assistmimic-cvpr2026 test=False headless=True robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False
```





