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

# Evaluate Tracking Policy

Full evaluation command (Inter-X dataset):
```bash
python phc/run_hydra.py \
  env=env_im_interx_helpup \
  learning=im_simpleliftup_mlp \
  robot=smplx_humanoid robot.freeze_hand=False robot.box_body=False \
  exp_name=g-cluster-0-n10-assistmimic-cvpr2026 \
  test=True im_eval=True headless=True epoch=-1 \
  env.num_envs=1000 \
  ++env.recipient_mass_scale=0.7 \
  eval_subdir=normal \
  rl_device=cuda:0 device_id=0 \
  learning.params.network.freeze_recipient=false \
  ++env.interx_data_path=sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl
```

### Evaluation Parameters

| Parameter | Description |
|-----------|-------------|
| `test=True im_eval=True` | Enable evaluation mode |
| `headless=True` | Run without GUI |
| `epoch=-1` | Use the latest checkpoint |
| `env.num_envs=1000` | Number of parallel environments for evaluation |
| `eval_subdir` | Subdirectory for saving evaluation results (e.g., `normal`, `mass-1.2`) |
| `++env.interx_data_path` | Path to test data (different from training data) |
| `++env.recipient_mass_scale` | Recipient mass scale (default: 0.7) |
| `++env.recipient_weakness_scale` | Recipient stiffness/damping scale (default: 0.5) |
| `rl_device`, `device_id` | GPU device settings |

---

## Training Modes

### Freeze Recipient Mode
Freezes the recipient network and trains only the caregiver network.

```bash
python phc/run_hydra.py \
  env=env_im_hhi-assist \
  learning=im_hhi-assist_mlp \
  exp_name=g-cluster-0-n10-assistmimic-cvpr2026 \
  test=False headless=True \
  learning.params.network.freeze_recipient=true
```

### Residual Mode (Experimental)
Residual learning mode where aux_mlp output is directly added to main features (no final_fc layer).

```bash
python phc/run_hydra.py \
  env=env_im_hhi-assist \
  learning=im_hhi-assist_mlp \
  exp_name=g-cluster-0-n10-assistmimic-cvpr2026 \
  test=False headless=True \
  learning.params.network.residual_mode=true
```

---

## Evaluation Options

### Dynamic Hand Reference Adjustment
Dynamically adjusts the caregiver's hand reference positions based on the recipient's physical simulation state.

```bash
# Enable (default)
python phc/run_hydra.py ... ++env.enable_adjust_caregiver_hand_reference=true

# Disable
python phc/run_hydra.py ... ++env.enable_adjust_caregiver_hand_reference=false
```

### Recipient Kinematic Replay
Forces the recipient to follow the reference motion kinematically. Used as a single-agent baseline.

```bash
python phc/run_hydra.py \
  env=env_im_interx_helpup \
  learning=im_simpleliftup_mlp \
  exp_name=g-cluster-0-n10-assistmimic-cvpr2026 \
  test=True im_eval=True headless=True \
  ++env.recipient_kinematic_replay=true
```

### Evaluation Metrics
The following metrics are computed during evaluation (for recipients only):

| Metric | Description |
|--------|-------------|
| `max_torque` | Maximum torque applied by the recipient during the episode |
| `com_stability` | Standard deviation of COM position over the episode (3D norm, lower = more stable) |
| `com_stability_xy` | COM stability in the xy plane only |

These metrics are reported separately for all episodes and successful episodes only.

### Recipient Weakness Parameters

#### Inter-X Dataset

| Parameter | Description | Default |
|-----------|-------------|---------|
| `recipient_mass_scale` | Mass scale factor | 0.7 |
| `recipient_weakness_scale` | Stiffness/damping scale for lower body | 0.5 |

```bash
# Evaluation with increased mass (mass-1.2)
python phc/run_hydra.py ... \
  ++env.recipient_mass_scale=0.84 \
  eval_subdir=mass-1.2

# Evaluation with weakened lower body
python phc/run_hydra.py ... \
  ++env.recipient_weakness_scale=0.25 \
  eval_subdir=weakness-0.25
```

#### HHI-Assist Dataset

| Parameter | Description | Default |
|-----------|-------------|---------|
| `recipient_mass_scale` | Mass scale factor | 0.7 |
| `recipient_weakness_scale` | Stiffness/damping scale | 0.5 |
| `recipient_hip_effort` | Max torque for hip joints | 20.0 |
| `recipient_upper_body_effort` | Max torque for upper body | 40.0 |

```bash
# Evaluation with increased mass (mass-1.5)
python phc/run_hydra.py ... \
  ++env.recipient_mass_scale=1.05 \
  eval_subdir=mass-1.5

# Evaluation with weakened hip
python phc/run_hydra.py ... \
  ++env.recipient_hip_effort=10 \
  eval_subdir=hip-torque-0.5
```

---

## Evaluation Scripts

**Use the provided shell scripts for evaluation.** Each dataset has different default recipient constraint settings.

### HHI-Assist Dataset

```bash
bash scripts/hhi-assist-run-eval.sh
```

**Default recipient constraints:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recipient_mass_scale` | 0.7 | Mass scale factor |
| `recipient_weakness_scale` | 0.5 | Stiffness/damping scale |
| `recipient_hip_effort` | 20.0 | Max torque for hip joints |
| `recipient_upper_body_effort` | 40.0 | Max torque for upper body |

**Evaluation conditions:**
- `normal`: Default settings (`recipient_mass_scale=0.7`)
- `mass-1.5`: Increased mass (`recipient_mass_scale=1.05`, i.e., 0.7 × 1.5)
- `hip-torque-0.5`: Weakened hip (`recipient_hip_effort=10`)

### Inter-X Dataset

```bash
bash scripts/inter-x-run-eval.sh
```

**Default recipient constraints:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recipient_mass_scale` | 0.7 | Mass scale factor |
| `recipient_weakness_scale` | 0.5 | Stiffness/damping scale (lower body only) |

**Evaluation conditions:**
- `normal`: Default settings (`recipient_mass_scale=0.7`)
- `mass-1.2`: Increased mass (`recipient_mass_scale=0.84`, i.e., 0.7 × 1.2)
- `weakness-0.25`: Weakened lower body (`recipient_weakness_scale=0.25`)

### Dataset Comparison

| Setting | HHI-Assist | Inter-X |
|---------|------------|---------|
| Default mass scale | 0.7 | 0.7 |
| Env config | `env_im_hhi-assist` | `env_im_interx_helpup` |
| Data path param | `hhi_assist_bed_data_path` | `interx_data_path` |
| Hip/Upper body effort | Configurable | Not used |

### Fair Policy Comparison (Common Success Intersection)
When comparing COM stability between policies, early-terminated episodes have unfairly low COM variance. Use this script to compute metrics only on motions where ALL policies succeeded:

```bash
python scripts/compare_policies_common_success.py \
    --policies output/HumanoidIm/policy_A \
               output/HumanoidIm/policy_B \
               output/HumanoidIm/policy_C \
    --condition normal
```

Options:
- `--policies`: Paths to 2+ policy directories (must have `evaluation/<condition>/evaluation_results.json`)
- `--condition`: Evaluation condition (`normal`, `mass-1.5`, `hip-torque-0.5`)
- `--all-conditions`: Compare all conditions at once
- `--output`: Save results to JSON file

The script reports:
- Per-policy success rates
- Metrics (max_torque, com_stability, com_stability_xy) computed only on common successful episodes
- Paired t-tests for statistical significance

