#!/usr/bin/env python3
"""Automated Evaluation Script for Trained Policies"""

import subprocess
import os
import json
import argparse
from datetime import datetime

DEFAULT_MODELS = [
    "CVPR2026-assistmimic-ablation-residual",
    "CVPR2026-assistmimic-ablation-freeze-recipient",
]

DEFAULT_TEST_DATA = "sample_data/interx_processed_fixed_v9_cluster_ids_0_n_clusters_10_100.pkl"
DEFAULT_NUM_ENVS = 1000

CONDITIONS = {
    "normal": {"recipient_mass_scale": 0.7, "kp_scale": 1.0, "kd_scale": 1.0},
    "mass-1.2": {"recipient_mass_scale": 0.84, "kp_scale": 1.0, "kd_scale": 1.0},
    "kp-0.5": {"recipient_mass_scale": 0.7, "kp_scale": 0.5, "kd_scale": 0.5},
}


CONDA_ENV = "multi_isaac"
CONDA_PREFIX = os.path.expanduser("~/slocal1/miniconda3/envs/multi_isaac")

def run_evaluation(model, condition_name, params, test_data, num_envs, gpu_id=1, dry_run=False):
    # Build hydra arguments
    hydra_args = [
        # Environment and learning configs
        "env=env_im_interx_helpup",
        "learning=im_simpleliftup_mlp",
        # Robot config (SMPLX with 52 bodies)
        "robot=smplx_humanoid",
        "robot.freeze_hand=False",
        "robot.box_body=False",
        # Model and evaluation settings
        f"exp_name={model}",
        "im_eval=True", "test=True", "headless=True", "no_log=True", "epoch=-1",
        f"eval_subdir={condition_name}",
        f"env.num_envs={num_envs}",
        f"++env.interx_data_path={test_data}",
        f"++env.recipient_mass_scale={params['recipient_mass_scale']}",
        f"++env.kp_scale={params['kp_scale']}",
        f"++env.kd_scale={params['kd_scale']}",
        f"rl_device=cuda:{gpu_id}",
        f"device_id={gpu_id}",
    ]

    # Build shell command with proper environment setup
    shell_cmd = (
        f"source ~/slocal1/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"export LD_LIBRARY_PATH=\"{CONDA_PREFIX}/lib:$LD_LIBRARY_PATH\" && "
        f"python phc/run_hydra.py {' '.join(hydra_args)}"
    )

    print(f"\n{'='*60}")
    print(f"Model: {model} | Condition: {condition_name}")
    print(f"Command: python phc/run_hydra.py {' '.join(hydra_args)}")

    if dry_run:
        return {"status": "dry_run", "model": model, "condition": condition_name}

    try:
        subprocess.run(shell_cmd, shell=True, check=True, executable="/bin/bash")
        return {"status": "success", "model": model, "condition": condition_name}
    except subprocess.CalledProcessError as e:
        return {"status": "failed", "model": model, "condition": condition_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--conditions", default=",".join(CONDITIONS.keys()))
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--gpu-id", type=int, default=1, help="GPU device ID (default: 1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    conditions = [c.strip() for c in args.conditions.split(",")]

    results = []
    for model in models:
        for cond in conditions:
            result = run_evaluation(
                model, cond, CONDITIONS[cond],
                args.test_data, args.num_envs, args.gpu_id, args.dry_run
            )
            results.append(result)

    # Summary
    print(f"\n{'='*60}\nSummary")
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"Success: {success}, Failed: {failed}")

    # Save summary
    os.makedirs("output/evaluation_results", exist_ok=True)
    summary_path = f"output/evaluation_results/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
