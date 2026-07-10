#!/usr/bin/env python3
"""
Compare multiple policies on common successful motions.

This script compares evaluation metrics between multiple policies, computing statistics
only on motions where ALL policies succeeded. This ensures fair comparison by
avoiding bias from episodes that failed quickly.

Usage:
    python scripts/compare_policies_common_success.py \
        --policies output/HumanoidIm/AA-RM-wo-FullAssist-ours-v16-sep-con-adj \
                   output/HumanoidIm/AA-RM-wo-FullAssist-ours-v14-sep \
                   output/HumanoidIm/AA-RM-wo-FullAssist-ours-v14-sep-con \
        --condition normal
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from scipy import stats


def load_evaluation_results(policy_dir: str, condition: str) -> dict:
    """Load evaluation results JSON for a policy and condition."""
    json_path = Path(policy_dir) / "evaluation" / condition / "evaluation_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {json_path}")

    with open(json_path, 'r') as f:
        return json.load(f)


def compute_common_success_metrics_multi(data_list: list, policy_names: list) -> dict:
    """
    Compute metrics on common successful motions across multiple policies.

    Args:
        data_list: List of evaluation results from each policy
        policy_names: List of policy names

    Returns:
        Dictionary containing comparison results
    """
    # Get per-episode metrics from each policy
    per_ep_list = [data.get('per_episode_metrics', {}) for data in data_list]

    for i, per_ep in enumerate(per_ep_list):
        if not per_ep:
            raise ValueError(f"per_episode_metrics not found in policy {policy_names[i]}")

    # Get success masks
    success_masks = [np.array(per_ep['success_mask']) for per_ep in per_ep_list]

    # Verify same length
    num_episodes = len(success_masks[0])
    for i, mask in enumerate(success_masks):
        if len(mask) != num_episodes:
            raise ValueError(f"Different number of episodes: {policy_names[0]} has {num_episodes}, {policy_names[i]} has {len(mask)}")

    # Find common successful episodes (intersection of all)
    common_success = success_masks[0].copy()
    for mask in success_masks[1:]:
        common_success = common_success & mask
    common_success_indices = np.where(common_success)[0]

    # Get metrics arrays for each policy
    max_torques_list = [np.array(per_ep['max_torques']) for per_ep in per_ep_list]
    com_stability_list = [np.array(per_ep['com_stability']) for per_ep in per_ep_list]
    com_stability_xy_list = [np.array(per_ep['com_stability_xy']) for per_ep in per_ep_list]

    # Build results
    results = {
        'num_episodes': num_episodes,
        'num_policies': len(policy_names),
        'policy_names': policy_names,
        'common_success_count': int(common_success.sum()),
        'policies': {}
    }

    # Per-policy statistics
    for i, name in enumerate(policy_names):
        policy_success = success_masks[i]
        results['policies'][name] = {
            'success_count': int(policy_success.sum()),
            'success_rate': float(policy_success.sum() / num_episodes),
            'max_torque_mean': float(np.mean(max_torques_list[i][common_success_indices])),
            'max_torque_std': float(np.std(max_torques_list[i][common_success_indices])),
            'com_stability_mean': float(np.mean(com_stability_list[i][common_success_indices])),
            'com_stability_std': float(np.std(com_stability_list[i][common_success_indices])),
            'com_stability_xy_mean': float(np.mean(com_stability_xy_list[i][common_success_indices])),
            'com_stability_xy_std': float(np.std(com_stability_xy_list[i][common_success_indices])),
        }

    # Pairwise comparisons and t-tests
    results['pairwise_comparisons'] = {}
    for i in range(len(policy_names)):
        for j in range(i + 1, len(policy_names)):
            name_i, name_j = policy_names[i], policy_names[j]
            key = f"{name_i} vs {name_j}"

            if len(common_success_indices) > 1:
                # Max torque t-test
                t_stat, p_value = stats.ttest_rel(
                    max_torques_list[i][common_success_indices],
                    max_torques_list[j][common_success_indices]
                )
                max_torque_diff = float(np.mean(max_torques_list[i][common_success_indices]) -
                                        np.mean(max_torques_list[j][common_success_indices]))

                # COM stability t-test
                t_stat_com, p_value_com = stats.ttest_rel(
                    com_stability_list[i][common_success_indices],
                    com_stability_list[j][common_success_indices]
                )
                com_diff = float(np.mean(com_stability_list[i][common_success_indices]) -
                                np.mean(com_stability_list[j][common_success_indices]))

                # COM stability XY t-test
                t_stat_xy, p_value_xy = stats.ttest_rel(
                    com_stability_xy_list[i][common_success_indices],
                    com_stability_xy_list[j][common_success_indices]
                )
                com_xy_diff = float(np.mean(com_stability_xy_list[i][common_success_indices]) -
                                   np.mean(com_stability_xy_list[j][common_success_indices]))

                results['pairwise_comparisons'][key] = {
                    'max_torque': {
                        'diff': max_torque_diff,
                        't_stat': float(t_stat),
                        'p_value': float(p_value)
                    },
                    'com_stability': {
                        'diff': com_diff,
                        't_stat': float(t_stat_com),
                        'p_value': float(p_value_com)
                    },
                    'com_stability_xy': {
                        'diff': com_xy_diff,
                        't_stat': float(t_stat_xy),
                        'p_value': float(p_value_xy)
                    }
                }

    return results


def print_comparison_report_multi(results: dict, condition: str):
    """Print a formatted comparison report for multiple policies."""
    policy_names = results['policy_names']

    print("=" * 100)
    print(f"Multi-Policy Comparison Report - Condition: {condition}")
    print("=" * 100)

    print(f"\nPolicies compared ({len(policy_names)}):")
    for i, name in enumerate(policy_names):
        print(f"  P{i+1}: {name}")

    print(f"\n{'='*50}")
    print("Episode Statistics")
    print(f"{'='*50}")
    print(f"Total episodes:           {results['num_episodes']}")
    print(f"Common success (ALL):     {results['common_success_count']} ({100*results['common_success_count']/results['num_episodes']:.1f}%)")
    print()
    for i, name in enumerate(policy_names):
        policy_stats = results['policies'][name]
        print(f"  P{i+1} success: {policy_stats['success_count']} ({100*policy_stats['success_rate']:.1f}%)")

    print(f"\n{'='*50}")
    print("Metrics on Common Successful Episodes")
    print(f"{'='*50}")

    # Summary table
    print(f"\n{'Policy':<45} {'MaxTorque':>12} {'COM Stab':>12} {'COM XY':>12}")
    print("-" * 85)
    for i, name in enumerate(policy_names):
        policy_stats = results['policies'][name]
        short_name = name[-40:] if len(name) > 40 else name
        print(f"P{i+1}: {short_name:<42} {policy_stats['max_torque_mean']:>8.1f}±{policy_stats['max_torque_std']:<5.1f}"
              f"{policy_stats['com_stability_mean']:>8.4f}±{policy_stats['com_stability_std']:<6.4f}"
              f"{policy_stats['com_stability_xy_mean']:>8.4f}±{policy_stats['com_stability_xy_std']:<6.4f}")

    print(f"\n{'='*100}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple policies on common successful motions')
    parser.add_argument('--policies', type=str, nargs='+', required=True,
                        help='Paths to policy output directories (2 or more)')
    parser.add_argument('--condition', type=str, default='normal',
                        choices=['normal', 'mass-1.5', 'hip-torque-0.5', 'mass-1.2', 'weakness-0.25'],
                        help='Evaluation condition to compare')
    parser.add_argument('--all-conditions', action='store_true',
                        help='Compare all available conditions (auto-detected from first policy)')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['hhi-assist', 'inter-x'],
                        help='Dataset type for --all-conditions (auto-detected if not specified)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (optional)')

    args = parser.parse_args()

    if len(args.policies) < 2:
        parser.error("At least 2 policies are required for comparison")

    if args.all_conditions:
        hhi_conditions = ['normal', 'mass-1.5', 'hip-torque-0.5']
        interx_conditions = ['normal', 'mass-1.2', 'weakness-0.25']

        if args.dataset == 'hhi-assist':
            conditions = hhi_conditions
        elif args.dataset == 'inter-x':
            conditions = interx_conditions
        else:
            # Auto-detect by checking which condition dirs exist in the first policy
            eval_dir = Path(args.policies[0]) / "evaluation"
            if eval_dir.exists():
                existing = {d.name for d in eval_dir.iterdir() if d.is_dir()}
                if 'mass-1.2' in existing or 'weakness-0.25' in existing:
                    conditions = interx_conditions
                else:
                    conditions = hhi_conditions
            else:
                conditions = hhi_conditions
    else:
        conditions = [args.condition]
    all_results = {}

    policy_names = [os.path.basename(p) for p in args.policies]

    for condition in conditions:
        try:
            data_list = [load_evaluation_results(policy, condition) for policy in args.policies]
            results = compute_common_success_metrics_multi(data_list, policy_names)
            all_results[condition] = results
            print_comparison_report_multi(results, condition)

        except FileNotFoundError as e:
            print(f"Skipping condition '{condition}': {e}")
        except Exception as e:
            print(f"Error processing condition '{condition}': {e}")
            raise

    # Save results to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
