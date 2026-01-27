#!/usr/bin/env python
"""Compare model sizes in a checkpoint file.

This script analyzes checkpoint files to show:
1. Parameter counts for each model component (pnn, caregiver_aux_mlp, recipient_aux_mlp, etc.)
2. Which components are actually used for action output based on configuration
3. Layer-by-layer architecture breakdown
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model sizes in checkpoint")
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name (folder in output/HumanoidIm/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Humanoid.pth",
        help="Checkpoint filename (default: Humanoid.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/HumanoidIm",
        help="Output directory containing experiments",
    )
    return parser.parse_args()


def format_params(count):
    """Format parameter count with commas."""
    return f"{count:,}"


def infer_layer_shape(key, tensor):
    """Infer layer type and shape from key and tensor."""
    shape = tuple(tensor.shape)
    if len(shape) == 2:
        # Linear layer: (out_features, in_features)
        return f"Linear({shape[1]}, {shape[0]})"
    elif len(shape) == 1:
        return f"Bias({shape[0]})"
    elif len(shape) == 4:
        # Conv layer: (out_channels, in_channels, kernel_h, kernel_w)
        return f"Conv2d({shape[1]}, {shape[0]}, kernel={shape[2]}x{shape[3]})"
    else:
        return f"Tensor{shape}"


def extract_component_name(key):
    """Extract the component name from a state dict key."""
    # Remove 'a2c_network.' prefix
    if key.startswith("a2c_network."):
        key = key[len("a2c_network.") :]

    # Group by first component
    parts = key.split(".")
    if parts[0] in ["pnn", "caregiver_aux_mlp", "recipient_aux_mlp", "aux_mlp"]:
        return parts[0]
    elif parts[0] in ["caregiver_final_fc", "recipient_final_fc", "final_fc"]:
        return parts[0]
    elif parts[0] == "critic_mlp":
        return "critic_mlp"
    elif parts[0] == "_disc_mlp" or parts[0] == "_disc_logits":
        return "discriminator"
    elif parts[0] == "trajectory_cnn":
        return "trajectory_cnn"
    elif parts[0] in ["mu", "value"]:
        return parts[0]
    elif parts[0] in ["sigma", "caregiver_sigma", "recipient_sigma"]:
        return "sigma"
    else:
        return parts[0]


def analyze_checkpoint(ckpt_path):
    """Analyze a checkpoint file and return component statistics."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" not in ckpt:
        raise ValueError("Checkpoint does not contain 'model' key")

    state_dict = ckpt["model"]

    # Group parameters by component
    components = defaultdict(lambda: {"params": 0, "layers": []})

    for key, tensor in sorted(state_dict.items()):
        component = extract_component_name(key)
        param_count = tensor.numel()

        components[component]["params"] += param_count

        # Only add weight layers (not bias) for layer info
        if not key.endswith(".bias") and "running_" not in key and "num_batches" not in key:
            layer_info = infer_layer_shape(key, tensor)
            short_key = key.replace("a2c_network.", "")
            components[component]["layers"].append(
                {"key": short_key, "shape": layer_info, "params": param_count}
            )

    return components


def print_component(name, data, note=""):
    """Print component details."""
    header = f"[{name}]"
    if note:
        header += f" ({note})"
    print(f"\n{header}")
    print("-" * 60)

    for layer in data["layers"]:
        # Truncate key if too long
        key = layer["key"]
        if len(key) > 40:
            key = "..." + key[-37:]
        print(f"  {key:<42} {layer['shape']:<25} {format_params(layer['params']):>12}")

    print(f"  {'Total:':<42} {'':<25} {format_params(data['params']):>12}")


def main():
    args = parse_args()

    # Construct checkpoint path
    script_dir = Path(__file__).parent.parent  # Go up from scripts/
    ckpt_path = script_dir / args.output_dir / args.exp_name / args.checkpoint

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print(f"\nAvailable experiments:")
        exp_dir = script_dir / args.output_dir
        if exp_dir.exists():
            for d in sorted(exp_dir.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}")
        return 1

    components = analyze_checkpoint(ckpt_path)

    # Print header
    print("\n" + "=" * 70)
    print("                     MODEL SIZE COMPARISON")
    print("=" * 70)
    print(f"Checkpoint: {ckpt_path}")

    # Define display order and notes
    component_info = {
        "pnn": "COMPUTED BUT NOT USED when weight_share=False, residual_mode=False",
        "caregiver_aux_mlp": "ACTUAL CAREGIVER ACTION MODEL",
        "recipient_aux_mlp": "ACTUAL RECIPIENT ACTION MODEL",
        "aux_mlp": "Shared aux MLP (if exists)",
        "caregiver_final_fc": "Caregiver final output layer",
        "recipient_final_fc": "Recipient final output layer",
        "final_fc": "Shared final output layer",
        "critic_mlp": "Value function critic",
        "discriminator": "AMP discriminator",
        "trajectory_cnn": "Trajectory encoder CNN",
        "mu": "Policy mean output",
        "value": "Value head output",
        "sigma": "Policy std parameters",
    }

    # Print actor components first (most relevant for comparison)
    print("\n" + "=" * 70)
    print("                     ACTOR NETWORKS")
    print("=" * 70)

    actor_keys = ["pnn", "caregiver_aux_mlp", "recipient_aux_mlp", "aux_mlp",
                  "caregiver_final_fc", "recipient_final_fc", "final_fc"]

    total_actor_params = 0
    used_actor_params = 0

    for key in actor_keys:
        if key in components:
            note = component_info.get(key, "")
            print_component(key, components[key], note)
            total_actor_params += components[key]["params"]
            if key != "pnn":  # pnn is not used
                used_actor_params += components[key]["params"]

    # Print other components
    print("\n" + "=" * 70)
    print("                     OTHER COMPONENTS")
    print("=" * 70)

    other_total = 0
    for key, data in sorted(components.items()):
        if key not in actor_keys:
            note = component_info.get(key, "")
            print_component(key, data, note)
            other_total += data["params"]

    # Print summary
    print("\n" + "=" * 70)
    print("                     SUMMARY")
    print("=" * 70)

    total_params = sum(d["params"] for d in components.values())

    print(f"\n{'Component':<35} {'Parameters':>15}")
    print("-" * 52)

    for key in actor_keys:
        if key in components:
            status = " (NOT USED)" if key == "pnn" else ""
            print(f"  {key + status:<33} {format_params(components[key]['params']):>15}")

    print("-" * 52)
    print(f"  {'Actor Total (with pnn)':<33} {format_params(total_actor_params):>15}")
    print(f"  {'Actor Total (without pnn)':<33} {format_params(used_actor_params):>15}")
    print(f"  {'Other Components':<33} {format_params(other_total):>15}")
    print("-" * 52)
    print(f"  {'TOTAL MODEL':<33} {format_params(total_params):>15}")

    # Calculate what percentage is wasted
    if "pnn" in components and total_params > 0:
        pnn_params = components["pnn"]["params"]
        pct = (pnn_params / total_params) * 100
        print(f"\n⚠️  {format_params(pnn_params)} params ({pct:.1f}%) in 'pnn' are computed but NOT USED")
        print("    for action output when weight_share=False and residual_mode=False.")

    return 0


if __name__ == "__main__":
    exit(main())
