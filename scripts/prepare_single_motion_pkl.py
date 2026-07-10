"""
Extract a single motion from a pkl file.
Usage:
    python scripts/prepare_single_motion_pkl.py \
        --src sample_data/interx_processed_fixed_v9_cluster_ids_0_1_2_4_n_clusters_10.pkl \
        --motion_id G058T010A035R003 \
        --out sample_data/interx_G058T010A035R003.pkl
"""
import argparse
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Source pkl file")
    parser.add_argument("--motion_id", type=str, required=True, help="Motion sequence ID (e.g. G058T010A035R003)")
    parser.add_argument("--out", type=str, required=True, help="Output pkl file")
    args = parser.parse_args()

    data = joblib.load(args.src)

    caregiver_key = f"{args.motion_id}_caregiver"
    recipient_key = f"{args.motion_id}_recipient"

    assert caregiver_key in data, f"{caregiver_key} not found in {args.src}"
    assert recipient_key in data, f"{recipient_key} not found in {args.src}"

    new_data = {
        caregiver_key: data[caregiver_key],
        recipient_key: data[recipient_key],
    }

    joblib.dump(new_data, args.out)
    print(f"Saved {caregiver_key} and {recipient_key} to {args.out}")


if __name__ == "__main__":
    main()
