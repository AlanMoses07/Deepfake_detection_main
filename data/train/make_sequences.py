# data/train/make_sequences.py
import argparse
import json
import random
from pathlib import Path

def get_feature_files(features_root):
    items = []
    for cls_type in ["real", "fake"]:
        cls_dir = Path(features_root) / cls_type
        label = 0 if cls_type == "real" else 1
        for npy_file in sorted(cls_dir.glob("*.npy")):
            items.append((str(npy_file), label))
    return items

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create data splits for training.")
    parser.add_argument("--features_root", default="data/features", help="Directory with feature .npy files.")
    parser.add_argument("--seq_len", type=int, default=20, help="Length of frame sequences for the LSTM.")
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], help="Train/Val/Test split ratio.")
    parser.add_argument("--out", default="data/sequences.json", help="Output JSON file for data splits.")
    args = parser.parse_args()

    all_items = get_feature_files(args.features_root)
    random.seed(42)
    random.shuffle(all_items)

    num_items = len(all_items)
    num_train = int(num_items * args.split[0])
    num_val = int(num_items * args.split[1])
    
    splits = {
        "train": all_items[:num_train],
        "val": all_items[num_train : num_train + num_val],
        "test": all_items[num_train + num_val:],
    }

    metadata = {"seq_len": args.seq_len, "splits": splits}
    Path(args.out).write_text(json.dumps(metadata, indent=4))
    
    print(f"Wrote data splits to {args.out}:")
    for split_name, split_items in splits.items():
        print(f"  - {split_name}: {len(split_items)} videos")