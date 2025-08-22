# --- THIS IS THE FULLY CORRECTED SCRIPT for data/eval/eval.py ---
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# --- FIX: Add project root to Python's path ---
# This allows the script to find the 'data' module for importing.
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
# ----------------------------------------------

# This import will now work because the project root is in the path.
from data.train.train_lstm import LSTMDetector, SequenceDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test set.")
    parser.add_argument("--meta", default="data/sequences.json", help="Path to the data splits JSON file, relative to project root.")
    parser.add_argument("--ckpt", default="data/checkpoints/best_model.pt", help="Path to the model checkpoint, relative to project root.")
    args = parser.parse_args()

    # Use absolute paths based on the detected project root to avoid file not found errors.
    meta_path = project_root / args.meta
    ckpt_path = project_root / args.ckpt

    if not meta_path.exists():
        print(f"[ERROR] Metadata file not found at: {meta_path}")
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint file not found at: {ckpt_path}")
        sys.exit(1)

    # Load the data splits.
    with open(meta_path, 'r') as f:
        test_items = json.load(f)["splits"]["test"]

    # Load the trained model checkpoint.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    feature_dim = ckpt["feature_dim"]
    seq_len = ckpt["seq_len"]

    # Initialize the model and load its learned weights.
    model = LSTMDetector(feature_dim)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Adjust paths in test_items to be absolute so the script can find the .npy files.
    abs_test_items = [(str(project_root / p), l) for p, l in test_items]

    test_ds = SequenceDataset(abs_test_items, seq_len)
    all_predictions, all_ground_truths = [], []

    if len(test_ds) == 0:
        print("[ERROR] No data found in the test set. Cannot evaluate.")
        sys.exit(1)

    # Run predictions on the test set.
    with torch.no_grad():
        for i in tqdm(range(len(test_ds)), desc="Evaluating on test set"):
            x, y = test_ds[i]
            # Add a batch dimension (unsqueeze) for the model.
            logit = model(x.unsqueeze(0))
            probability = torch.sigmoid(logit).item()
            all_predictions.append(probability)
            all_ground_truths.append(int(y.item()))

    # Calculate and print performance metrics.
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

    auc = roc_auc_score(all_ground_truths, all_predictions)

    # Find the best threshold that maximizes the F1 score.
    precision, recall, thresholds = precision_recall_curve(all_ground_truths, all_predictions)
    f1_scores = 2 * recall[:-1] * precision[:-1] / (recall[:-1] + precision[:-1] + 1e-9)
    
    best_threshold = 0.5
    if len(f1_scores) > 0:
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

    # Classify predictions based on the optimal threshold.
    y_pred_class = [1 if p >= best_threshold else 0 for p in all_predictions]
    f1 = f1_score(all_ground_truths, y_pred_class)

    print("\n--- Evaluation Results ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Best F1 Score: {f1:.4f}")
    print(f"Optimal Threshold for Classification: {best_threshold:.4f}")
    print("--------------------------")