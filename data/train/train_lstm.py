# data/train/train_lstm.py
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class SequenceDataset(Dataset):
    def __init__(self, items, seq_len):
        self.items = items
        self.seq_len = seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        features = np.load(path).astype(np.float32)
        
        # Pad if the sequence is too short
        if len(features) < self.seq_len:
            repeats = (self.seq_len + len(features) - 1) // len(features)
            features = np.tile(features, (repeats, 1))
        
        start_idx = random.randint(0, len(features) - self.seq_len)
        sequence = features[start_idx : start_idx + self.seq_len]
        
        return torch.tensor(sequence), torch.tensor(label, dtype=torch.float32)

class LSTMDetector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        # Use the output of the last time step
        return self.fc(h[:, -1, :]).squeeze(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM deepfake detector.")
    parser.add_argument("--meta", default="data/sequences.json", help="Path to the data splits JSON file.")
    parser.add_argument("--features_root", default="data/features", help="Path to the features root to read metadata.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision.")
    parser.add_argument("--out", default="data/checkpoints/best_model.pt", help="Path to save the best model.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training.")
    args = parser.parse_args()

    with open(args.meta, 'r') as f:
        metadata = json.load(f)
    seq_len = metadata["seq_len"]
    splits = metadata["splits"]

    feature_dim = int((Path(args.features_root) / "feature_dim.txt").read_text().strip())
    model_name = (Path(args.features_root) / "model_name.txt").read_text().strip()

    train_ds = SequenceDataset(splits["train"], seq_len)
    val_ds = SequenceDataset(splits["val"], seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMDetector(feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_auc = -1.0
    start_epoch = 1

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_auc = ckpt.get("best_auc", -1.0)
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"[RESUME] Loaded checkpoint from {args.resume} with AUC {best_auc:.4f}. Resuming from epoch {start_epoch}.")

    Path(args.out).parent.mkdir(exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):
            x, y = x.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = loss_fn(logits, y) / args.accum_steps
            
            scaler.scale(loss).backward()
            
            if i % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # Validation
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_gts.extend(y.numpy().tolist())
        
        auc = roc_auc_score(all_gts, all_preds)
        print(f"Epoch {epoch}/{args.epochs} | Val ROC-AUC: {auc:.4f} (Best: {best_auc:.4f})")

        if auc > best_auc:
            best_auc = auc
            checkpoint_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "feature_dim": feature_dim,
                "model_name": model_name,
                "seq_len": seq_len,
                "best_auc": best_auc
            }
            torch.save(checkpoint_data, args.out)
            print(f"[SAVE] New best model saved to {args.out} with AUC {best_auc:.4f}") 