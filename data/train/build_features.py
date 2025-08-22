# data/train/build_features.py
import argparse
import os
import numpy as np
import torch
import timm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def load_model(model_name, device):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.eval().to(device)
    
    # Get model's feature dimension
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    return model, output.shape[-1]

def process_video_crops(video_dir, out_path, model, transform, device, batch_size=32, use_amp=False):
    image_paths = sorted(list(video_dir.glob("*.jpg")))
    if not image_paths:
        return 0

    features_list = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch_tensor = torch.stack(batch_images).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                features = model(batch_tensor).float()
            
            features_list.append(features.cpu().numpy())
    
    if features_list:
        all_features = np.concatenate(features_list, axis=0).astype(np.float32)
        np.save(out_path, all_features)
        return len(all_features)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature embeddings from face crops.")
    parser.add_argument("--faces_root", default="data/faces", help="Directory with face crops.")
    parser.add_argument("--features_root", default="data/features", help="Directory to save feature embeddings.")
    parser.add_argument("--model", default="convnext_tiny", help="Name of the timm model to use for feature extraction.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP).")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor, feature_dim = load_model(args.model, device)
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for cls_type in ["real", "fake"]:
        input_cls_dir = Path(args.faces_root) / cls_type
        output_cls_dir = Path(args.features_root) / cls_type
        output_cls_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = [d for d in input_cls_dir.iterdir() if d.is_dir()]
        
        for video_dir in tqdm(video_dirs, desc=f"Building features for {cls_type} videos"):
            output_path = output_cls_dir / f"{video_dir.name}.npy"
            if output_path.exists():
                continue
            
            process_video_crops(video_dir, output_path, feature_extractor, data_transform, device, args.batch_size, args.amp)

    # Save metadata for later steps
    (Path(args.features_root) / "model_name.txt").write_text(args.model)
    (Path(args.features_root) / "feature_dim.txt").write_text(str(feature_dim))
    
    print("\nFeature building complete!")