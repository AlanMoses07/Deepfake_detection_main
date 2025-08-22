# predict_video.py
import argparse
import cv2
import torch
import timm
import numpy as np
from pathlib import Path
from torchvision import transforms
import sys

# --- FIX: Add project root to Python path ---
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
# -----------------------------------------

from facenet_pytorch import MTCNN
from data.train.train_lstm import LSTMDetector # Reuse the model class

def extract_faces_from_video(video_path, mtcnn, max_frames=60, fps=2.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_step = max(1, int(vid_fps // fps))
    
    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                x1, y1, x2, y2 = [int(c) for c in box]
                face = frame[max(0, y1):y2, max(0, x1):x2]
                if face.size > 0:
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    frames.append(face_rgb)
        frame_idx += 1
    
    cap.release()
    return frames

def build_features_from_faces(face_frames, feature_extractor, transform, device):
    if not face_frames:
        return np.array([])

    batch_tensors = [transform(f).unsqueeze(0) for f in face_frames]
    input_tensor = torch.cat(batch_tensors).to(device)
    
    with torch.no_grad():
        features = feature_extractor(input_tensor).cpu().numpy()
    
    return features

def predict_from_features(features, model, seq_len, device):
    if len(features) < seq_len:
        if len(features) == 0: return 0.5 # Default to uncertain
        repeats = (seq_len + len(features) - 1) // len(features)
        features = np.tile(features, (repeats, 1))

    # Use sliding windows for a more robust prediction
    predictions = []
    with torch.no_grad():
        for i in range(0, len(features) - seq_len + 1, seq_len // 2):
            # --- THIS IS THE FIXED LINE ---
            # The input tensor must be moved to the same device as the model
            sequence = torch.tensor(features[i : i + seq_len]).unsqueeze(0).to(device)
            # --------------------------
            logit = model(sequence)
            prob = torch.sigmoid(logit).item()
            predictions.append(prob)
            
    return np.mean(predictions) if predictions else 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if a video is a deepfake.")
    parser.add_argument("video_path", type=str, help="Path to the video file to predict.")
    parser.add_argument("--ckpt", default="data/checkpoints/best_model.pt", help="Path to the model checkpoint.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device)

    # Load checkpoint and metadata
    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = ckpt['model_name']
    feature_dim = ckpt['feature_dim']
    seq_len = ckpt['seq_len']

    # Load feature extractor and detector models
    feature_extractor = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    detector = LSTMDetector(feature_dim).eval().to(device)
    detector.load_state_dict(ckpt['model'])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Full Pipeline ---
    print(f"1. Extracting faces from '{args.video_path}'...")
    face_frames = extract_faces_from_video(args.video_path, mtcnn)

    print(f"2. Building features using '{model_name}'...")
    features = build_features_from_faces(face_frames, feature_extractor, transform, device)

    print(f"3. Predicting with LSTM detector...")
    fake_probability = predict_from_features(features, detector, seq_len, device)
    
    prediction = "FAKE" if fake_probability > 0.5 else "REAL"
    
    print("\n--- Prediction Result ---")
    print(f"Video: {args.video_path}")
    print(f"Predicted as: {prediction}")
    print(f"Fake Probability: {fake_probability:.2%}")
    print("-------------------------")