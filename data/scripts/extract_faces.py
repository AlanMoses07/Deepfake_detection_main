# data/scripts/extract_faces.py
import os
import cv2
import argparse
import torch
from pathlib import Path
from facenet_pytorch import MTCNN
from tqdm import tqdm

def process_videos_in_dir(videos_dir, out_dir, mtcnn, fps=2.0, max_frames=60):
    """
    Detects faces in all videos within a directory and saves the crops.
    """
    videos = sorted(list(Path(videos_dir).glob("*.mp4")))
    print(f"Found {len(videos)} videos in '{videos_dir}'. Starting extraction...")

    for video_path in tqdm(videos, desc=f"Processing videos in {os.path.basename(videos_dir)}"):
        video_name = video_path.stem
        save_dir = Path(out_dir) / video_name
        
        # Skip if already processed
        if save_dir.exists() and len(list(save_dir.glob('*.jpg'))) > 0:
            print(f"\n[SKIP] '{video_name}' already processed.")
            continue
        
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"\n[ERROR] Could not open video: {video_path}")
            continue

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_step = max(1, int(vid_fps // fps))
        
        frame_idx = 0
        saved_count = 0

        while saved_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                boxes, _ = mtcnn.detect(frame)
                if boxes is not None:
                    # Select the largest face
                    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    face = frame[max(0, y1):y2, max(0, x1):x2]
                    
                    if face.size > 0:
                        face_resized = cv2.resize(face, (224, 224))
                        save_path = save_dir / f"frame_{saved_count:04d}.jpg"
                        cv2.imwrite(str(save_path), face_resized)
                        saved_count += 1
            
            frame_idx += 1
        
        cap.release()
        if saved_count > 0:
             print(f"\n[OK] Saved {saved_count} faces for '{video_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract face crops from videos.")
    parser.add_argument("--videos_root", type=str, default="data/videos", help="Root directory containing 'real' and 'fake' video subfolders.")
    parser.add_argument("--faces_root", type=str, default="data/faces", help="Root directory to save face crops.")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample from video.")
    parser.add_argument("--max_frames", type=int, default=60, help="Maximum number of face crops to save per video.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    mtcnn = MTCNN(keep_all=True, device=device)

    # Process both real and fake videos
    process_videos_in_dir(os.path.join(args.videos_root, 'real'), os.path.join(args.faces_root, 'real'), mtcnn, args.fps, args.max_frames)
    process_videos_in_dir(os.path.join(args.videos_root, 'fake'), os.path.join(args.faces_root, 'fake'), mtcnn, args.fps, args.max_frames)
    
    print("\nExtraction complete!")