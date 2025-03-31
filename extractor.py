import torch
import clip
from PIL import Image
import os
import numpy as np
import cv2
from tqdm import tqdm

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --- Parameters ---
video_dir = 'data/videos'
output_file = 'clip_features.npy'
max_frames = 64
image_size = 224

# --- Load videos ---
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
print(f"[INFO] Found {len(video_files)} video(s) in {video_dir}")

clip_features = []

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    num_chunks = total_frames // max_frames
    print(f"[INFO] Processing '{video_file}' with {num_chunks} chunks...")

    for chunk_idx in tqdm(range(num_chunks), desc=f"Processing {video_file}"):
        frames = []
        start_frame = chunk_idx * max_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = preprocess(frame).unsqueeze(0).to(device)
            frames.append(frame)

        if len(frames) < max_frames:
            print(f"[WARNING] Chunk {chunk_idx} in {video_file} has too few frames ({len(frames)}).")
            continue

        # Stack all frames and pass them through CLIP
        frames_tensor = torch.cat(frames, dim=0)
        with torch.no_grad():
            clip_features_chunk = model.encode_image(frames_tensor).cpu().numpy()
        
        # Aggregate the embeddings by taking the mean
        clip_feature = np.mean(clip_features_chunk, axis=0)
        clip_features.append(clip_feature)

    cap.release()

# Save the CLIP features
clip_features = np.array(clip_features)
np.save(output_file, clip_features)
print(f"[INFO] Saved CLIP features to '{output_file}'")
