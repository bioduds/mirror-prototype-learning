"""Perception module for processing video frames."""
import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from pca_processor import process_features


class PerceptionNet:
    """Neural network for visual perception."""

    def __init__(self, target_size: tuple = (32, 32)):
        self.target_size = target_size
        print("[INFO] PerceptionNet ready for feature extraction.")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame."""
        # Resize frame
        frame = cv2.resize(frame, self.target_size)
        # Normalize pixel values
        frame = frame.astype(np.float32) / 255.0
        # Flatten the frame
        return frame.flatten()

    def extract_features(self, video_paths: List[str], save_path: str = "pca_features.npy") -> Optional[np.ndarray]:
        """Extract features from multiple videos."""
        if not video_paths:
            print("[INFO] No videos to process.")
            return None

        all_features = []

        for video_path in video_paths:
            if not os.path.exists(video_path):
                print(f"[WARNING] Video not found: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count == 0:
                print(f"[WARNING] No frames in video: {video_path}")
                continue

            # Process frames with progress bar
            with tqdm(total=frame_count, desc=f"Extracting Features") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    features = self.preprocess_frame(frame)
                    all_features.append(features)
                    pbar.update(1)

            cap.release()

        if not all_features:
            print("[WARNING] No features extracted from any video.")
            return None

        # Convert to numpy array
        features_array = np.array(all_features)

        # Save raw features
        np.save(save_path, features_array)
        print(f"[INFO] Saved features to '{save_path}'")

        # Process through PCA if possible
        transformed_features, _ = process_features(features_array)
        return transformed_features
