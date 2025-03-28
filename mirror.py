import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

class VideoDataset(Dataset):
    """
    Dataset for loading video files and splitting them into frame chunks.
    Each chunk is returned as a tensor of shape [1, C, D, H, W].
    
    Args:
        video_dir (str): Path to the folder containing .mp4 videos.
        transform (callable, optional): Transformations to apply to each frame.
        max_frames (int): Number of frames per chunk.
    """
    def __init__(self, video_dir: str, transform=None, max_frames: int = 64):
        self.video_dir = video_dir
        self.transform = transform
        self.max_frames = max_frames
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"[INFO] Found {len(self.video_files)} video(s) in {video_dir}")

    def __len__(self) -> int:
        return sum(self._num_chunks(f) for f in self.video_files)

    def _num_chunks(self, filename: str) -> int:
        """Returns the number of chunks a video file can be split into."""
        cap = cv2.VideoCapture(os.path.join(self.video_dir, filename))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(0, total_frames // self.max_frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetches the idx-th chunk across all videos."""
        current = 0
        for fname in self.video_files:
            num_chunks = self._num_chunks(fname)
            if idx < current + num_chunks:
                chunk_idx = idx - current
                return self._get_chunk(fname, chunk_idx)
            current += num_chunks
        raise IndexError("Index out of range")

    def _get_chunk(self, filename: str, chunk_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts a specific chunk of frames from a given video file."""
        video_path = os.path.join(self.video_dir, filename)
        cap = cv2.VideoCapture(video_path)
        frames = []
        start_frame = chunk_idx * self.max_frames
        end_frame = start_frame + self.max_frames
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or current_frame >= end_frame:
                break
            if current_frame >= start_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            current_frame += 1
        cap.release()

        if len(frames) < self.max_frames:
            raise ValueError(f"[WARNING] Chunk {chunk_idx} in {filename} has too few frames ({len(frames)}).")

        frames = torch.stack(frames)       # [D, C, H, W]
        frames = frames.permute(1, 0, 2, 3)  # [C, D, H, W]
        frames = frames.unsqueeze(0)         # [1, C, D, H, W]
        label = torch.tensor(0)
        return frames, label

class PerceptionNet(nn.Module):
    """
    3D Convolutional Neural Network for extracting spatiotemporal features
    from a chunk of video frames.
    """
    def __init__(self):
        super(PerceptionNet, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        return x

# --- Data transformation pipeline ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# --- Load video dataset ---
video_dir = 'data/videos'
dataset = VideoDataset(video_dir, transform=transform)
trainloader = DataLoader(dataset, batch_size=1, shuffle=False)

# --- Initialize perception model ---
perception_net = PerceptionNet()
print("[INFO] PerceptionNet ready for feature extraction.")

# --- Extract features ---
all_features = []
with torch.no_grad():
    for i, (inputs, _) in enumerate(tqdm(trainloader, desc="Extracting Features")):
        inputs = inputs.squeeze(1)
        features = perception_net(inputs)
        all_features.append(features.squeeze(0).numpy())

# --- Save raw feature vectors ---
all_features = np.array(all_features)
np.save("pca_features.npy", all_features)
print("[INFO] Saved features to 'pca_features.npy'")

# --- Apply PCA for visualization ---
if all_features.shape[0] >= 2:
    print("[INFO] Performing PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(all_features)
    np.save("pca_coords.npy", X_pca)
    print("[INFO] Saved PCA coordinates to 'pca_coords.npy'")

    # Save PCA scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.7)
    plt.title("PCA of Video Chunks")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pca_visualization.png")
    print("[INFO] PCA plot saved as 'pca_visualization.png'")
else:
    print("[INFO] Not enough data points for PCA. Add more videos or longer ones.")
