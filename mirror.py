import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Custom Dataset for Video Frames
class FootballVideoDataset(Dataset):
    def __init__(self, video_dir, transform=None, max_frames=16):
        self.video_dir = video_dir
        self.transform = transform
        self.max_frames = max_frames
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"[INFO] Found {len(self.video_files)} video(s) in {video_dir}")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        print(f"[DATA] Loaded video: {video_path} with {len(frames)} frame(s)")

        if len(frames) < self.max_frames:
            raise ValueError(f"[WARNING] Video {video_path} has too few frames ({len(frames)}). Need at least {self.max_frames}.")

        frames = frames[:self.max_frames]  # Limit number of frames
        frames = torch.stack(frames)       # [D, C, H, W]
        frames = frames.permute(1, 0, 2, 3)  # [C, D, H, W]
        frames = frames.unsqueeze(0)         # [1, C, D, H, W]
        label = torch.tensor(0)  # Placeholder label
        return frames, label

# 3D CNN for Feature Extraction
class PerceptionNet(nn.Module):
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
        # Final layer removed for feature extraction

    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        return x  # Return features instead of classification

# Data transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load dataset
video_dir = 'data/videos'
football_dataset = FootballVideoDataset(video_dir, transform=transform, max_frames=16)
trainloader = DataLoader(football_dataset, batch_size=1, shuffle=False)

# Instantiate model for feature extraction
perception_net = PerceptionNet()
print("[INFO] PerceptionNet ready for feature extraction.")

# Extract features from each video
all_features = []
with torch.no_grad():
    for i, (inputs, _) in enumerate(trainloader):
        inputs = inputs.squeeze(1)
        features = perception_net(inputs)
        all_features.append(features.squeeze(0).numpy())
        print(f"[FEATURE] Extracted features from video {i+1}/{len(trainloader)}")

# Convert features to numpy array
all_features = np.array(all_features)
print("[INFO] Feature matrix shape:", all_features.shape)

# Apply PCA
print("[INFO] Performing PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(all_features)

# Plot the PCA result
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.7)
plt.title("PCA of Video Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_visualization.png")
print("[INFO] PCA plot saved as 'pca_visualization.png'")
