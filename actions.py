import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import os
import numpy as np

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

# 3D CNN for Action Recognition
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
        # After two poolings: D=16 → 4, H/W=32 → 8
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Data transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Load dataset
video_dir = 'data/videos'
football_dataset = FootballVideoDataset(video_dir, transform=transform, max_frames=16)
trainloader = DataLoader(football_dataset, batch_size=2, shuffle=True)

# Training function with explainers
def train_network(net, dataloader, epochs=5):
    print(f"\n[INFO] Starting training for {epochs} epoch(s)...\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.squeeze(1)  # Remove extra wrapper dim if batch_size=1
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[EPOCH {epoch+1}/{epochs}] Average Loss: {running_loss / len(dataloader):.4f}")

    print("\n[INFO] Training complete.\n")

# Instantiate and train the perception network
perception_net = PerceptionNet()
print("[INFO] Model architecture:")
print(perception_net)

train_network(perception_net, trainloader)

print("[INFO] Perception network trained on real football videos ✅")
