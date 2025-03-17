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
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

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
        frames = torch.stack(frames)
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
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8 * 8, 128),
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
video_dir = 'path_to_videos'  # Replace with your directory path
football_dataset = FootballVideoDataset(video_dir, transform=transform)
trainloader = DataLoader(football_dataset, batch_size=2, shuffle=True)

def train_network(net, dataloader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Instantiate and train the perception network
perception_net = PerceptionNet()
train_network(perception_net, trainloader)

print("Perception network trained on real football videos.")
