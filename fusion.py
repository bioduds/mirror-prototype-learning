# fusion.py
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

# --- Load inputs ---
self_vectors = []
experience_vectors = []

snapshots_dir = "snapshots"
available_videos = sorted([name for name in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, name))])

for video in available_videos:
    video_path = os.path.join(snapshots_dir, video)
    self_path = os.path.join(video_path, "self_reference_vector.npy")
    latent_path = os.path.join(video_path, "mirror_attention_output.npy")

    if os.path.exists(self_path) and os.path.exists(latent_path):
        s = np.load(self_path).squeeze()
        x = np.load(latent_path).mean(axis=0)  # Aggregate latent experience
        self_vectors.append(s)
        experience_vectors.append(x)

self_vectors = torch.tensor(np.stack(self_vectors), dtype=torch.float32)
experience_vectors = torch.tensor(np.stack(experience_vectors), dtype=torch.float32)

# --- Fusion Module ---
class FusionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, self_vec, exp_vec):
        combined = torch.cat([self_vec, exp_vec], dim=1)
        return self.fuse(combined)

# --- Train fusion model to ensure temporal cohesion ---
fusion_model = FusionLayer(self_vectors.shape[1])
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

fusion_model.train()
for epoch in range(20):
    total_loss = 0.0
    for i in tqdm(range(len(self_vectors) - 1), desc=f"Epoch {epoch+1}/20"):
        a_t = fusion_model(self_vectors[i].unsqueeze(0), experience_vectors[i].unsqueeze(0))
        a_next = fusion_model(self_vectors[i+1].unsqueeze(0), experience_vectors[i+1].unsqueeze(0))

        loss = criterion(a_t, a_next.detach())  # Encourage temporal cohesion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[EPOCH {epoch+1}] Cohesion Loss: {total_loss / (len(self_vectors) - 1):.6f}")

# --- Save fused representations ---
fusion_model.eval()
fused_representations = []
with torch.no_grad():
    for i in range(len(self_vectors)):
        fused = fusion_model(self_vectors[i].unsqueeze(0), experience_vectors[i].unsqueeze(0))
        fused_representations.append(fused.squeeze(0).numpy())

np.save("fused_consciousness_vectors.npy", np.array(fused_representations))
print("[INFO] Saved fused consciousness representations.")