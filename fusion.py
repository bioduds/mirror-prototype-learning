# fusion.py
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --- Load inputs ---
def load_vectors(snapshots_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load z_self vectors and average attended representations from video snapshots.

    Args:
        snapshots_dir (str): Directory containing processed video folders.

    Returns:
        tuple: (self_vectors, experience_vectors) as stacked torch tensors.
    """
    self_vectors = []
    experience_vectors = []

    available_videos = sorted([
        name for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ])

    for video in available_videos:
        video_path = os.path.join(snapshots_dir, video)
        self_path = os.path.join(video_path, "self_reference_vector.npy")
        latent_path = os.path.join(video_path, "mirror_attention_output.npy")

        if os.path.exists(self_path) and os.path.exists(latent_path):
            z_self = np.load(self_path).squeeze()
            experience = np.load(latent_path).mean(axis=0)  # Aggregate temporal info
            self_vectors.append(z_self)
            experience_vectors.append(experience)

    return (
        torch.tensor(np.stack(self_vectors), dtype=torch.float32),
        torch.tensor(np.stack(experience_vectors), dtype=torch.float32)
    )

class FusionLayer(nn.Module):
    """
    Neural layer to fuse self and experience representations into a unified vector.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, self_vec: torch.Tensor, exp_vec: torch.Tensor) -> torch.Tensor:
        """
        Concatenate and fuse self and experience into a single representation.

        Args:
            self_vec (torch.Tensor): [B, D] self vector
            exp_vec (torch.Tensor): [B, D] experience vector

        Returns:
            torch.Tensor: [B, 128] fused consciousness vector
        """
        combined = torch.cat([self_vec, exp_vec], dim=1)
        return self.fuse(combined)

# --- Main execution ---
if __name__ == "__main__":
    snapshots_dir = "snapshots"
    self_vectors, experience_vectors = load_vectors(snapshots_dir)

    # Initialize model
    input_dim = self_vectors.shape[1]
    fusion_model = FusionLayer(input_dim)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train to encourage temporal cohesion
    fusion_model.train()
    for epoch in range(20):
        total_loss = 0.0
        for i in tqdm(range(len(self_vectors) - 1), desc=f"Epoch {epoch+1}/20"):
            a_t = fusion_model(self_vectors[i].unsqueeze(0), experience_vectors[i].unsqueeze(0))
            a_next = fusion_model(self_vectors[i+1].unsqueeze(0), experience_vectors[i+1].unsqueeze(0))

            loss = criterion(a_t, a_next.detach())  # Temporal consistency
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[EPOCH {epoch+1}] Cohesion Loss: {total_loss / (len(self_vectors) - 1):.6f}")

    # Inference & Save
    fusion_model.eval()
    fused_representations = []
    with torch.no_grad():
        for i in range(len(self_vectors)):
            fused = fusion_model(self_vectors[i].unsqueeze(0), experience_vectors[i].unsqueeze(0))
            fused_representations.append(fused.squeeze(0).numpy())

    np.save("fused_consciousness_vectors.npy", np.array(fused_representations))
    print("[INFO] Saved fused consciousness representations.")
