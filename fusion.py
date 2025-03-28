# fusion.py
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def load_vectors_from_base() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the self and experience vectors from the base directory ("."),
    where the latest .npy files are located after running the pipeline.
    
    Returns:
        tuple: (self_vectors, experience_vectors) as stacked torch tensors.
    """
    self_path = "self_reference_vector.npy"
    latent_path = "mirror_attention_output.npy"

    if not os.path.exists(self_path) or not os.path.exists(latent_path):
        raise FileNotFoundError("Required .npy files not found in the current directory.")

    try:
        z_self = np.load(self_path).squeeze()
        experience = np.load(latent_path)
        if experience.ndim == 2:
            experience = experience.mean(axis=0)  # Temporal aggregation
    except Exception as e:
        raise RuntimeError(f"Failed to load input vectors: {e}")

    # Return single-sample tensors with batch dimension
    return (
        torch.tensor(z_self[None, :], dtype=torch.float32),
        torch.tensor(experience[None, :], dtype=torch.float32)
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
        combined = torch.cat([self_vec, exp_vec], dim=1)
        return self.fuse(combined)

# --- Main ---
if __name__ == "__main__":
    try:
        self_vectors, experience_vectors = load_vectors_from_base()
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)

    input_dim = self_vectors.shape[1]
    fusion_model = FusionLayer(input_dim)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train for temporal consistency (only applies if more than one sample)
    fusion_model.train()
    if self_vectors.shape[0] > 1:
        for epoch in range(20):
            total_loss = 0.0
            for i in tqdm(range(len(self_vectors) - 1), desc=f"Epoch {epoch+1}/20"):
                a_t = fusion_model(self_vectors[i].unsqueeze(0), experience_vectors[i].unsqueeze(0))
                a_next = fusion_model(self_vectors[i+1].unsqueeze(0), experience_vectors[i+1].unsqueeze(0))
                loss = criterion(a_t, a_next.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[EPOCH {epoch+1}] Cohesion Loss: {total_loss / (len(self_vectors) - 1):.6f}")
    else:
        print("[INFO] Only one sample: skipping training loop.")

    # Inference
    fusion_model.eval()
    fused_representations = []
    with torch.no_grad():
        for i in range(len(self_vectors)):
            fused = fusion_model(self_vectors[i].unsqueeze(0), experience_vectors[i].unsqueeze(0))
            fused_representations.append(fused.squeeze(0).numpy())

    np.save("fused_consciousness_vectors.npy", np.array(fused_representations))
    print("[INFO] Saved fused consciousness representations ✅")
