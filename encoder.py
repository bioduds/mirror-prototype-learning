import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# --- Load extracted visual features from PerceptionNet ---
features_path = "pca_features.npy"
features = np.load(features_path)
features = torch.tensor(features, dtype=torch.float32)

class MirrorNet(nn.Module):
    """
    Autoencoder model for compressing high-dimensional perception features.

    Args:
        input_dim (int): Dimension of the input feature vector.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Reconstructed input and latent code.
    """
    def __init__(self, input_dim: int):
        super(MirrorNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# --- Initialize model, optimizer, and loss function ---
input_dim = features.shape[1]
mirrornet = MirrorNet(input_dim)
optimizer = optim.Adam(mirrornet.parameters(), lr=0.001)
criterion = nn.MSELoss()

# --- Training loop ---
epochs = 20
mirrornet.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for i in tqdm(range(len(features)), desc=f"Epoch {epoch+1}/{epochs}"):
        input_vec = features[i].unsqueeze(0)
        optimizer.zero_grad()
        output, _ = mirrornet(input_vec)
        loss = criterion(output, input_vec)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"[EPOCH {epoch+1}] Loss: {epoch_loss / len(features):.6f}")

# --- Inference: Encode all features ---
mirrornet.eval()
with torch.no_grad():
    reconstructed = []
    latents = []
    for i in range(len(features)):
        out, z = mirrornet(features[i].unsqueeze(0))
        reconstructed.append(out.squeeze(0).numpy())
        latents.append(z.squeeze(0).numpy())

# --- Save outputs ---
np.save("mirrornet_reconstructed.npy", np.array(reconstructed))
np.save("mirrornet_latents.npy", np.array(latents))
print("[INFO] Saved mirror network outputs: reconstructed + latents")
