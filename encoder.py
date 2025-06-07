"""Neural encoder for mirror learning."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# --- Check for GPU ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
using_gpu = torch.cuda.is_available()
print(f"[INFO] Using device: {'GPU' if using_gpu else 'CPU'}")

# Initialize dimensions
DEFAULT_INPUT_DIM = 3 * 32 * 32  # Default for 32x32 RGB images
DEFAULT_HIDDEN_DIM = 256
DEFAULT_LATENT_DIM = 128


class MirrorNet(nn.Module):
    """Neural network for mirror learning."""

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 latent_dim: int = DEFAULT_LATENT_DIM):
        super(MirrorNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through the network."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def initialize_mirror_network():
    """Initialize the mirror network with appropriate dimensions."""
    if os.path.exists("pca_features.npy"):
        try:
            features = np.load("pca_features.npy")
            if features.size == 0 or len(features.shape) < 2:
                print(
                    "[INFO] PCA features file exists but is empty or invalid. Using default dimensions.")
                input_dim = DEFAULT_INPUT_DIM
                features = None
            else:
                input_dim = features.shape[1]
                features = torch.tensor(
                    features, dtype=torch.float32).to(device)
                print(
                    f"[INFO] Loaded PCA features with shape: {features.shape}")
        except Exception as e:
            print(
                f"[INFO] Error loading PCA features: {e}. Using default dimensions.")
            input_dim = DEFAULT_INPUT_DIM
            features = None
    else:
        input_dim = DEFAULT_INPUT_DIM
        features = None
        print(
            "[INFO] No existing features found. Using default input dimension:", input_dim)

    # Initialize model, optimizer, and loss function
    mirrornet = MirrorNet(input_dim).to(device)
    optimizer = optim.Adam(mirrornet.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train if features exist
    if features is not None and features.size(0) > 0:
        print("[INFO] Training started on", device)
        n_epochs = 20
        batch_size = 32

        for epoch in range(n_epochs):
            running_loss = 0.0

            # Create progress bar for batches
            n_batches = len(features) // batch_size
            if n_batches == 0:
                n_batches = 1
                batch_size = len(features)

            pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{n_epochs}')

            for i in pbar:
                # Get batch
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(features))
                batch = features[start_idx:end_idx]

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                reconstructed, _ = mirrornet(batch)

                # Compute loss
                loss = criterion(reconstructed, batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update statistics
                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{running_loss/(i+1):.6f}'})

            print(f'[EPOCH {epoch+1}] Loss: {running_loss/n_batches:.6f}')
    else:
        print(
            "[INFO] No features available for training. Network initialized with random weights.")

    print("[INFO] Mirror network ready for inference")
    return mirrornet


# Initialize the network
mirrornet = initialize_mirror_network()
