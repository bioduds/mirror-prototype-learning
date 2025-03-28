import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# --- Load temporal latent sequence from MirrorAttention output ---
attended_path = "mirror_attention_output.npy"
attended = np.load(attended_path)  # Shape: [T, D]
attended_tensor = torch.tensor(attended, dtype=torch.float32)  # Shape: [T, D]

class SelfReferentialNet(nn.Module):
    """
    GRU-based model that encodes a latent trajectory into a compressed self-representation,
    and decodes that representation to predict the final latent state.

    Args:
        input_dim (int): Dimension of input features (D).
        hidden_dim (int): Dimension of internal representation (z_self).
    
    Returns:
        Tuple[prediction, z_self]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T, D].

        Returns:
            prediction (torch.Tensor): Predicted final latent vector [B, D]
            z_self (torch.Tensor): Encoded self-representation [B, H]
        """
        _, hidden = self.encoder(x)  # hidden: [1, B, H]
        z_self = hidden.squeeze(0)   # [B, H]
        prediction = self.decoder(z_self)  # [B, D]
        return prediction, z_self

# --- Model instantiation ---
input_dim = attended_tensor.shape[1]
model = SelfReferentialNet(input_dim=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# --- Prepare sequences: predict z_{T} from [z_0, ..., z_{T-1}] ---
sequences = attended_tensor[:-1].unsqueeze(0)  # [1, T-1, D]
targets = attended_tensor[1:]                 # [T-1, D]
target_final = targets[-1].unsqueeze(0)       # [1, D]

# --- Training loop ---
epochs = 20
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    prediction, z_self = model(sequences)
    loss = loss_fn(prediction, target_final)
    loss.backward()
    optimizer.step()
    print(f"[EPOCH {epoch+1}] Loss: {loss.item():.6f}")

# --- Save the self vector (z_self) ---
z_self_np = z_self.detach().numpy()
np.save("self_reference_vector.npy", z_self_np)
print("[INFO] Self representation vector saved: self_reference_vector.npy")
