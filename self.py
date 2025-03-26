import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Load attention output (latent trajectory)
attended_path = "mirror_attention_output.npy"
attended = np.load(attended_path)  # Shape: [T, D]
attended_tensor = torch.tensor(attended, dtype=torch.float32)  # [T, D]

# --- SelfReferentialNet ---
class SelfReferentialNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [B, T, D] → hidden: [1, B, H]
        _, hidden = self.encoder(x)
        z_self = hidden.squeeze(0)  # [B, H]
        prediction = self.decoder(z_self)  # [B, D]
        return prediction, z_self

# Instantiate model
input_dim = attended_tensor.shape[1]
model = SelfReferentialNet(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Prepare input/target pairs (z_t → z_{t+1})
sequences = attended_tensor[:-1].unsqueeze(0)  # [1, T-1, D]
targets = attended_tensor[1:]  # [T-1, D]

# Training loop
epochs = 20
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    pred, z_self = model(sequences)
    loss = loss_fn(pred, targets[-1].unsqueeze(0))  # Last target
    loss.backward()
    optimizer.step()
    print(f"[EPOCH {epoch+1}] Loss: {loss.item():.6f}")

# Save z_self vector
z_self_np = z_self.detach().numpy()
np.save("self_reference_vector.npy", z_self_np)
print("[INFO] Self representation vector saved: self_reference_vector.npy")
