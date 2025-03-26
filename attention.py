import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Load MirrorNet latents (sequence of vectors)
latents_path = "mirrornet_latents.npy"
latents = np.load(latents_path)  # [N, D]

# Convert to tensor and reshape for attention: [B, T, D]
# For simplicity, we consider B=1 (single sequence), T=N
latents_tensor = torch.tensor(latents, dtype=torch.float32).unsqueeze(0)  # [1, T, D]

# MirrorAttentionBlock: self-attention over latent sequence
class MirrorAttentionBlock(nn.Module):
    def __init__(self, input_dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, x):
        # x: [B, T, D]
        attn_out, _ = self.attn(x, x, x)
        out = self.ff(attn_out + x)  # Residual connection
        return out

# Initialize model
input_dim = latents_tensor.shape[2]
attention_block = MirrorAttentionBlock(input_dim)

# Run attention
attention_block.eval()
with torch.no_grad():
    attended_output = attention_block(latents_tensor)  # [1, T, D]
    attended_output = attended_output.squeeze(0).numpy()

# Save attended output
np.save("mirror_attention_output.npy", attended_output)
print("[INFO] Saved attended latent representations to 'mirror_attention_output.npy'")