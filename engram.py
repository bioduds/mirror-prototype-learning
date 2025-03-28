# engram.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st

st.set_page_config(page_title="Engram Visualization", layout="wide")
st.title("🧬 Engram Space Viewer")

# --- Load fused data from snapshots ---
data_dir = "snapshots"

fused_vectors = []
labels = []
colors = []

for i, video in enumerate(sorted(os.listdir(data_dir))):
    fuse_path = os.path.join(data_dir, video, "fused_representation.npy")
    if os.path.exists(fuse_path):
        fused_vectors.append(np.load(fuse_path).squeeze())
        labels.append(video)
        colors.append(f"C{i}")

# --- PCA ---
all_fused = np.stack(fused_vectors)
pca = PCA(n_components=2)
fused_pca = pca.fit_transform(all_fused)

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(fused_pca)):
    ax.scatter(fused_pca[i, 0], fused_pca[i, 1], color=colors[i], label=labels[i], s=120, edgecolors='black')
    ax.text(fused_pca[i, 0] + 0.2, fused_pca[i, 1] + 0.2, labels[i], fontsize=9)

ax.set_title("Engram Space (PCA of [Self + Experience])")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(True)
ax.legend()

st.image(fig, caption="Engram Clustering", use_container_width=True)
