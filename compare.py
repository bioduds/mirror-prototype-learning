import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
import io
from PIL import Image

# Use a non-interactive backend to avoid rendering errors
matplotlib.use('Agg')

# Load z_self vectors with squeeze to remove extra dimensions
z1 = np.load("snapshots/video1/self_reference_vector.npy").squeeze()
z2 = np.load("snapshots/video2/self_reference_vector.npy").squeeze()

# Stack and label
all_z = np.stack([z1, z2])
labels = ["Soccer (z_self_1)", "Formula 1 (z_self_2)"]
colors = ["lime", "magenta"]

# PCA projection
pca = PCA(n_components=2)
z_proj = pca.fit_transform(all_z)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(z_proj)):
    ax.scatter(z_proj[i, 0], z_proj[i, 1], c=colors[i], label=labels[i], edgecolors='black', s=140)
    ax.text(float(z_proj[i, 0]) + 0.01, float(z_proj[i, 1]) + 0.01, labels[i], fontsize=10)

ax.set_title("Comparison of z_self Vectors (Video 1 vs Video 2)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
ax.grid(True)

# Save using PIL as fallback to avoid matplotlib canvas issues
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = Image.open(buf)
image.save("compare_z_self.png")
buf.close()