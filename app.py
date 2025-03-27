import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

st.set_page_config(page_title="Mirror Modeling Dashboard", layout="wide")
st.title("🧠 Mirror Modeling Dashboard")

# --- Sidebar ---
st.sidebar.header("📂 Video Selection")

# Scan available video snapshots
data_dir = "snapshots"
available_videos = sorted([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))])
selected_video = st.sidebar.selectbox("Choose a processed video:", available_videos)

# Input for new video URL
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Add New Video")
yt_url = st.sidebar.text_input("YouTube video URL")
if st.sidebar.button("Download & Process Video"):
    st.sidebar.warning("This feature will be implemented in the backend. 🚧")
    # Here we'll call the pipeline script and update data

# --- Main Area ---
col1, col2 = st.columns([2, 1])

# --- Load and show PCA comparison ---
st.subheader("🔍 Comparison of Self Representations")

z_self_paths = []
labels = []
colors = []

for i, video in enumerate(available_videos):
    path = os.path.join(data_dir, video, "self_reference_vector.npy")
    if os.path.exists(path):
        z_self_paths.append(path)
        labels.append(video)
        colors.append("C" + str(i))  # C0, C1, ...

z_data = [np.load(path).squeeze() for path in z_self_paths]
all_z = np.stack(z_data)

# Apply PCA
pca = PCA(n_components=2)
z_proj = pca.fit_transform(all_z)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
for i in range(len(z_proj)):
    ax.scatter(z_proj[i, 0], z_proj[i, 1], color=colors[i], label=labels[i], s=120, edgecolors='black')
    ax.text(z_proj[i, 0] + 0.2, z_proj[i, 1] + 0.2, labels[i], fontsize=9)

ax.set_title("PCA of Self Vectors")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Show basic info for selected video ---
st.markdown(f"### 📽️ Details for: `{selected_video}`")
video_dir = os.path.join(data_dir, selected_video)

# Try to show first frame
try:
    npy_path = os.path.join(video_dir, "mirror_attention_output.npy")
    latent = np.load(npy_path)
    st.write(f"Loaded latent shape: {latent.shape}")
except Exception as e:
    st.warning(f"Could not load data for {selected_video}: {e}")
