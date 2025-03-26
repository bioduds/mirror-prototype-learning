import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Load PerceptionNet PCA outputs
features_path = "pca_features.npy"
coords_path = "pca_coords.npy"
X_pca = np.load(coords_path)

# Optionally load MirrorNet latents (if they exist)
mirror_latents = None
mirror_latents_path = "mirrornet_latents.npy"
if os.path.exists(mirror_latents_path):
    mirror_latents = np.load(mirror_latents_path)

# --- Gradio UI Functions ---
def show_pca_plot():
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.6, label="PerceptionNet")
    if mirror_latents is not None:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latents_pca = pca.fit_transform(mirror_latents)
        ax.scatter(latents_pca[:, 0], latents_pca[:, 1], c='red', alpha=0.4, label="MirrorNet")
    ax.set_title("PCA: PerceptionNet & MirrorNet")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    return fig

def show_chunk(index):
    load_dataset()
    try:
        tensor, _ = dataset[int(index)]
        frames = tensor.squeeze().permute(1, 0, 2, 3)  # [D, C, H, W]
        first_frame = frames[0].permute(1, 2, 0).numpy()
        return (first_frame * 255).astype(np.uint8)
    except Exception as e:
        return f"Error loading chunk: {e}"

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Mirror Learning Dashboard")
    with gr.Row():
        plot = gr.Plot(label="PCA Plot")
        slider = gr.Slider(0, 238, step=1, label="Select Chunk")  # Default max index for now
    image = gr.Image(label="First Frame of Chunk")

    slider.change(fn=show_chunk, inputs=slider, outputs=image)
    demo.load(fn=show_pca_plot, outputs=plot)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)