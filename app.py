import streamlit as st
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import io
import matplotlib
import yt_dlp
import subprocess
import sys
import threading
import queue
import hashlib

matplotlib.use('Agg')
st.set_page_config(page_title="Mirror Modeling Dashboard", layout="wide")
st.title("🧠 Mirror Modeling Dashboard")

VECTORS_DIR = "vectors"

# --- Video Download + Setup de Diretório ---
st.sidebar.subheader("📥 Add New Video")
yt_url = st.sidebar.text_input("YouTube video URL")

if st.sidebar.button("Download Video"):
    try:
        video_output_dir = "data/videos"
        shutil.rmtree(video_output_dir, ignore_errors=True)
        os.makedirs(video_output_dir, exist_ok=True)

        # Etapa 1: obter o título do vídeo sem baixar ainda
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(yt_url, download=False)
            video_title = info['title']
            hash_id = hashlib.sha256(video_title.encode("utf-8")).hexdigest()[:8]
            video_hash_name = f"v{hash_id}"
            video_filename = f"{video_hash_name}.mp4"

        # Etapa 2: configurar saída forçada com o nome hash
        ydl_opts = {
            'outtmpl': os.path.join(video_output_dir, video_filename),
            'format': 'mp4/bestaudio/best',
            'quiet': True
        }

        # Etapa 3: baixar com o nome correto
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([yt_url])

        # Criar pasta para os vetores
        vector_path = os.path.join(VECTORS_DIR, video_hash_name)
        os.makedirs(vector_path, exist_ok=True)

        # Guardar no session_state
        st.session_state["video_hash_name"] = video_hash_name
        st.session_state["video_title"] = video_title

        st.sidebar.success(f"Video downloaded as {video_filename} ✅")

    except Exception as e:
        st.sidebar.error(f"Failed to download video: {e}")

        
# --- Run .py Scripts with live output ---
def run_script_live(script_name: str):
    st.sidebar.info(f"Running {script_name}...")
    with st.spinner(f"Processing with {script_name}..."):
        with st.expander(f"🔄 {script_name} logs (click to expand)", expanded=True):
            process = subprocess.Popen(["python", script_name],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       universal_newlines=True,
                                       bufsize=1)
            output_area = st.empty()
            q = queue.Queue()
            def stream_output():
                for line in iter(process.stdout.readline, ''):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    q.put(line.strip())
            thread = threading.Thread(target=stream_output)
            thread.start()
            while thread.is_alive() or not q.empty():
                try:
                    last_line = q.get(timeout=0.1)
                    output_area.code(last_line, language="bash")
                except queue.Empty:
                    continue
            process.wait()
            thread.join()

    if process.returncode == 0:
        st.sidebar.success(f"{script_name} executed successfully ✅")
    else:
        st.sidebar.error(f"{script_name} execution failed ❌")


# --- Pipeline Buttons ---
st.sidebar.subheader("🧪 Run Full Pipeline")
scripts = ["mirror.py", "encoder.py", "attention.py", "self.py", "fusion.py"]
for script in scripts:
    if st.sidebar.button(f"▶️ Run {script}"):
        run_script_live(script)

if st.sidebar.button("🚀 Run Full Pipeline"):
    for script in scripts:
        run_script_live(script)

    # Após o pipeline, mova os vetores
    try:
        if "video_hash_name" in st.session_state:
            folder_id = hash_and_store_vectors(st.session_state["video_hash_name"])
            st.sidebar.success(f"Vectors saved in vectors/{folder_id} ✅")
        else:
            st.sidebar.warning("No video title found in session. Please redownload the video.")
    except Exception as e:
        st.sidebar.error(f"Failed to move vectors: {e}")
        
        

# --- Utility: PCA Plotting ---
def plot_pca_scatter(proj, labels, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(proj)):
        ax.scatter(proj[i, 0], proj[i, 1], s=120, edgecolors='black')
        ax.text(proj[i, 0] + 0.2, proj[i, 1] + 0.2, labels[i], fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.image(Image.open(buf), caption=title, use_container_width=True)
    plt.close(fig)

# --- Global Self Comparison ---
st.subheader("🧭 Self Vector Comparison (All Videos)")
try:
    all_vectors, all_labels = [], []
    for subdir in sorted(os.listdir(VECTORS_DIR)):
        fpath = os.path.join(VECTORS_DIR, subdir, "self_reference_vector.npy")
        if os.path.exists(fpath):
            vec = np.load(fpath).squeeze()
            all_vectors.append(vec)
            all_labels.append(subdir)
    if len(all_vectors) >= 2:
        z_proj = PCA(n_components=2).fit_transform(np.stack(all_vectors))
        plot_pca_scatter(z_proj, all_labels, "PCA of Self Vectors")
    elif len(all_vectors) == 1:
        st.info("Only one vector found. PCA not applicable.")
except Exception as e:
    st.warning(f"Could not load self vectors: {e}")

# --- Feature Abstraction Evolution for Latest ---
st.subheader("🔍 Feature Abstraction Evolution (Latest)")
try:
    latest_video = sorted(os.listdir(VECTORS_DIR))[-1]
    latest_path = os.path.join(VECTORS_DIR, latest_video)
    features = {
        "pca_features.npy": "Raw Perception",
        "mirrornet_latents.npy": "Compressed Latents",
        "mirror_attention_output.npy": "Attended Latents"
    }
    vecs, labels = [], []
    for fname, label in features.items():
        fpath = os.path.join(latest_path, fname)
        if os.path.exists(fpath):
            arr = np.load(fpath)
            vecs.append(np.mean(arr, axis=0) if arr.ndim == 2 else arr)
            labels.append(label)
    min_len = min(len(v) for v in vecs)
    stacked = np.stack([v[:min_len] for v in vecs])
    proj = PCA(n_components=2).fit_transform(stacked)
    plot_pca_scatter(proj, labels, f"Feature Abstraction Trajectory ({latest_video})")
except Exception as e:
    st.warning(f"Could not visualize evolution: {e}")

# --- Self vs Encoded Abstractions for Latest ---
st.subheader("🧠 Self vs Encoded Abstractions (Latest)")
try:
    base = os.path.join(VECTORS_DIR, latest_video)
    raw = np.load(os.path.join(base, "pca_features.npy")).mean(axis=0)
    latent = np.load(os.path.join(base, "mirrornet_latents.npy")).mean(axis=0)
    z_self = np.load(os.path.join(base, "self_reference_vector.npy")).squeeze()
    min_len = min(len(raw), len(latent), len(z_self))
    stacked = np.stack([raw[:min_len], latent[:min_len], z_self[:min_len]])
    proj = PCA(n_components=2).fit_transform(stacked)
    labels = ["Raw Perception", "MirrorNet Latents", "Self Vector"]
    colors = ["blue", "orange", "purple"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(3):
        ax.scatter(proj[i, 0], proj[i, 1], color=colors[i], label=labels[i], s=100, edgecolors="black")
        ax.text(proj[i, 0] + 0.3, proj[i, 1] + 0.3, labels[i], fontsize=9)
    ax.set_title(f"Self vs Encoded Abstractions ({latest_video})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not generate comparison plot: {e}")

def hash_and_store_vectors(video_hash_name: str):
    """
    Move arquivos .npy e pca_visualization.png para vectors/v<hash>
    """
    folder_name = video_hash_name
    target_dir = os.path.join("vectors", folder_name)
    os.makedirs(target_dir, exist_ok=True)

    files_to_move = [
        "pca_features.npy",
        "mirrornet_latents.npy",
        "mirror_attention_output.npy",
        "self_reference_vector.npy",
        "fused_consciousness_vectors.npy",
        "pca_visualization.png"
    ]

    moved = []
    for fname in files_to_move:
        if os.path.exists(fname):
            shutil.move(fname, os.path.join(target_dir, fname))
            moved.append(fname)

    print(f"[INFO] Moved to vectors/{folder_name}: {moved}")
    return folder_name


