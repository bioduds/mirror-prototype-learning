import streamlit as st
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from PIL import Image
import io
import matplotlib
import yt_dlp
import subprocess
import sys
import threading
import queue
import hashlib
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from fpdf import FPDF
from datetime import datetime



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




############################################
############################################
# --- Análise Estatística e Visualização ---
############################################
############################################

st.sidebar.subheader("📊 Analysis Tools")
analyze_all = st.sidebar.checkbox("Analyze All Vectors", value=True)
selected_analysis = st.sidebar.selectbox("Select Analysis Type", [
    "Statistical Summary", "Correlation Analysis", "Clustering", 
    "Dimensionality Reduction", "Comparative Analysis"
])

def load_vectors():
    vectors = {}
    for subdir in sorted(os.listdir(VECTORS_DIR)):
        vector_path = os.path.join(VECTORS_DIR, subdir, "fused_consciousness_vectors.npy")
        
        if os.path.exists(vector_path):
            try:
                data = np.load(vector_path, allow_pickle=True)
                
                # Check if data is non-empty
                if isinstance(data, np.ndarray) and len(data) > 0:
                    vectors[subdir] = data
                else:
                    st.warning(f"[WARNING] {subdir} contains an empty or invalid .npy file and will be skipped.")
            
            except Exception as e:
                st.warning(f"[ERROR] Failed to load {vector_path}: {e}")
    
    if not vectors:
        st.error("No valid vectors found. Please check your .npy files or run the pipeline again.")
        
    return vectors


vectors = load_vectors()

# --- Estatísticas Básicas ---
def statistical_summary(vectors):
    st.subheader("📈 Statistical Summary")
    summaries = []
    for name, data in vectors.items():
        df = pd.DataFrame(data)
        summary = df.describe().T
        summary["Video"] = name
        summaries.append(summary)
    
    full_summary = pd.concat(summaries)
    st.write(full_summary)
    
    st.subheader("📊 Aggregated Histogram Visualization")
    
    # Prepare data for histogram comparison
    all_data = []
    for name, data in vectors.items():
        flattened_data = np.concatenate(data)
        all_data.append(pd.DataFrame({"Values": flattened_data, "Vector Name": name}))
        
    combined_df = pd.concat(all_data)
    
    # Plot histogram overlays
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=combined_df, x="Values", hue="Vector Name", element="step", stat="density", common_norm=False, ax=ax)
    plt.title("Overlaid Histograms of All Vectors")
    st.pyplot(fig)

    # Plot Kernel Density Estimation (KDE)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=combined_df, x="Values", hue="Vector Name", common_norm=False, ax=ax)
    plt.title("Kernel Density Estimation (KDE) Comparison")
    st.pyplot(fig)

    # Plot Heatmap of Histogram Bins
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_data = []
    bins = np.linspace(combined_df['Values'].min(), combined_df['Values'].max(), 30)
    for name, group in combined_df.groupby('Vector Name'):
        hist_counts, _ = np.histogram(group['Values'], bins=bins)
        hist_data.append(hist_counts)
    
    sns.heatmap(np.array(hist_data), cmap="viridis", xticklabels=np.round(bins, 2), yticklabels=combined_df['Vector Name'].unique(), ax=ax)
    plt.title("Heatmap of Histogram Bins")
    plt.xlabel("Values")
    plt.ylabel("Vectors")
    st.pyplot(fig)

    # Plot Boxplot for statistical comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=combined_df, x="Values", y="Vector Name", orient="h", ax=ax)
    plt.title("Boxplot Comparison of Vectors")
    st.pyplot(fig)

# --- Correlação entre vetores ---
def correlation_analysis(vectors):
    st.subheader("🔗 Correlation Analysis")
    all_vectors = {name: np.mean(vec, axis=0) for name, vec in vectors.items()}
    all_vectors_df = pd.DataFrame(all_vectors)
    
    # Pearson Correlation
    corr = all_vectors_df.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Spearman Correlation
    corr_spearman = all_vectors_df.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_spearman, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- Clusterização com K-Means e DBSCAN ---
def clustering(vectors):
    st.subheader("🌌 Clustering Analysis")
    combined_vectors = np.concatenate([vec for vec in vectors.values()])
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(combined_vectors)
    dbscan = DBSCAN(eps=0.5).fit(combined_vectors)

    fig = px.scatter_3d(x=combined_vectors[:,0], y=combined_vectors[:,1], z=combined_vectors[:,2], 
                        color=kmeans.labels_, title="K-Means Clustering")
    st.plotly_chart(fig)

    fig = px.scatter_3d(x=combined_vectors[:,0], y=combined_vectors[:,1], z=combined_vectors[:,2], 
                        color=dbscan.labels_, title="DBSCAN Clustering")
    st.plotly_chart(fig)

# --- Redução Dimensional (PCA, TSNE, UMAP) ---
def dimensionality_reduction(vectors):
    st.subheader("📉 Dimensionality Reduction")
    combined_vectors = np.concatenate([vec for vec in vectors.values()])

    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(combined_vectors)
    
    fig = px.scatter(x=pca_proj[:, 0], y=pca_proj[:, 1], title="PCA Projection")
    st.plotly_chart(fig)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_proj = tsne.fit_transform(combined_vectors)
    
    fig = px.scatter(x=tsne_proj[:, 0], y=tsne_proj[:, 1], title="t-SNE Projection")
    st.plotly_chart(fig)

# --- Análise Comparativa ---
def comparative_analysis(vectors):
    st.subheader("📊 Comparative Analysis")
    all_vectors = {name: np.mean(vec, axis=0) if vec.ndim > 1 else vec for name, vec in vectors.items()}
    all_vectors_df = pd.DataFrame(all_vectors)
    
    fig = go.Figure()
    for name, vector in all_vectors.items():
        if isinstance(vector, np.ndarray) and vector.ndim > 0:  # Ensure vector is not a single float
            fig.add_trace(go.Scatter(x=np.arange(len(vector)), y=vector, mode='lines', name=name))
        else:
            st.warning(f"Vector {name} is not a valid array for plotting.")
    
    fig.update_layout(title="Comparative Analysis of Vectors", xaxis_title="Index", yaxis_title="Value")
    st.plotly_chart(fig)


# --- Mostrar Resultados ---
if selected_analysis == "Statistical Summary":
    statistical_summary(vectors)
elif selected_analysis == "Correlation Analysis":
    correlation_analysis(vectors)
elif selected_analysis == "Clustering":
    clustering(vectors)
elif selected_analysis == "Dimensionality Reduction":
    dimensionality_reduction(vectors)
elif selected_analysis == "Comparative Analysis":
    comparative_analysis(vectors)


import os
import numpy as np
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns

def generate_statistical_report(vectors):
    # Create a new PDF report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title of the report
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Mirror Modeling Statistical Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)

    # Table of Contents
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Table of Contents", ln=True)
    contents = [
        "1. Statistical Summary",
        "2. Correlation Analysis",
        "3. Clustering Analysis",
        "4. Dimensionality Reduction",
        "5. Comparative Analysis",
        "6. Summary and Conclusion"
    ]
    for item in contents:
        pdf.cell(0, 10, item, ln=True)
    pdf.ln(10)
    
    # Convert vectors to a dataframe for analysis
    combined_vectors = np.concatenate([vec for vec in vectors.values()])
    all_vectors_df = pd.DataFrame({name: np.mean(vec, axis=0) for name, vec in vectors.items()})
    
    # 1. Statistical Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Statistical Summary", ln=True)
    pdf.set_font("Arial", size=10)
    summaries = [pd.DataFrame(data).describe().T.assign(Video=name) for name, data in vectors.items()]
    full_summary = pd.concat(summaries)
    pdf.multi_cell(0, 10, full_summary.to_string())
    pdf.ln(10)
    
    # 2. Correlation Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Correlation Analysis", ln=True)
    pdf.set_font("Arial", size=10)
    
    corr_pearson = all_vectors_df.corr(method='pearson')
    corr_spearman = all_vectors_df.corr(method='spearman')
    
    pdf.multi_cell(0, 10, "Pearson Correlation Matrix:\n" + corr_pearson.to_string())
    pdf.ln(5)
    pdf.multi_cell(0, 10, "Spearman Correlation Matrix:\n" + corr_spearman.to_string())
    pdf.ln(10)
    
    # 3. Clustering Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Clustering Analysis", ln=True)
    pdf.set_font("Arial", size=10)
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(combined_vectors)
    dbscan = DBSCAN(eps=0.5).fit(combined_vectors)
    
    # Plot K-Means Clustering
    plt.figure(figsize=(6, 4))
    plt.scatter(combined_vectors[:, 0], combined_vectors[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title('K-Means Clustering')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig("kmeans_plot.png")
    plt.close()
    pdf.image("kmeans_plot.png", x=10, w=180)
    
    # Plot DBSCAN Clustering
    plt.figure(figsize=(6, 4))
    plt.scatter(combined_vectors[:, 0], combined_vectors[:, 1], c=dbscan.labels_, cmap='plasma')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig("dbscan_plot.png")
    plt.close()
    pdf.image("dbscan_plot.png", x=10, w=180)
    pdf.ln(10)
    
    # 4. Dimensionality Reduction
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "4. Dimensionality Reduction", ln=True)
    pdf.set_font("Arial", size=10)
    
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(combined_vectors)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_proj = tsne.fit_transform(combined_vectors)
    
    # PCA Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(pca_proj[:, 0], pca_proj[:, 1], c='blue')
    plt.title('PCA Projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig("pca_plot.png")
    plt.close()
    pdf.image("pca_plot.png", x=10, w=180)
    
    # TSNE Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c='red')
    plt.title('t-SNE Projection')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig("tsne_plot.png")
    plt.close()
    pdf.image("tsne_plot.png", x=10, w=180)
    pdf.ln(10)
    
    # 5. Comparative Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "5. Comparative Analysis", ln=True)
    pdf.set_font("Arial", size=10)
    
    comparative_df = pd.DataFrame({name: np.mean(vec, axis=0) for name, vec in vectors.items()})
    pdf.multi_cell(0, 10, "Comparative Analysis Data:\n" + comparative_df.to_string())
    pdf.ln(10)
    
    # 6. Summary and Conclusion
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "6. Summary and Conclusion", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, "The analyses performed reveal relationships between various components of the Mirror Modeling system. Further testing is required to establish statistical significance.")
    
    # Save the report
    if not os.path.exists("reports"):
        os.makedirs("reports")
    report_file = f"reports/statistical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(report_file)
    
    # Cleanup images
    os.remove("kmeans_plot.png")
    os.remove("dbscan_plot.png")
    os.remove("pca_plot.png")
    os.remove("tsne_plot.png")


if st.sidebar.button("📄 Generate Statistical Report"):
    generate_statistical_report(vectors)
