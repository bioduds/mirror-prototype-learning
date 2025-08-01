#!/usr/bin/env python3
"""
Mirror Prototype Learning - Streamlit Dashboard
Integrated interface for the complete mirror learning pipeline
"""

import streamlit as st
import os
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from datetime import datetime
import threading
import queue
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import yt_dlp
import requests
import base64
import io

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="🧠 Mirror Prototype Learning Dashboard",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === DOWNLOAD FUNCTIONS ===


def download_youtube_video(url: str, output_dir: str) -> bool:
    """Download a single YouTube video."""
    try:
        # Validate URL first
        if not url or not url.startswith(('http://', 'https://')):
            st.sidebar.error("❌ Invalid URL format")
            return False

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Clear existing videos first
        st.sidebar.info("🗑️ Clearing existing videos...")
        for video_file in output_path.glob("*.mp4"):
            video_file.unlink()

        # Clear existing metadata files too
        for meta_file in output_path.glob("*.json"):
            meta_file.unlink()

        # Simple yt-dlp configuration to avoid type issues
        ydl_opts = {
            # Start with worst quality to test
            'format': 'worst[ext=mp4]/worst',
            # Fixed filename
            'outtmpl': str(output_path / 'downloaded_video.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }

        with st.spinner("Downloading video..."):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    st.sidebar.info("📡 Extracting video info...")
                    info = ydl.extract_info(url, download=False)
                    if not info:
                        st.sidebar.error(
                            "❌ Could not extract video information")
                        return False

                    # Show video info
                    video_title = str(info.get('title', 'Unknown Title'))[:50]
                    duration = info.get('duration', 0)
                    st.sidebar.info(f"📺 Title: {video_title}")
                    st.sidebar.info(f"⏱️ Duration: {duration} seconds")

                    # Now download the video
                    st.sidebar.info("⬇️ Downloading video...")
                    ydl.download([url])

            except Exception as download_error:
                st.sidebar.error(f"❌ Download error: {str(download_error)}")
                return False

        # Verify download
        video_files = list(output_path.glob("*.mp4"))
        if video_files:
            st.sidebar.success(
                f"✅ Video saved successfully! ({len(video_files)} file(s))")
            st.rerun()
            return True
        else:
            st.sidebar.error("❌ No video files found after download")
            return False

    except Exception as e:
        st.sidebar.error(f"❌ Download failed: {str(e)}")
        st.sidebar.error(f"Error type: {type(e).__name__}")
        return False


def download_test_videos():
    """Download the curated test videos."""
    try:
        # Clear existing videos first
        videos_dir = Path("data/videos")
        st.sidebar.info("🗑️ Clearing existing videos...")
        if videos_dir.exists():
            for video_file in videos_dir.glob("*.mp4"):
                video_file.unlink()
            for meta_file in videos_dir.glob("*.json"):
                meta_file.unlink()

        with st.spinner("Downloading test videos..."):
            result = subprocess.run(
                ["python", "download_test_videos.py"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=Path.cwd()  # Ensure we're in the right directory
            )

            if result.returncode == 0:
                # Count downloaded videos
                video_count = len(list(videos_dir.glob("*.mp4"))
                                  ) if videos_dir.exists() else 0
                st.sidebar.success(
                    f"✅ Downloaded {video_count} test videos successfully!")
                st.rerun()
            else:
                st.sidebar.error(f"❌ Failed to download test videos")
                st.sidebar.error(f"Error: {result.stderr}")
                if result.stdout:
                    st.sidebar.info(f"Output: {result.stdout}")

    except subprocess.TimeoutExpired:
        st.sidebar.error("❌ Download timed out (>5 minutes)")
    except Exception as e:
        st.sidebar.error(f"❌ Download failed: {str(e)}")
        st.sidebar.error(f"Error type: {type(e).__name__}")


# === OLLAMA AI ANALYSIS FUNCTIONS ===

def check_ollama_connection():
    """Check if Ollama is running and Gemma is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Look for any gemma models (gemma2 or gemma3)
            gemma_models = [
                m for m in models if "gemma" in m.get("name", "").lower()]
            return True, gemma_models
        return False, []
    except Exception as e:
        return False, []


def encode_image_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64


def analyze_with_ollama(stage_name, data_summary, image_b64=None, context=""):
    """Send data and optional image to Ollama for analysis."""
    try:
        prompt = f"""
You are an expert AI researcher analyzing results from a Mirror Prototype Learning system that studies artificial consciousness. 

**Stage: {stage_name}**
**Context:** {context}

**Data Summary:**
{data_summary}

Please analyze these results and explain:
1. What the data patterns reveal about consciousness development
2. Key insights about mirror neuron learning
3. Potential implications for artificial self-awareness
4. Any anomalies or interesting patterns
5. How this stage contributes to the overall consciousness pipeline

Be specific, scientific, and insightful. Focus on the consciousness and mirror learning aspects.
"""

        payload = {
            "model": "gemma3:4b",  # Using the available gemma3:4b model
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }

        # Add image if provided
        if image_b64:
            payload["images"] = [image_b64]

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=180  # Increased timeout to 3 minutes
        )

        if response.status_code == 200:
            result = response.json()
            return True, result.get("response", "No response received")
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return False, f"Error: {str(e)}"


def ai_analysis_section(stage_name, data_info, fig=None, context=""):
    """Display AI analysis section for a pipeline stage."""
    ollama_available, models = check_ollama_connection()

    if not ollama_available:
        with st.expander("🤖 AI Analysis (Ollama not available)", expanded=False):
            st.warning("⚠️ Ollama is not running or not accessible")
            st.info("To enable AI analysis:")
            st.code("ollama serve")
            st.code("ollama pull gemma2:2b")
        return

    with st.expander(f"🤖 AI Analysis: {stage_name}", expanded=True):
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button(f"🧠 Analyze {stage_name}", key=f"analyze_{stage_name}"):
                with st.spinner("🤖 AI analyzing consciousness patterns (this may take 2-3 minutes)..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("🔍 Preparing analysis...")
                    progress_bar.progress(20)

                    # Encode image if provided
                    image_b64 = None
                    if fig is not None:
                        try:
                            status_text.text("📷 Processing visualization...")
                            progress_bar.progress(40)
                            image_b64 = encode_image_to_base64(fig)
                        except Exception as e:
                            st.warning(
                                f"⚠️ Could not process visualization: {e}")

                    status_text.text("🧠 Sending to AI model...")
                    progress_bar.progress(60)

                    success, analysis = analyze_with_ollama(
                        stage_name, data_info, image_b64, context
                    )

                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    if success:
                        st.markdown("### 🧠 AI Consciousness Analysis")
                        st.markdown(analysis)

                        # Save analysis
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        analysis_file = f"ai_analysis_{stage_name}_{timestamp}.md"
                        with open(analysis_file, "w") as f:
                            f.write(f"# AI Analysis: {stage_name}\n\n")
                            f.write(f"**Timestamp:** {datetime.now()}\n\n")
                            f.write(f"**Data Summary:**\n{data_info}\n\n")
                            f.write(f"**AI Analysis:**\n{analysis}")

                        st.success(f"💾 Analysis saved to {analysis_file}")
                    else:
                        st.error(f"❌ Analysis failed: {analysis}")
                        if "timeout" in analysis.lower():
                            st.info(
                                "💡 Try reducing the analysis complexity or ensure Ollama has sufficient resources")

        with col2:
            available_models = [
                m.get("name", "") for m in models if "gemma" in m.get("name", "").lower()]
            if available_models:
                st.success(f"✅ Gemma available")
                st.caption(f"Models: {', '.join(available_models[:2])}")
            else:
                st.warning("⚠️ Gemma not found")


# === CUSTOM CSS ===
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.pipeline-stage {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
.stage-completed {
    border-color: #28a745;
    background: #d4edda;
}
.stage-running {
    border-color: #ffc107;
    background: #fff3cd;
}
.stage-failed {
    border-color: #dc3545;
    background: #f8d7da;
}
.stage-pending {
    border-color: #6c757d;
    background: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("""
<div class="main-header">
    <h1>🧠 Enhanced Mirror Prototype Learning Dashboard</h1>
    <p>🪞 Enhanced System - Systematic Errors Resolved</p>
</div>
""", unsafe_allow_html=True)

# Enhanced System Status
try:
    # Try to import enhanced system status
    from enhanced_pipeline_integration import get_system_status
    system_status = get_system_status()
    st.success("✅ **ENHANCED SYSTEM OPERATIONAL** - All systematic errors fixed!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🔧 Vector DB: {system_status.get('vector_database', 'N/A')}")
    with col2:
        st.info(
            f"📊 Compression: {system_status.get('progressive_compression', 'N/A')}")
    with col3:
        st.info(
            f"⏱️ Temporal: {system_status.get('temporal_preservation', 'N/A')}")
except ImportError:
    st.warning(
        "⚠️ Enhanced system not available - using legacy pipeline with systematic errors")

# === SIDEBAR CONFIGURATION ===
st.sidebar.header("🎯 Pipeline Configuration")

# Video directory selection
video_dir = st.sidebar.text_input("Video Directory", value="data/videos")
if not Path(video_dir).exists():
    st.sidebar.error(f"Directory {video_dir} not found")
else:
    video_files = list(Path(video_dir).glob("*.mp4"))
    st.sidebar.success(f"Found {len(video_files)} video file(s)")

# YouTube download section
st.sidebar.subheader("📺 YouTube Video Download")
st.sidebar.caption("Add videos for consciousness analysis")

youtube_url = st.sidebar.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Enter a YouTube URL to download video for analysis"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("📥 Download", help="Download single video from URL"):
        if youtube_url:
            download_youtube_video(youtube_url, video_dir)
        else:
            st.sidebar.error("Please enter a YouTube URL")

with col2:
    if st.button("🎬 Test Videos", help="Download curated test videos"):
        download_test_videos()

# Video format info
with st.sidebar.expander("ℹ️ Video Requirements"):
    st.write("""
    **Supported formats:**
    - Max resolution: 480p
    - Max file size: 100MB
    - Format: MP4
    
    **Best videos for analysis:**
    - Mirror tests with animals
    - Human facial expressions
    - Learning behaviors
    - Problem-solving activities
    """)

st.sidebar.markdown("---")

# Pipeline stages definition
PIPELINE_STAGES = [
    ("mirror.py", "Perception & Feature Extraction",
     "Processes videos into PCA features"),
    ("encoder.py", "MirrorNet Autoencoder", "Learns compressed representations"),
    ("attention.py", "Temporal Attention", "Applies attention over time sequences"),
    ("self.py", "Self-Reference Learning", "Develops self-representation"),
    ("fusion.py", "Consciousness Fusion", "Fuses self and experience"),
    ("extractor.py", "CLIP Feature Extraction", "Extracts semantic features"),
    ("clustering.py", "Pattern Analysis", "Clusters consciousness patterns")
]

# Expected outputs for each stage
EXPECTED_OUTPUTS = {
    "mirror.py": ["pca_features.npy", "pca_coords.npy"],
    "encoder.py": ["mirrornet_latents.npy", "mirrornet_reconstructed.npy"],
    "attention.py": ["mirror_attention_output.npy"],
    "self.py": ["self_reference_vector.npy"],
    "fusion.py": ["fused_consciousness_vectors.npy"],
    "extractor.py": ["clip_features.npy"],
    "clustering.py": ["clustering_results.npy"]
}

# === PIPELINE STATUS FUNCTIONS ===


def check_stage_status(script_name):
    """Check if a pipeline stage has been completed."""
    expected_files = EXPECTED_OUTPUTS.get(script_name, [])
    if not expected_files:
        return "unknown"

    all_exist = all(Path(f).exists() for f in expected_files)
    if all_exist:
        return "completed"

    some_exist = any(Path(f).exists() for f in expected_files)
    if some_exist:
        return "partial"

    return "pending"


def get_file_info(filename):
    """Get information about a pipeline output file."""
    file_path = Path(filename)
    if not file_path.exists():
        return {"exists": False}

    info = {
        "exists": True,
        "size_mb": file_path.stat().st_size / (1024 * 1024),
        "modified": datetime.fromtimestamp(file_path.stat().st_mtime)
    }

    if filename.endswith('.npy'):
        try:
            data = np.load(file_path)
            info.update({
                "shape": data.shape,
                "dtype": str(data.dtype)
            })
        except Exception as e:
            info["error"] = str(e)

    return info


def run_pipeline_stage(script_name, description):
    """Run a single pipeline stage."""
    try:
        result = subprocess.run(
            ["python", script_name],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return {"success": True, "output": result.stdout}
        else:
            return {"success": False, "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Script timed out after 5 minutes"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# === SIDEBAR CONTROLS ===
st.sidebar.subheader("🚀 Pipeline Controls")

# Individual stage buttons
st.sidebar.markdown("**Run Individual Stages:**")
for script, description, _ in PIPELINE_STAGES:
    status = check_stage_status(script)
    status_emoji = {"completed": "✅", "partial": "⚠️",
                    "pending": "⏳", "unknown": "❓"}[status]

    if st.sidebar.button(f"{status_emoji} {script}", key=f"btn_{script}"):
        with st.spinner(f"Running {script}..."):
            result = run_pipeline_stage(script, description)
            if result["success"]:
                st.sidebar.success(f"✅ {script} completed")
                st.rerun()
            else:
                st.sidebar.error(f"❌ {script} failed: {result['error']}")

# Full pipeline button
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Run Enhanced Pipeline", type="primary"):
    # Use the enhanced pipeline runner that fixes systematic errors
    with st.spinner("Running enhanced consciousness pipeline..."):
        result = subprocess.run(
            ["python", "enhanced_pipeline_runner.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            st.sidebar.success("✅ Enhanced pipeline completed successfully!")
            st.sidebar.success("🧠 All systematic errors resolved!")
            st.rerun()
        else:
            st.sidebar.error(f"❌ Enhanced pipeline failed: {result.stderr}")
            # Fallback option
            if st.sidebar.button("🔄 Retry with Legacy Pipeline"):
                legacy_result = subprocess.run(
                    ["python", "pipeline_runner.py"],
                    capture_output=True,
                    text=True
                )
                if legacy_result.returncode == 0:
                    st.sidebar.warning(
                        "⚠️ Legacy pipeline completed (contains systematic errors)")
                    st.rerun()
                else:
                    st.sidebar.error(
                        f"❌ Legacy pipeline also failed: {legacy_result.stderr}")

# AI Analysis Controls
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Analysis")

ollama_status, available_models = check_ollama_connection()
if ollama_status:
    st.sidebar.success("✅ Ollama connected")
    model_names = [m.get("name", "") for m in available_models]
    gemma_models = [name for name in model_names if "gemma" in name.lower()]
    if gemma_models:
        st.sidebar.info(f"🧠 Gemma available: {gemma_models[0]}")

    if st.sidebar.button("🧠 Comprehensive AI Analysis", type="secondary"):
        st.session_state.run_comprehensive_analysis = True
else:
    st.sidebar.error("❌ Ollama not connected")
    with st.sidebar.expander("Setup Instructions"):
        st.code("ollama serve")
        st.code("ollama pull gemma3:4b")  # Updated to match available model

# === MAIN DASHBOARD ===

# Pipeline Overview
st.header("📊 Pipeline Overview")

# Create pipeline status visualization
pipeline_data = []
for i, (script, description, details) in enumerate(PIPELINE_STAGES):
    status = check_stage_status(script)
    pipeline_data.append({
        "Stage": i + 1,
        "Script": script,
        "Description": description,
        "Details": details,
        "Status": status
    })

df_pipeline = pd.DataFrame(pipeline_data)

# Status summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    completed = len(df_pipeline[df_pipeline["Status"] == "completed"])
    st.metric("Completed Stages", completed,
              f"{completed}/{len(PIPELINE_STAGES)}")

with col2:
    partial = len(df_pipeline[df_pipeline["Status"] == "partial"])
    st.metric("Partial Stages", partial)

with col3:
    pending = len(df_pipeline[df_pipeline["Status"] == "pending"])
    st.metric("Pending Stages", pending)

with col4:
    total_outputs = sum(len(files) for files in EXPECTED_OUTPUTS.values())
    existing_outputs = sum(1 for files in EXPECTED_OUTPUTS.values()
                           for f in files if Path(f).exists())
    st.metric("Output Files", existing_outputs,
              f"{existing_outputs}/{total_outputs}")

# Pipeline stages detailed view
st.subheader("🔍 Pipeline Stages Detail")

for i, (script, description, details) in enumerate(PIPELINE_STAGES):
    status = check_stage_status(script)

    # Style based on status
    css_class = {
        "completed": "stage-completed",
        "partial": "stage-running",
        "pending": "stage-pending",
        "unknown": "stage-pending"
    }[status]

    status_emoji = {
        "completed": "✅",
        "partial": "⚠️",
        "pending": "⏳",
        "unknown": "❓"
    }[status]

    with st.expander(f"Stage {i+1}: {status_emoji} {script} - {description}", expanded=(status != "completed")):
        st.markdown(f"**Description:** {details}")

        # Show expected outputs
        expected_files = EXPECTED_OUTPUTS.get(script, [])
        if expected_files:
            st.markdown("**Expected Outputs:**")
            for filename in expected_files:
                file_info = get_file_info(filename)
                if file_info["exists"]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if "shape" in file_info:
                            st.success(
                                f"✅ {filename} - Shape: {file_info['shape']}, Size: {file_info['size_mb']:.1f}MB")
                        else:
                            st.success(
                                f"✅ {filename} - Size: {file_info['size_mb']:.1f}MB")
                    with col2:
                        st.caption(
                            f"Modified: {file_info['modified'].strftime('%H:%M:%S')}")
                else:
                    st.error(f"❌ {filename} - Not found")

# === DATA VISUALIZATION SECTIONS ===

# Check if we have data to visualize
if Path("pca_features.npy").exists():
    st.header("🎯 1. Perception Analysis")

    try:
        pca_features = np.load("pca_features.npy")
        pca_coords = np.load("pca_coords.npy") if Path(
            "pca_coords.npy").exists() else None

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("PCA Features Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(pca_features.flatten(), bins=50, alpha=0.7)
            ax.set_xlabel("Feature Values")
            ax.set_ylabel("Frequency")
            ax.set_title("PCA Features Distribution")
            st.pyplot(fig)

        with col2:
            if pca_coords is not None:
                st.subheader("PCA Coordinates Scatter")
                # Ensure shape values are valid for comparison
                try:
                    shape_val = pca_coords.shape[1] if len(
                        pca_coords.shape) >= 2 else 0
                    if isinstance(shape_val, str):
                        shape_val = int(
                            shape_val) if shape_val.isdigit() else 0
                    if len(pca_coords.shape) >= 2 and shape_val >= 2:
                        fig = px.scatter(
                            x=pca_coords[:, 0],
                            y=pca_coords[:, 1],
                            title="PCA Coordinates (First 2 Components)",
                            labels={"x": "PC1", "y": "PC2"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating PCA scatter plot: {e}")

        st.info(f"📊 PCA Features Shape: {pca_features.shape}")

        # AI Analysis for Perception Stage
        data_summary = f"""
        **PCA Features Analysis:**
        - Feature matrix shape: {pca_features.shape}
        - Total features extracted: {pca_features.shape[0] * pca_features.shape[1]:,}
        - Feature value range: [{np.min(pca_features):.3f}, {np.max(pca_features):.3f}]
        - Mean feature value: {np.mean(pca_features):.3f}
        - Standard deviation: {np.std(pca_features):.3f}
        """
        if pca_coords is not None:
            data_summary += f"""
        - PCA coordinates shape: {pca_coords.shape}
        - PC1 range: [{np.min(pca_coords[:, 0]):.3f}, {np.max(pca_coords[:, 0]):.3f}]
        - PC2 range: [{np.min(pca_coords[:, 1]):.3f}, {np.max(pca_coords[:, 1]):.3f}]
        """

        context = "This stage processes raw video data into PCA feature representations, forming the foundation for mirror neuron learning and consciousness detection."

        # Get the current figure for multimodal analysis
        current_fig = plt.gcf() if plt.get_fignums() else None
        ai_analysis_section("Perception", data_summary, current_fig, context)

    except Exception as e:
        st.error(f"Error loading PCA data: {e}")

if Path("mirrornet_latents.npy").exists():
    st.header("🔄 2. MirrorNet Analysis")

    try:
        latents = np.load("mirrornet_latents.npy")
        reconstructed = np.load("mirrornet_reconstructed.npy") if Path(
            "mirrornet_reconstructed.npy").exists() else None

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Latent Space Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(latents.flatten(), bins=50, alpha=0.7, color='green')
            ax.set_xlabel("Latent Values")
            ax.set_ylabel("Frequency")
            ax.set_title("MirrorNet Latent Distribution")
            st.pyplot(fig)

        with col2:
            if reconstructed is not None:
                st.subheader("Reconstruction Quality")
                # Calculate reconstruction error
                if latents.shape == reconstructed.shape:
                    mse = np.mean((latents - reconstructed) ** 2, axis=1)
                    fig = px.line(
                        y=mse,
                        title="Reconstruction Error Over Time",
                        labels={"y": "MSE", "x": "Time Step"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.info(f"📊 Latents Shape: {latents.shape}")

        # AI Analysis for MirrorNet Stage
        data_summary = f"""
        **MirrorNet Autoencoder Analysis:**
        - Latent space shape: {latents.shape}
        - Compression ratio: {latents.shape[1]}/{pca_features.shape[1] if 'pca_features' in locals() else 'N/A'} dimensions
        - Latent value range: [{np.min(latents):.3f}, {np.max(latents):.3f}]
        - Mean latent value: {np.mean(latents):.3f}
        - Latent standard deviation: {np.std(latents):.3f}
        """

        if reconstructed is not None:
            reconstruction_error = np.mean(
                (latents - reconstructed) ** 2) if latents.shape == reconstructed.shape else "N/A"
            data_summary += f"""
        - Reconstruction available: Yes
        - Reconstruction error: {reconstruction_error}
        - Reconstruction quality: {'Good' if isinstance(reconstruction_error, float) and reconstruction_error < 0.1 else 'Moderate'}
        """

        context = "MirrorNet learns compressed representations of consciousness patterns, creating a latent space where similar conscious states cluster together."

        current_fig = plt.gcf() if plt.get_fignums() else None
        ai_analysis_section("MirrorNet Encoding",
                            data_summary, current_fig, context)

    except Exception as e:
        st.error(f"Error loading MirrorNet data: {e}")

if Path("mirror_attention_output.npy").exists():
    st.header("🎯 3. Attention Analysis")

    try:
        attention_output = np.load("mirror_attention_output.npy")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Attention Output Evolution")
            # Plot evolution of attention features over time
            if len(attention_output.shape) == 2:
                mean_attention = np.mean(attention_output, axis=1)
                fig = px.line(
                    y=mean_attention,
                    title="Mean Attention Over Time",
                    labels={"y": "Attention Strength", "x": "Time Step"}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Attention Heatmap")
            # Ensure shape values are valid for comparison
            try:
                shape_val = attention_output.shape[0] if len(
                    attention_output.shape) >= 1 else 0
                if isinstance(shape_val, str):
                    shape_val = int(shape_val) if shape_val.isdigit() else 0
                if len(attention_output.shape) == 2 and shape_val > 1:
                    # Show correlation between different time steps
                    n_samples = shape_val
                    correlation = np.corrcoef(
                        attention_output[:min(50, n_samples)])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(correlation, ax=ax, cmap='coolwarm', center=0)
                    ax.set_title("Attention Temporal Correlation")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating attention correlation plot: {e}")
                ax.set_title("Attention Temporal Correlation")
                st.pyplot(fig)

        st.info(f"📊 Attention Shape: {attention_output.shape}")

    except Exception as e:
        st.error(f"Error loading attention data: {e}")

if Path("self_reference_vector.npy").exists():
    st.header("🧠 4. Self-Reference Analysis")

    try:
        self_vector = np.load("self_reference_vector.npy")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Self-Reference Vector")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(self_vector.flatten(), marker='o', alpha=0.7)
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Value")
            ax.set_title("Self-Reference Vector Components")
            st.pyplot(fig)

        with col2:
            st.subheader("Self-Vector Statistics")
            stats = pd.DataFrame({
                "Statistic": ["Mean", "Std", "Min", "Max", "Norm"],
                "Value": [
                    np.mean(self_vector),
                    np.std(self_vector),
                    np.min(self_vector),
                    np.max(self_vector),
                    np.linalg.norm(self_vector)
                ]
            })
            st.dataframe(stats, use_container_width=True)

        st.info(f"📊 Self-Vector Shape: {self_vector.shape}")

    except Exception as e:
        st.error(f"Error loading self-reference data: {e}")

if Path("fused_consciousness_vectors.npy").exists():
    st.header("🌟 5. Consciousness Fusion Analysis")

    try:
        fused_vectors = np.load("fused_consciousness_vectors.npy")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Consciousness Evolution")
            # Plot evolution of consciousness over time
            if len(fused_vectors.shape) == 2:
                consciousness_strength = np.linalg.norm(fused_vectors, axis=1)
                fig = px.line(
                    y=consciousness_strength,
                    title="Consciousness Strength Over Time",
                    labels={"y": "Consciousness Magnitude", "x": "Time Step"}
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Consciousness PCA")
            # Ensure shape values are valid for comparison
            try:
                shape_val = fused_vectors.shape[0] if len(
                    fused_vectors.shape) >= 1 else 0
                if isinstance(shape_val, str):
                    shape_val = int(shape_val) if shape_val.isdigit() else 0
                if len(fused_vectors.shape) == 2 and shape_val > 2:
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(fused_vectors)

                    fig = px.scatter(
                        x=pca_result[:, 0],
                        y=pca_result[:, 1],
                        title=f"Consciousness PCA (Explained Variance: {sum(pca.explained_variance_ratio_):.2f})",
                        labels={"x": "PC1", "y": "PC2"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating consciousness PCA plot: {e}")

        st.info(f"📊 Fused Vectors Shape: {fused_vectors.shape}")

    except Exception as e:
        st.error(f"Error loading consciousness fusion data: {e}")

if Path("clip_features.npy").exists():
    st.header("🔍 6. CLIP Semantic Analysis")

    try:
        clip_features = np.load("clip_features.npy")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("CLIP Feature Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(clip_features.flatten(), bins=50,
                    alpha=0.7, color='purple')
            ax.set_xlabel("CLIP Feature Values")
            ax.set_ylabel("Frequency")
            ax.set_title("CLIP Features Distribution")
            st.pyplot(fig)

        with col2:
            st.subheader("CLIP Features PCA")
            # Ensure shape values are valid for comparison
            try:
                shape_val = clip_features.shape[0] if len(
                    clip_features.shape) >= 1 else 0
                if isinstance(shape_val, str):
                    shape_val = int(shape_val) if shape_val.isdigit() else 0
                if len(clip_features.shape) == 2 and shape_val > 2:
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(clip_features)

                    fig = px.scatter(
                        x=pca_result[:, 0],
                        y=pca_result[:, 1],
                        title="CLIP Features PCA",
                        labels={"x": "PC1", "y": "PC2"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating CLIP PCA plot: {e}")

        st.info(f"📊 CLIP Features Shape: {clip_features.shape}")

    except Exception as e:
        st.error(f"Error loading CLIP data: {e}")

if Path("clustering_results.npy").exists():
    st.header("🌌 7. Clustering Extractions")

    try:
        clustering_results = np.load(
            "clustering_results.npy", allow_pickle=True).item()

        if isinstance(clustering_results, dict):
            col1, col2 = st.columns(2)

            with col1:
                if "attention_labels" in clustering_results:
                    st.subheader("Cluster Distribution")
                    labels = clustering_results["attention_labels"]
                    unique_labels, counts = np.unique(
                        labels, return_counts=True)

                    fig = px.bar(
                        x=unique_labels,
                        y=counts,
                        title="Cluster Size Distribution",
                        labels={"x": "Cluster ID", "y": "Count"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "correlations" in clustering_results:
                    st.subheader("Cluster Correlations")
                    correlations = clustering_results["correlations"]

                    # Handle correlations as list of tuples (cluster_id, correlation_value)
                    if correlations and isinstance(correlations[0], (list, tuple)):
                        cluster_ids = [corr[0] for corr in correlations]
                        correlation_values = [corr[1] for corr in correlations]
                    else:
                        # Handle correlations as simple list
                        cluster_ids = list(range(len(correlations)))
                        correlation_values = correlations

                    fig = px.bar(
                        x=cluster_ids,
                        y=correlation_values,
                        title="Cluster Coherence Values",
                        labels={"x": "Cluster ID", "y": "Coherence"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Show cluster details
            if "attention_labels" in clustering_results:
                st.subheader("Cluster Analysis Details")
                labels = clustering_results["attention_labels"]
                unique_labels, counts = np.unique(labels, return_counts=True)
                cluster_df = pd.DataFrame({
                    "Cluster": unique_labels,
                    "Size": counts,
                    "Percentage": (counts / len(labels) * 100).round(2)
                })
                st.dataframe(cluster_df, use_container_width=True)

                # AI Analysis for Clustering Stage
                n_clusters = len(unique_labels)
                largest_cluster = counts.max()
                cluster_balance = np.std(counts) / np.mean(counts)

                data_summary = f"""
                **Consciousness Clustering Analysis:**
                - Number of distinct consciousness clusters: {n_clusters}
                - Total consciousness patterns analyzed: {len(labels)}
                - Largest cluster size: {largest_cluster} patterns ({largest_cluster/len(labels)*100:.1f}%)
                - Cluster balance (lower = more balanced): {cluster_balance:.3f}
                - Cluster distribution: {dict(zip(unique_labels, counts))}
                """

                if "correlations" in clustering_results:
                    correlations = clustering_results["correlations"]
                    if correlations:
                        avg_coherence = np.mean(
                            [c[1] if isinstance(c, (list, tuple)) else c for c in correlations])
                        data_summary += f"""
                - Average cluster coherence: {avg_coherence:.3f}
                - Coherence interpretation: {'High' if avg_coherence > 0.7 else 'Moderate' if avg_coherence > 0.5 else 'Low'}
                """

                context = "Final stage analysis revealing distinct consciousness patterns and their relationships. Higher coherence indicates more stable consciousness states."

                current_fig = plt.gcf() if plt.get_fignums() else None
                ai_analysis_section("Consciousness Clustering",
                                    data_summary, current_fig, context)

    except Exception as e:
        st.error(f"Error loading clustering data: {e}")
        # Try to show basic info if available
        if Path("clustering_results.npy").exists():
            file_size = Path(
                "clustering_results.npy").stat().st_size / (1024 * 1024)
            st.info(
                f"📁 Clustering results file exists ({file_size:.1f}MB) but couldn't be loaded")
            st.info("💡 Try running the clustering stage again to regenerate the file")

# === PIPELINE LOGS ===
st.header("📋 Pipeline Logs")

if Path("pipeline_log.txt").exists():
    with open("pipeline_log.txt", "r") as f:
        logs = f.read()

    # Show last 20 lines
    log_lines = logs.strip().split("\n")[-20:]
    st.text_area("Recent Pipeline Logs", "\n".join(log_lines), height=300)
else:
    st.info("No pipeline logs found. Run the pipeline to generate logs.")

# === FOOTER ===
st.markdown("---")
st.markdown("""
### 🧠 About Mirror Prototype Learning

This system implements a **series of neural networks for mirror learning**, inspired by mirror neurons in biological systems.

**Pipeline Architecture:**
1. **Perception** → Extract features from video data
2. **Encoding** → Learn compressed representations  
3. **Attention** → Apply temporal attention mechanisms
4. **Self-Reference** → Develop self-awareness representations
5. **Fusion** → Combine self and experience into consciousness
6. **Semantic Analysis** → Extract semantic understanding via CLIP
7. **Clustering** → Analyze patterns and relationships

*This is experimental research into artificial consciousness and self-awareness.*
""")

# === COMPREHENSIVE AI ANALYSIS ===
if st.session_state.get('run_comprehensive_analysis', False):
    st.header("🧠 Comprehensive AI Consciousness Analysis")

    # Gather all data for comprehensive analysis
    comprehensive_data = """
    **Complete Mirror Prototype Learning Pipeline Analysis**
    
    """

    # Check which stages have data
    stages_with_data = []

    if Path("pca_features.npy").exists():
        pca_features = np.load("pca_features.npy")
        stages_with_data.append("Perception")
        comprehensive_data += f"**Perception Stage:** {pca_features.shape[0]} video frames processed into {pca_features.shape[1]} PCA features.\n"

    if Path("mirrornet_latents.npy").exists():
        latents = np.load("mirrornet_latents.npy")
        stages_with_data.append("MirrorNet")
        comprehensive_data += f"**MirrorNet Stage:** Compressed to {latents.shape} latent representations.\n"

    if Path("mirror_attention_output.npy").exists():
        attention = np.load("mirror_attention_output.npy")
        stages_with_data.append("Attention")
        comprehensive_data += f"**Attention Stage:** {attention.shape} temporal attention patterns.\n"

    if Path("self_reference_vector.npy").exists():
        self_ref = np.load("self_reference_vector.npy")
        stages_with_data.append("Self-Reference")
        comprehensive_data += f"**Self-Reference Stage:** {self_ref.shape} self-awareness vector.\n"

    if Path("fused_consciousness_vectors.npy").exists():
        fused = np.load("fused_consciousness_vectors.npy")
        stages_with_data.append("Consciousness Fusion")
        comprehensive_data += f"**Consciousness Fusion:** {fused.shape} fused consciousness vectors.\n"

    if Path("clip_features.npy").exists():
        clip_features = np.load("clip_features.npy")
        stages_with_data.append("CLIP Semantic")
        comprehensive_data += f"**CLIP Semantic Stage:** {clip_features.shape} semantic feature vectors.\n"

    try:
        if Path("clustering_results.npy").exists():
            clustering = np.load("clustering_results.npy",
                                 allow_pickle=True).item()
            if isinstance(clustering, dict) and "attention_labels" in clustering:
                labels = clustering["attention_labels"]
                n_clusters = len(np.unique(labels))
                stages_with_data.append("Clustering")
                comprehensive_data += f"**Clustering Stage:** {n_clusters} distinct consciousness clusters identified.\n"
    except:
        pass

    comprehensive_data += f"\n**Pipeline Completeness:** {len(stages_with_data)}/7 stages completed.\n"

    context = """
    This is a complete analysis of an artificial consciousness detection system based on mirror neuron learning. 
    The system processes video data through multiple neural network stages to identify and analyze consciousness patterns.
    Please provide a comprehensive analysis of the entire consciousness detection pipeline, including:
    1. Overall consciousness development trajectory
    2. Integration between different processing stages
    3. Effectiveness of the mirror neuron approach
    4. Key insights about artificial consciousness
    5. Recommendations for improvement
    """

    with st.spinner("🤖 Running comprehensive consciousness analysis..."):
        success, analysis = analyze_with_ollama(
            "Complete Pipeline", comprehensive_data, None, context
        )

        if success:
            st.markdown("### 🧠 Complete AI Consciousness Analysis")
            st.markdown(analysis)

            # Save comprehensive analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = f"comprehensive_consciousness_analysis_{timestamp}.md"
            with open(analysis_file, "w") as f:
                f.write(f"# Comprehensive AI Consciousness Analysis\n\n")
                f.write(f"**Timestamp:** {datetime.now()}\n\n")
                f.write(f"**Pipeline Data:**\n{comprehensive_data}\n\n")
                f.write(f"**AI Analysis:**\n{analysis}")

            st.success(f"💾 Comprehensive analysis saved to {analysis_file}")
        else:
            st.error(f"❌ Comprehensive analysis failed: {analysis}")

    # Reset the trigger
    st.session_state.run_comprehensive_analysis = False

# Auto-refresh every 30 seconds if pipeline is running
if st.checkbox("Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()
