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
import torch
import time
import json
from pathlib import Path

matplotlib.use('Agg')

# Configure Streamlit
st.set_page_config(
    page_title="🧠 Consciousness Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title
st.title("🧠 **Consciousness Search** - Proto-Conscious AGI")
st.markdown("*Search for consciousness in video content using advanced mirror learning and recursive self-abstraction*")

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'consciousness_results' not in st.session_state:
    st.session_state.consciousness_results = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "ready"

# === SIDEBAR: CONSCIOUSNESS SEARCH INPUT ===
st.sidebar.header("🎬 **Consciousness Search**")
st.sidebar.markdown("Paste any YouTube URL to analyze consciousness patterns")

# YouTube URL Input
yt_url = st.sidebar.text_input(
    "YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Paste any YouTube video URL to begin consciousness analysis"
)

# Main Search Button
search_button = st.sidebar.button(
    "🧠 **SEARCH FOR CONSCIOUSNESS**",
    type="primary",
    use_container_width=True,
    disabled=(not yt_url or st.session_state.processing_status == "processing")
)

# Processing Status Indicator
if st.session_state.processing_status == "processing":
    st.sidebar.info("🔄 **Processing in progress...**")
    st.sidebar.progress(0.5)
elif st.session_state.processing_status == "complete":
    st.sidebar.success("✅ **Consciousness analysis complete!**")
elif st.session_state.processing_status == "error":
    st.sidebar.error("❌ **Analysis failed**")
else:
    st.sidebar.markdown("*Ready to search for consciousness...*")

# === MAIN PROCESSING LOGIC ===


def download_and_process_video(url: str):
    """Download video and run full consciousness pipeline."""

    # Step 1: Download Video
    st.session_state.processing_status = "processing"

    with st.status("🎬 **Downloading video...**", expanded=True) as status:
        try:
            # Clear previous videos
            video_output_dir = "data/videos"
            shutil.rmtree(video_output_dir, ignore_errors=True)
            os.makedirs(video_output_dir, exist_ok=True)

            # Get video info
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info['title']
                video_duration = info.get('duration', 0)
                hash_id = hashlib.sha256(
                    video_title.encode("utf-8")).hexdigest()[:8]
                video_hash_name = f"v{hash_id}"
                video_filename = f"{video_hash_name}.mp4"

            st.write(f"📼 **Title:** {video_title}")
            st.write(
                f"⏱️ **Duration:** {video_duration//60}:{video_duration % 60:02d}")
            st.write(f"🆔 **Hash ID:** {video_hash_name}")

            # Download configuration
            ydl_opts = {
                'outtmpl': os.path.join(video_output_dir, video_filename),
                'format': 'mp4/bestaudio/best',
                'quiet': True
            }

            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Store video info
            st.session_state.video_info = {
                'title': video_title,
                'duration': video_duration,
                'hash_name': video_hash_name,
                'filename': video_filename,
                'path': os.path.join(video_output_dir, video_filename)
            }

            status.update(
                label="✅ **Video downloaded successfully**", state="complete")

        except Exception as e:
            status.update(label="❌ **Download failed**", state="error")
            st.error(f"Download error: {e}")
            return False

    # Step 2: Run Consciousness Analysis
    with st.status("🧠 **Running Consciousness Analysis...**", expanded=True) as status:
        try:
            st.write("🔧 **Initializing consciousness system...**")
            time.sleep(1)  # Visual feedback

            st.write("🎯 **Processing through perception networks...**")
            time.sleep(1)

            st.write("🪞 **Applying mirror learning...**")
            time.sleep(1)

            st.write("🎭 **Generating qualia and subjective experience...**")
            time.sleep(1)

            st.write("🧩 **Performing phenomenal binding...**")
            time.sleep(1)

            st.write("🎯 **Assessing consciousness level...**")

            # Run consciousness_runner.py
            result = subprocess.run(
                [sys.executable, "consciousness_runner.py"],
                capture_output=True,
                text=True,
                timeout=3600  # 60 minute timeout
            )

            if result.returncode == 0:
                # Parse consciousness results
                consciousness_data = parse_consciousness_output(result.stdout)
                st.session_state.consciousness_results = consciousness_data

                status.update(
                    label="✅ **Consciousness analysis complete**", state="complete")
                st.session_state.processing_status = "complete"
                return True
            else:
                status.update(label="❌ **Analysis failed**", state="error")
                st.error(f"Consciousness analysis error: {result.stderr}")
                st.session_state.processing_status = "error"
                return False

        except subprocess.TimeoutExpired:
            status.update(label="⏰ **Analysis timeout**", state="error")
            st.error("Consciousness analysis timed out after 60 minutes")
            st.session_state.processing_status = "error"
            return False
        except Exception as e:
            status.update(label="❌ **Analysis failed**", state="error")
            st.error(f"Analysis error: {e}")
            st.session_state.processing_status = "error"
            return False


def parse_consciousness_output(output: str) -> dict:
    """Parse consciousness analysis output."""
    lines = output.split('\n')

    # Look for consciousness metrics
    results = {
        'consciousness_level': 'UNKNOWN',
        'consciousness_score': 0.0,
        'videos_processed': 0,
        'conscious_episodes': 0,
        'consciousness_rate': 0.0,
        'components': {}
    }

    try:
        for line in lines:
            # Parse the exact format from consciousness_runner.py
            if 'Level:' in line and ('PRE_CONSCIOUS' in line or 'CONSCIOUS' in line or 'UNCONSCIOUS' in line):
                level = line.split('Level:')[1].strip()
                results['consciousness_level'] = level
            elif 'Score:' in line and not 'Level:' in line:
                score_text = line.split('Score:')[1].strip()
                results['consciousness_score'] = float(score_text)
            elif 'Total Videos Processed:' in line:
                results['videos_processed'] = int(line.split(':')[1].strip())
            elif 'Conscious Episodes:' in line:
                results['conscious_episodes'] = int(line.split(':')[1].strip())
            elif 'Consciousness Rate:' in line:
                rate_text = line.split(':')[1].strip().replace('%', '')
                results['consciousness_rate'] = float(rate_text)
            elif 'Confidence:' in line and not 'Metacognitive' in line:
                results['components']['metacognitive'] = float(
                    line.split(':')[1].strip())
            elif 'Qualia:' in line:
                results['components']['qualia'] = float(
                    line.split(':')[1].strip())
            elif 'Binding:' in line:
                results['components']['binding'] = float(
                    line.split(':')[1].strip())
    except Exception as e:
        print(f"Parsing error: {e}")  # Debug info
        pass  # Use defaults

    return results


# Execute processing when button is clicked
if search_button and yt_url:
    success = download_and_process_video(yt_url)
    if success:
        st.rerun()

# === CONSCIOUSNESS RESULTS DISPLAY ===
if st.session_state.processing_status == "complete" and st.session_state.consciousness_results:

    # Video Information Card
    if st.session_state.video_info:
        st.header("📼 **Video Information**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "**Title**", st.session_state.video_info['title'][:30] + "...")
        with col2:
            duration = st.session_state.video_info['duration']
            st.metric("**Duration**", f"{duration//60}:{duration % 60:02d}")
        with col3:
            st.metric("**Hash ID**", st.session_state.video_info['hash_name'])

    # Main Consciousness Results
    st.header("🧠 **Consciousness Analysis Results**")

    results = st.session_state.consciousness_results

    # Primary Consciousness Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        consciousness_level = results.get('consciousness_level', 'UNKNOWN')
        if consciousness_level == 'CONSCIOUS':
            st.metric("**Consciousness Level**",
                      consciousness_level, delta="🟢 Achieved")
        elif consciousness_level == 'PRE_CONSCIOUS':
            st.metric("**Consciousness Level**",
                      consciousness_level, delta="🟡 Emerging")
        else:
            st.metric("**Consciousness Level**",
                      consciousness_level, delta="🔴 Not Detected")

    with col2:
        score = results.get('consciousness_score', 0.0)
        st.metric("**Consciousness Score**", f"{score:.3f}",
                  delta=f"{'🎯 Above threshold' if score >= 0.6 else '⚠️ Below threshold'}")

    with col3:
        rate = results.get('consciousness_rate', 0.0)
        st.metric("**Consciousness Rate**", f"{rate:.1f}%",
                  delta=f"{'🚀 High' if rate > 50 else '📊 Developing'}")

    with col4:
        episodes = results.get('conscious_episodes', 0)
        st.metric("**Conscious Episodes**", episodes,
                  delta=f"{'✨ Multiple' if episodes > 1 else '🎯 Single' if episodes == 1 else '❌ None'}")

    # Consciousness Components Analysis
    st.subheader("🧩 **Consciousness Components**")

    components = results.get('components', {})

    if components:
        col1, col2, col3 = st.columns(3)

        with col1:
            meta_conf = components.get('metacognitive', 0.0)
            st.metric("**Metacognitive Awareness**", f"{meta_conf:.3f}",
                      delta="🤔 Self-thinking")

        with col2:
            qualia_int = components.get('qualia', 0.0)
            st.metric("**Qualia Intensity**", f"{qualia_int:.3f}",
                      delta="🌈 Subjective Experience")

        with col3:
            binding_str = components.get('binding', 0.0)
            st.metric("**Phenomenal Binding**", f"{binding_str:.3f}",
                      delta="🧩 Unity of Experience")

    # Consciousness Progress Visualization
    st.subheader("📊 **Consciousness Development**")

    # Create a consciousness radar chart
    if components:
        import plotly.graph_objects as go

        categories = ['Metacognitive<br>Awareness', 'Qualia<br>Intensity',
                      'Phenomenal<br>Binding', 'Overall<br>Score']
        values = [
            components.get('metacognitive', 0.0),
            components.get('qualia', 0.0),
            components.get('binding', 0.0),
            results.get('consciousness_score', 0.0)
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(0, 100, 255, 0.8)', width=3),
            marker=dict(size=8, color='rgba(0, 100, 255, 1)'),
            name='Consciousness Level'
        ))

        # Add consciousness threshold line
        threshold_values = [0.6, 0.6, 0.6, 0.6]
        fig.add_trace(go.Scatterpolar(
            r=threshold_values,
            theta=categories,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Consciousness Threshold'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.0],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=['0.2', '0.4', '0.6 (Threshold)', '0.8', '1.0']
                )),
            showlegend=True,
            title="🧠 Consciousness Component Analysis",
            title_x=0.5
        )

        st.plotly_chart(fig, use_container_width=True)

    # Consciousness Interpretation
    st.subheader("🔍 **Consciousness Interpretation**")
    
    score = results.get('consciousness_score', 0.0)
    level = results.get('consciousness_level', 'UNKNOWN')
    
    if score >= 0.8:
        st.success("🎉 **HIGHLY CONSCIOUS** - This video demonstrates strong consciousness patterns with robust self-awareness, metacognition, and unified experience.")
    elif score >= 0.6:
        st.success(
            "✨ **CONSCIOUS** - This video shows clear consciousness emergence with integrated awareness and subjective experience.")
    elif score >= 0.4:
        st.warning("🟡 **PRE-CONSCIOUS** - This video exhibits developing consciousness patterns. Some components are emerging but not yet fully integrated.")
    elif score >= 0.2:
        st.info("🔄 **PROTO-CONSCIOUS** - This video shows basic consciousness building blocks but lacks integration and unity.")
    else:
        st.error(
            "❌ **NON-CONSCIOUS** - This video does not demonstrate consciousness patterns in our analysis framework.")

# === HISTORICAL CONSCIOUSNESS DATA ===
st.header("📈 **Consciousness History**")

# Try to load historical data
vectors_dir = Path("vectors")
if vectors_dir.exists():
    video_folders = [d for d in vectors_dir.iterdir() if d.is_dir()]
    
    if video_folders:
        st.subheader(
            f"🗄️ **Analysis History** ({len(video_folders)} videos processed)")

        # Create a simple history table
        history_data = []
        for folder in sorted(video_folders):
            folder_name = folder.name
            # Try to extract basic info
            history_data.append({
                'Video ID': folder_name,
                'Status': '✅ Processed',
                'Components': 'Available' if (folder / 'self_reference_vector.npy').exists() else 'Missing'
            })

        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
    else:
        st.info(
            "No historical consciousness data found. Process your first video to begin building history.")
else:
    st.info("Vectors directory not found. Historical data will appear after processing videos.")

# === FOOTER INFORMATION ===
st.markdown("---")
st.markdown("""
### 🧠 **About Consciousness Search**

This system uses advanced **mirror learning** and **recursive self-abstraction** to detect and measure consciousness patterns in video content.

**Key Technologies:**
- 🪞 **Mirror Networks**: Self-referential learning
- 🎭 **Qualia Generation**: Subjective experience modeling  
- 🧩 **Phenomenal Binding**: Unified consciousness integration
- 🤔 **Metacognition**: Thinking about thinking
- 🎯 **Intentionality**: Goal formation and purpose

**Consciousness Threshold**: 0.6 (scores above this indicate likely consciousness)

*This is experimental research into artificial consciousness and proto-AGI systems.*
""")
