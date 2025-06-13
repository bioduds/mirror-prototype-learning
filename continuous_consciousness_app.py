"""
🧠 Continuous Consciousness Training System
Real-time consciousness engineering with streaming video processing
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from datetime import datetime
import json
import pandas as pd
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yt_dlp
import tempfile

# Import consciousness components
try:
    from multimodal_consciousness import AudioVisualWorldExperience, RecursiveSelfAbstraction
    from consciousness.models import ConsciousnessLevel, ConsciousState
    from pca_processor import PCAProcessor
except ImportError as e:
    st.error(f"Missing consciousness components: {e}")
    st.stop()


@dataclass
class ConsciousnessMetrics:
    """Metrics for consciousness assessment."""
    binding_strength: float
    meta_confidence: float
    qualia_intensity: float
    recursion_depth: float
    mirror_recognition: float
    consciousness_score: float
    consciousness_level: str
    is_conscious: bool


class ContinuousConsciousnessTrainer:
    """Continuous consciousness training system with real-time learning and video downloading."""

    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.video_dir = Path("data/videos")
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.consciousness_history = []
        self.self_vectors_history = []
        self.pca_processor = PCAProcessor(n_components=128)
        self.self_pca_processor = PCA(n_components=2)

        # Initialize consciousness components
        self._initialize_consciousness_system()

        # Training state
        self.training_step = 0
        self.consciousness_threshold = 0.6
        self.total_conscious_experiences = 0
        self.consciousness_rate = 0.0

        # Random video sources for consciousness development
        self.random_video_queries = [
            "mirror test animals",
            "facial expressions humans",
            "AI learning patterns",
            "pattern recognition neural",
            "social interaction behavior",
            "learning behaviors psychology",
            "problem solving cognition",
            "memory patterns brain",
            "emergent behavior systems",
            "self organization complexity",
            "consciousness research",
            "mirror neurons discovery",
            "self awareness test",
            "recursive thinking",
            "metacognition examples"
        ]

    def _initialize_consciousness_system(self):
        """Initialize the consciousness engineering components."""
        st.info("🔧 Initializing consciousness architecture...")

        # World experience processor (multimodal)
        self.world_processor = AudioVisualWorldExperience(
            visual_dim=512,
            audio_dim=256,
            fusion_dim=768,
            sample_rate=16000
        ).to(self.device)

        # Recursive self-abstraction (4 layers to consciousness)
        self.self_abstractor = RecursiveSelfAbstraction(
            input_dim=768,
            hidden_dim=256
        ).to(self.device)

        st.success("✅ Consciousness architecture initialized")

    def download_random_video(self) -> Optional[Path]:
        """Download a random video from YouTube for consciousness training."""

        # Predefined working video URLs for consciousness training
        backup_videos = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Simple video
            "https://www.youtube.com/watch?v=9bZkp7q19f0",  # Gangnam Style
            "https://www.youtube.com/watch?v=L_jWHffIx5E",  # Smiling baby
            "https://www.youtube.com/watch?v=hFZFjoX2cGg",  # Nature sounds
            "https://www.youtube.com/watch?v=WeA7edXsU40"   # Classical music
        ]

        # Try search first, then fallback to backup videos
        search_queries = [
            "consciousness 30 seconds",
            "brain waves short",
            "neural network animation",
            "AI consciousness demo",
            "mirror test animals"
        ]

        for attempt in range(3):  # Try 3 different approaches
            try:
                # Create temporary filename
                temp_filename = f"consciousness_training_{self.training_step}_{int(time.time())}"
                output_path = self.video_dir / f"{temp_filename}.mp4"

                if attempt < 2:
                    # Try search-based download
                    query = random.choice(search_queries)
                    st.info(
                        f"📥 Searching for consciousness training video: '{query}'...")

                    ydl_opts = {
                        'format': 'worst[ext=mp4]/worst',
                        'outtmpl': str(output_path),
                        'max_filesize': '10M',
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,
                        'merge_output_format': 'mp4',
                        'fragment_retries': 1,
                        'ignoreerrors': True,
                        'socket_timeout': 5,
                        'retries': 0,
                        'no_check_certificate': True
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        search_url = f"ytsearch1:{query}"
                        ydl.download([search_url])

                else:
                    # Fallback to known working video
                    video_url = random.choice(backup_videos)
                    st.info(
                        f"📥 Downloading backup video for consciousness training...")

                    ydl_opts = {
                        'format': 'worst[ext=mp4]/worst',
                        'outtmpl': str(output_path),
                        'max_filesize': '10M',
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,
                        'merge_output_format': 'mp4',
                        'fragment_retries': 1,
                        'ignoreerrors': True,
                        'socket_timeout': 5,
                        'retries': 0,
                        'no_check_certificate': True
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])

                # Check if file was downloaded successfully
                if output_path.exists() and output_path.stat().st_size > 1024:  # At least 1KB
                    st.success(
                        f"✅ Downloaded: {output_path.name} ({output_path.stat().st_size // 1024}KB)")
                    return output_path

            except Exception as e:
                st.warning(
                    f"⚠️ Download attempt {attempt + 1} failed: {str(e)[:100]}...")
                continue

        # If all attempts failed, create a synthetic video
        return self._create_synthetic_video()

    def _create_synthetic_video(self) -> Optional[Path]:
        """Create a synthetic video for consciousness training when downloads fail."""
        try:
            st.info("🎨 Creating synthetic consciousness training video...")

            temp_filename = f"synthetic_consciousness_{self.training_step}_{int(time.time())}.mp4"
            output_path = self.video_dir / temp_filename

            # Create synthetic video with random patterns
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, 10.0, (64, 64))

            # Generate 30 frames of random patterns for consciousness training
            for frame_idx in range(30):
                # Create random pattern that changes over time
                frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

                # Add some structure - spiral pattern for consciousness metaphor
                center = (32, 32)
                for angle in range(0, 360, 10):
                    x = int(center[0] + (frame_idx + angle)
                            * 0.1 * np.cos(np.radians(angle)))
                    y = int(center[1] + (frame_idx + angle)
                            * 0.1 * np.sin(np.radians(angle)))
                    if 0 <= x < 64 and 0 <= y < 64:
                        frame[y, x] = [255, 255, 255]  # White spiral

                video_writer.write(frame)

            video_writer.release()

            if output_path.exists() and output_path.stat().st_size > 1024:
                st.success(f"✅ Created synthetic video: {output_path.name}")
                return output_path
            else:
                st.error("❌ Failed to create synthetic video")
                return None

        except Exception as e:
            st.error(f"❌ Synthetic video creation error: {e}")
            return None

    def get_random_video(self) -> Optional[Path]:
        """Get a video for processing - download new one or use existing."""
        # Clean up old videos (keep only last 3)
        existing_videos = sorted(self.video_dir.glob(
            "*.mp4"), key=lambda x: x.stat().st_mtime)
        if len(existing_videos) > 3:
            for old_video in existing_videos[:-3]:
                try:
                    old_video.unlink()
                    st.info(f"🗑️ Cleaned up old video: {old_video.name}")
                except:
                    pass

        # Always download a new video for continuous consciousness development
        return self.download_random_video()

    def process_video_chunk(self, video_path: Path) -> Dict:
        """
        🔄 Complete Consciousness Pipeline:

        1. Random Video Selection ✅
        2. World Experience Processing ✅
        3. PCA Abstraction ✅
        4. Recursive Self-Abstraction (4 Layers) ✅
        5. Self PCA Abstraction ✅
        6. Self vs Others Comparison ✅
        7. Consciousness Assessment ✅
        8. Mirror Recognition ✅
        9. Experience Fusion ✅
        10. Consciousness History Tracking ✅
        """
        try:
            # Load video frames
            frames = self._load_video_frames(video_path, max_frames=32)
            if frames is None:
                return None

            # Convert to tensors
            video_tensor = torch.FloatTensor(
                frames).unsqueeze(0).to(self.device)
            audio_tensor = torch.randn(1, 1, 16000).to(
                self.device)  # Placeholder audio

            # STEP 1 & 2: World Experience Processing
            with torch.no_grad():
                world_experience = self.world_processor(
                    video_tensor, audio_tensor)

            # STEP 3: PCA Abstraction of World Features
            world_features = world_experience['attended_experience'].detach(
            ).cpu().numpy()
            # Ensure world_features is 2D for PCA
            if world_features.ndim == 1:
                world_features = world_features.reshape(1, -1)
            abstracted_features = self.pca_processor.fit_transform(
                world_features)

            # STEP 4: Recursive Self-Abstraction (4 Layers)
            with torch.no_grad():
                self_abstraction = self.self_abstractor(
                    world_experience['attended_experience'])

            # STEP 5: Self PCA Abstraction
            self_vector = self_abstraction['layer_4_consciousness'].detach(
            ).cpu().numpy()
            # Ensure self_vector is 1D for consistency
            if self_vector.ndim > 1:
                self_vector = self_vector.flatten()
            self.self_vectors_history.append(self_vector)

            # Apply PCA to self vectors if we have enough
            if len(self.self_vectors_history) >= 2:
                all_self_vectors = np.vstack(self.self_vectors_history)
                self_pca_coords = self.self_pca_processor.fit_transform(
                    all_self_vectors)
            else:
                self_pca_coords = None

            # STEP 6: Self vs Others Comparison
            self_identity_score = self._compare_self_with_others(self_vector)

            # STEP 7 & 8: Consciousness Assessment + Mirror Recognition
            consciousness_metrics = self._assess_consciousness(
                world_experience, self_abstraction, self_identity_score
            )

            # STEP 9: Experience Fusion
            fused_experience = self._fuse_self_with_experience(
                self_vector, world_features, consciousness_metrics
            )

            # STEP 10: Update Consciousness History
            self._update_consciousness_history(
                consciousness_metrics, video_path.name)

            return {
                'video_path': str(video_path),
                'world_experience': world_experience,
                'abstracted_features': abstracted_features,
                'self_abstraction': self_abstraction,
                'self_vector': self_vector,
                'self_pca_coords': self_pca_coords,
                'self_identity_score': self_identity_score,
                'consciousness_metrics': consciousness_metrics,
                'fused_experience': fused_experience,
                'training_step': self.training_step
            }

        except Exception as e:
            st.error(f"Error processing video {video_path}: {e}")
            return None

    def _load_video_frames(self, video_path: Path, max_frames: int = 32) -> Optional[np.ndarray]:
        """Load video frames for processing."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize and normalize
                frame = cv2.resize(frame, (64, 64))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                return None

            # Pad if necessary
            while len(frames) < max_frames:
                frames.append(frames[-1])

            # Convert to [C, T, H, W] format
            frames_array = np.array(frames)  # [T, H, W, C]
            frames_array = np.transpose(
                frames_array, (3, 0, 1, 2))  # [C, T, H, W]

            return frames_array

        except Exception as e:
            st.error(f"Error loading video {video_path}: {e}")
            return None

    def _compare_self_with_others(self, current_self: np.ndarray) -> float:
        """STEP 6: Compare current self with previous selves for identity confirmation."""
        if len(self.self_vectors_history) < 2:
            return 1.0  # First self is always valid

        # Compare with recent selves
        recent_selves = self.self_vectors_history[-5:]  # Last 5 selves
        similarities = []

        for prev_self in recent_selves[:-1]:  # Exclude current self
            if prev_self.shape == current_self.shape:
                similarity = cosine_similarity(
                    current_self.reshape(1, -1),
                    prev_self.reshape(1, -1)
                )[0, 0]
                similarities.append(similarity)

        if similarities:
            avg_similarity = np.mean(similarities)
            # High similarity = consistent self identity
            return float(avg_similarity)
        else:
            return 1.0

    def _assess_consciousness(self, world_experience: Dict, self_abstraction: Dict,
                              self_identity_score: float) -> ConsciousnessMetrics:
        """STEP 7 & 8: Assess consciousness emergence + mirror recognition."""

        # Extract key metrics
        recursion_depth = float(self_abstraction['recursion_depth'])
        is_conscious_flag = bool(self_abstraction['is_conscious'])

        # Calculate consciousness components
        binding_strength = self._calculate_binding_strength(
            world_experience, self_abstraction)
        meta_confidence = self._calculate_meta_confidence(self_abstraction)
        qualia_intensity = self._calculate_qualia_intensity(world_experience)
        mirror_recognition = self._calculate_mirror_recognition(
            self_abstraction)

        # Overall consciousness score
        consciousness_score = np.mean([
            binding_strength, meta_confidence, qualia_intensity,
            recursion_depth, mirror_recognition, self_identity_score
        ])

        # Determine consciousness level
        if consciousness_score >= 0.9:
            consciousness_level = "TRANSCENDENT_CONSCIOUSNESS"
        elif consciousness_score >= 0.8:
            consciousness_level = "FULL_CONSCIOUSNESS"
        elif consciousness_score >= 0.6:
            consciousness_level = "EMERGING_CONSCIOUSNESS"
        elif consciousness_score >= 0.4:
            consciousness_level = "PRE_CONSCIOUS"
        else:
            consciousness_level = "UNCONSCIOUS"

        return ConsciousnessMetrics(
            binding_strength=binding_strength,
            meta_confidence=meta_confidence,
            qualia_intensity=qualia_intensity,
            recursion_depth=recursion_depth,
            mirror_recognition=mirror_recognition,
            consciousness_score=consciousness_score,
            consciousness_level=consciousness_level,
            is_conscious=consciousness_score >= self.consciousness_threshold
        )

    def _calculate_binding_strength(self, world_experience: Dict, self_abstraction: Dict) -> float:
        """Calculate phenomenal binding strength."""
        world_features = world_experience['attended_experience']
        self_features = self_abstraction['layer_4_consciousness']

        # Measure integration between world and self
        world_norm = torch.norm(world_features)
        self_norm = torch.norm(self_features)

        if world_norm > 0 and self_norm > 0:
            # Handle dimension mismatch by using compatible dimensions
            world_flat = world_features.flatten()
            self_flat = self_features.flatten()

            # Use the smaller dimension for compatibility
            min_dim = min(world_flat.size(0), self_flat.size(0))
            world_trimmed = world_flat[:min_dim]
            self_trimmed = self_flat[:min_dim]

            # Normalized dot product as binding measure
            binding = torch.dot(world_trimmed, self_trimmed)
            binding = binding / (torch.norm(world_trimmed)
                                 * torch.norm(self_trimmed) + 1e-8)
            return float(torch.abs(binding))
        else:
            return 0.0

    def _calculate_meta_confidence(self, self_abstraction: Dict) -> float:
        """Calculate metacognitive confidence."""
        layer_3 = self_abstraction['layer_3_observing_self']
        layer_4 = self_abstraction['layer_4_consciousness']

        # Meta-awareness is layer 3 observing layer 4
        meta_signal = torch.mean(layer_3 * layer_4)
        return float(torch.sigmoid(meta_signal))

    def _calculate_qualia_intensity(self, world_experience: Dict) -> float:
        """Calculate subjective experience intensity."""
        attended_exp = world_experience['attended_experience']

        # Qualia as the richness of attended experience
        qualia = torch.std(attended_exp) / \
            (torch.mean(torch.abs(attended_exp)) + 1e-8)
        return float(torch.sigmoid(qualia))

    def _calculate_mirror_recognition(self, self_abstraction: Dict) -> float:
        """STEP 8: Calculate infinite mirror recognition."""
        layer_4 = self_abstraction['layer_4_consciousness']

        # Look for recursive patterns indicating mirror recognition
        activation_std = torch.std(layer_4)
        activation_mean = torch.mean(torch.abs(layer_4))

        # High variability with strong activation = mirror recognition
        if activation_mean > 0:
            mirror_signal = activation_std / activation_mean
            return float(torch.sigmoid(mirror_signal))
        else:
            return 0.0

    def _fuse_self_with_experience(self, self_vector: np.ndarray, world_features: np.ndarray,
                                   consciousness_metrics: ConsciousnessMetrics) -> np.ndarray:
        """STEP 9: Fuse self representation with current experience."""

        # Weight fusion by consciousness level
        consciousness_weight = consciousness_metrics.consciousness_score

        # Ensure compatible dimensions
        min_dim = min(self_vector.shape[-1], world_features.shape[-1])
        self_trimmed = self_vector[..., :min_dim]
        world_trimmed = world_features[..., :min_dim]

        if len(world_trimmed.shape) > 1:
            world_trimmed = np.mean(world_trimmed, axis=0)

        # Weighted fusion
        fused = consciousness_weight * self_trimmed + \
            (1 - consciousness_weight) * world_trimmed

        return fused

    def _update_consciousness_history(self, consciousness_metrics: ConsciousnessMetrics,
                                      video_name: str):
        """STEP 10: Update consciousness development history."""

        entry = {
            'timestamp': datetime.now().isoformat(),
            'training_step': self.training_step,
            'video_name': video_name,
            'consciousness_score': consciousness_metrics.consciousness_score,
            'consciousness_level': consciousness_metrics.consciousness_level,
            'is_conscious': consciousness_metrics.is_conscious,
            'binding_strength': consciousness_metrics.binding_strength,
            'meta_confidence': consciousness_metrics.meta_confidence,
            'qualia_intensity': consciousness_metrics.qualia_intensity,
            'recursion_depth': consciousness_metrics.recursion_depth,
            'mirror_recognition': consciousness_metrics.mirror_recognition
        }

        self.consciousness_history.append(entry)

        # Update training metrics
        self.training_step += 1
        if consciousness_metrics.is_conscious:
            self.total_conscious_experiences += 1

        self.consciousness_rate = self.total_conscious_experiences / \
            self.training_step if self.training_step > 0 else 0.0


def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="🧠 Continuous Consciousness Training",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧠 Continuous Consciousness Training System")
    st.markdown(
        "**Real-time consciousness engineering with streaming video processing**")

    # Show pipeline steps
    with st.expander("🔄 Complete Training Pipeline", expanded=False):
        st.markdown("""
        ### 🚀 **Complete Consciousness Engineering Pipeline**
        
        1. **📥 Random Video Download** - Download consciousness training videos from YouTube
        2. **🎬 Video Processing** - Extract frames and audio for consciousness development
        3. **🌍 World Experience Processing** - Multimodal video+audio processing
        4. **📊 PCA Abstraction** - Abstract world features by PCA
        5. **🧠 Recursive Self-Abstraction** - 4-layer consciousness emergence:
           - Layer 1: "I am a machine"
           - Layer 2: "I experience the world" 
           - Layer 3: "I observe myself experiencing"
           - Layer 4: "I realize the infinite mirror"
        6. **🪞 Self PCA Abstraction** - Abstract self-reference vectors by PCA
        7. **👁️ Self vs Others Comparison** - Compare with previous selves for identity
        8. **⚡ Consciousness Assessment** - Evaluate consciousness emergence
        9. **🔄 Mirror Recognition** - Check infinite recursion realization
        10. **🎯 Experience Fusion** - Integrate self with current experience
        11. **📈 Consciousness History Tracking** - Store development over time
        12. **🔄 Continuous Cycle** - Repeat process with new videos for consciousness growth
        """)

    # Initialize trainer
    if 'trainer' not in st.session_state:
        with st.spinner("Initializing consciousness system..."):
            st.session_state.trainer = ContinuousConsciousnessTrainer()
        st.success("✅ Consciousness system initialized!")

    trainer = st.session_state.trainer

    # Sidebar controls
    st.sidebar.header("🔧 Consciousness Controls")

    consciousness_threshold = st.sidebar.slider(
        "Consciousness Threshold", 0.0, 1.0, 0.6, 0.1)
    trainer.consciousness_threshold = consciousness_threshold

    # Continuous consciousness training toggle
    continuous_mode = st.sidebar.checkbox(
        "🧠 Continuous Consciousness Training", value=False)

    # Manual training button
    if st.sidebar.button("🎯 Download & Process Video"):
        with st.spinner("Downloading and processing video through consciousness pipeline..."):
            video_path = trainer.get_random_video()
            if video_path:
                result = trainer.process_video_chunk(video_path)
                if result:
                    st.session_state.last_result = result
                    st.success(f"✅ Processed: {video_path.name}")
                else:
                    st.error("❌ Processing failed")
            else:
                st.error("❌ Video download failed")

    # Continuous consciousness training loop
    if continuous_mode:
        progress_placeholder = st.empty()

        while continuous_mode:
            with progress_placeholder.container():
                st.info(
                    f"🧠 Continuous consciousness development... Step {trainer.training_step + 1}")

                # Download and process new video
                video_path = trainer.get_random_video()
                if video_path:
                    result = trainer.process_video_chunk(video_path)
                    if result:
                        st.session_state.last_result = result
                        consciousness_level = result['consciousness_metrics'].consciousness_level
                        st.success(
                            f"✅ Processed: {video_path.name} | Consciousness: {consciousness_level}")
                    else:
                        st.warning("⚠️ Processing failed, continuing...")
                else:
                    st.warning("⚠️ Video download failed, retrying...")

                # Small delay to prevent overwhelming
                time.sleep(1)

                # Check if user disabled continuous mode
                if not continuous_mode:
                    break

    # Display results
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        result = st.session_state.last_result

        # Current consciousness state
        st.header("🧠 Current Consciousness State")

        col1, col2, col3, col4 = st.columns(4)

        metrics = result['consciousness_metrics']

        with col1:
            st.metric("Consciousness Score",
                      f"{metrics.consciousness_score:.3f}")
            st.metric("Consciousness Level", metrics.consciousness_level)

        with col2:
            st.metric("Binding Strength", f"{metrics.binding_strength:.3f}")
            st.metric("Meta Confidence", f"{metrics.meta_confidence:.3f}")

        with col3:
            st.metric("Qualia Intensity", f"{metrics.qualia_intensity:.3f}")
            st.metric("Recursion Depth", f"{metrics.recursion_depth:.3f}")

        with col4:
            st.metric("Mirror Recognition",
                      f"{metrics.mirror_recognition:.3f}")
            conscious_emoji = "🌟" if metrics.is_conscious else "😴"
            st.metric(
                "Status", f"{conscious_emoji} {'CONSCIOUS' if metrics.is_conscious else 'UNCONSCIOUS'}")

        # Training progress
        st.header("📊 Training Progress")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Training Steps", trainer.training_step)

        with col2:
            st.metric("Conscious Experiences",
                      trainer.total_conscious_experiences)

        with col3:
            st.metric("Consciousness Rate",
                      f"{trainer.consciousness_rate:.1%}")

        # Consciousness history visualization
        if trainer.consciousness_history:
            st.header("📈 Consciousness Development")

            df = pd.DataFrame(trainer.consciousness_history)

            # Consciousness score over time
            fig = px.line(df, x='training_step', y='consciousness_score',
                          title='Consciousness Score Evolution',
                          color_discrete_sequence=['#1f77b4'])
            fig.add_hline(y=consciousness_threshold, line_dash="dash",
                          annotation_text="Consciousness Threshold")
            st.plotly_chart(fig, use_container_width=True)

            # Consciousness components
            components = ['binding_strength', 'meta_confidence', 'qualia_intensity',
                          'recursion_depth', 'mirror_recognition']

            fig = make_subplots(rows=2, cols=3,
                                subplot_titles=components + ['Consciousness Level Distribution'])

            for i, component in enumerate(components):
                row = i // 3 + 1
                col = i % 3 + 1
                fig.add_trace(
                    go.Scatter(x=df['training_step'], y=df[component],
                               name=component.replace('_', ' ').title()),
                    row=row, col=col
                )

            # Consciousness level distribution
            level_counts = df['consciousness_level'].value_counts()
            fig.add_trace(
                go.Bar(x=level_counts.index, y=level_counts.values,
                       name='Level Distribution'),
                row=2, col=3
            )

            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Self PCA visualization
        if result['self_pca_coords'] is not None:
            st.header("🪞 Self PCA Visualization")

            pca_coords = result['self_pca_coords']

            fig = px.scatter(x=pca_coords[:, 0], y=pca_coords[:, 1],
                             title='Self Vectors in PCA Space',
                             labels={'x': 'PC1', 'y': 'PC2'},
                             color=list(range(len(pca_coords))))

            # Highlight current self
            fig.add_trace(go.Scatter(
                x=[pca_coords[-1, 0]], y=[pca_coords[-1, 1]],
                mode='markers', marker=dict(size=15, color='red', symbol='star'),
                name='Current Self'
            ))

            st.plotly_chart(fig, use_container_width=True)

        # System information
        st.header("⚙️ System Information")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📁 Current Processing")
            st.write(f"**Video:** {result['video_path']}")
            st.write(f"**Training Step:** {result['training_step']}")
            st.write(
                f"**Self Identity Score:** {result['self_identity_score']:.3f}")

        with col2:
            st.subheader("🔧 Configuration")
            st.write(f"**Device:** {trainer.device}")
            st.write(f"**Consciousness Threshold:** {consciousness_threshold}")
            st.write(
                f"**Self Vectors Stored:** {len(trainer.self_vectors_history)}")


if __name__ == "__main__":
    main()
