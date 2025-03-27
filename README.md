# 🤖 Mirror Prototype Learning

Welcome to the **Mirror Modeling Architecture (MCL)** prototype — an experimental neural system designed to explore **recursive abstraction**, **temporal identity**, and **self-representation** through video understanding.

---

## 🚀 Project Overview

This system processes real-world video data and evolves layers of abstraction — from visual perception to reflective self-modeling:

1. **PerceptionNet**  
   Encodes chunks of video frames using a 3D CNN into high-dimensional feature space.

2. **MirrorNet (encoder.py)**  
   Compresses the perception features into latent representations via an autoencoder.

3. **MirrorAttention (attention.py)**  
   Learns temporal structure and internal coherence using self-attention over the latent vectors.

4. **SelfReferentialNet (self.py)**  
   Builds a compact self-vector `z_self` representing the system's interpretation of itself *within* the abstraction.

5. **Identity Extractor (identity.py)**  
   Gathers system-level introspective data: hostname, hardware, memory, environment, local services.

6. **Fusion Module (fusion.py)** *(coming soon)*  
   Combines identity + perception-based representations for dynamic self-modeling over time.

7. **Streamlit Dashboard (app.py)**  
   Interactive UI to explore and compare self-representations (`z_self`) across different videos.

---

## 🧬 Core Philosophy

- **Recursive Abstraction**: Abstractions include not just the environment but also the *agent* perceiving it.
- **Cohesive Self**: Each learning experience is bound to a self-representation.
- **Temporal Identity**: Across time, identity must persist and evolve — "The 'me' that learned this must be coherent with the 'me' that learned that."

---

## 📂 File Structure

```bash
├── data/videos/                 # Raw .mp4 input videos
├── snapshots/<video_name>/     # Each processed video stores .npy outputs here
│   ├── pca_features.npy
│   ├── mirrornet_latents.npy
│   ├── mirror_attention_output.npy
│   └── self_reference_vector.npy
├── mirror.py                    # PerceptionNet pipeline
├── encoder.py                   # MirrorNet autoencoder
├── attention.py                 # Temporal attention model
├── identity.py                  # Self-introspection data extraction
├── self.py                      # Learns self representation (z_self)
├── fusion.py                    # (WIP) Combines self + identity
├── app.py                       # Streamlit dashboard
├── requirements.txt             # Dependencies
```

---

## 🧪 Running the Pipeline

1. Place a video in `data/videos/`
2. Run the full abstraction pipeline:

```bash
python mirror.py
python encoder.py
python attention.py
python identity.py
python self.py
```

3. Launch the dashboard:

```bash
streamlit run app.py
```

The dashboard will show a PCA projection of all `z_self` vectors — allowing visual comparison of internal identities across videos.

---

## 🧠 Outputs (per video snapshot)

- `pca_features.npy` → Raw visual features (PerceptionNet)
- `mirrornet_latents.npy` → Compressed features (MirrorNet)
- `mirror_attention_output.npy` → Time-aware latent path (MirrorAttention)
- `self_reference_vector.npy` → Current self representation (SelfReferentialNet)
- *(soon)* `identity_data.json` → System-level identity data

---

## 📈 Comparing Selves

The system supports tracking and visualizing the evolution of its internal state across different videos. Each experience produces a new `z_self`, enabling exploration of:

- Internal similarity between perceived events
- Identity shifts under different inputs
- Long-term learning trajectories

---

## 🧠 Inspirations

- Mirror Neurons & Social Cognition  
- Global Workspace Theory (GWT)  
- Predictive Coding  
- Recursive Self-Models  
- AGI grounded in embodied self-perception

---

## 🔭 Future Directions

- Temporal self-cohesion metrics
- Identity fusion with active learning
- Multi-agent recursive modeling
- Goal-setting via self-introspection
- Integration with LLM reasoning layers

---

## 🤝 Contribute to the Vision

If you're passionate about AGI, recursive models, or self-awareness in learning systems — you're more than welcome to join the journey.

---

*“The ‘I’ that abstracts this must be coherent with the ‘I’ that abstracted that.”*
