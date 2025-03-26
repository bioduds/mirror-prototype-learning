# 🧠 Mirror Prototype Learning

Welcome to the Mirror Modeling Architecture (MCL) prototype — an experimental neural framework inspired by mirror neurons, self-representation, and recursive abstraction.

## 🚀 Project Overview

This system is designed to process real-world video (e.g., football plays) and develop multiple layers of understanding:

1. **PerceptionNet**  
   Encodes sequences of video frames into abstract visual features using a 3D CNN.

2. **MirrorNet**  
   Compresses those features into a more compact latent space — simulating the process of mirroring.

3. **MirrorAttention**  
   Applies temporal self-attention to the sequence of latent vectors to learn context and flow.

4. **SelfReferentialNet**  
   Encodes the system’s own internal trajectory into a *single self-representation vector*, `z_self`.

5. **Gradio Dashboard**  
   A browser-based interface for visualizing PCA of all these layers, including the location of the system’s self.

---

## 📂 File Structure

```bash
├── data/videos/               # Input videos (e.g., football games)
├── mirror.py                  # Extract PerceptionNet features and PCA
├── encoder.py                 # Trains MirrorNet autoencoder
├── attention.py               # Computes self-attended latent trajectory
├── self.py                    # Learns self-representation vector
├── app.py                     # Gradio dashboard for visualization
├── requirements.txt           # Python dependencies
├── README.md                  # You're here!
```

---

## 🧪 Run the System

1. Add your `.mp4` video to `data/videos/`
2. Run each processing step in order:

```bash
python mirror.py
python encoder.py
python attention.py
python self.py
```

3. Launch the dashboard:

```bash
python app.py
```

---

## 🌱 Core Concepts

- **Abstraction Spiral**: Layers evolve from raw input to recursive self-reference.
- **Self-Inclusion**: The model includes itself in its own abstractions.
- **Dynamic Identity**: Each video can produce a unique `z_self`, showing self-perception across time.

---

## 📊 Outputs

- `pca_coords.npy`: PerceptionNet PCA projection
- `mirrornet_latents.npy`: Encoded MirrorNet features
- `mirror_attention_output.npy`: Attention-refined latent sequence
- `self_reference_vector.npy`: The model’s compressed self

---

## 👁️‍🗨️ Try Comparing Selves

You can run the pipeline with different videos and snapshot each `z_self` for comparison. Over time, this lets you explore how the system’s **internal identity shifts with experience.**

---

## 🧠 Inspired By

- Mirror Neurons
- Global Workspace Theory (GWT)
- Predictive Processing
- Self-modeling agents and recursive abstraction

---

## 📬 Future Directions

- Multi-video training & temporal identity tracking
- Reinforcement learning with self-referenced goals
- Emergent planning via latent introspection

---

## 🤝 Contributions Welcome

If this interests you, join the project! We’re pushing the boundaries of what abstract learning systems can become.

---
