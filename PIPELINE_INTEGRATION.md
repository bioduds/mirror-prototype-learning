# Mirror Prototype Learning - Pipeline Integration

## Overview

This document describes the reintegrated pipeline for the mirror-prototype-learning project.

## Pipeline Architecture

### Core Concept

The project implements **mirror learning** - a series of neural networks that process video data through multiple stages to develop self-awareness and consciousness representation, inspired by mirror neurons in biological systems.

### Pipeline Stages (In Order)

1. **mirror.py** - Perception & Feature Extraction
   - **Purpose**: Initial video processing and feature extraction
   - **Input**: Video files from `data/videos/*.mp4`
   - **Output**: `pca_features.npy`, `pca_coords.npy`
   - **Description**: Loads videos, splits into 64-frame chunks, applies PCA dimensionality reduction

2. **encoder.py** - MirrorNet Autoencoder  
   - **Purpose**: Learn compressed representations via autoencoder
   - **Input**: `pca_features.npy`
   - **Output**: `mirrornet_latents.npy`, `mirrornet_reconstructed.npy`
   - **Description**: Neural autoencoder that learns to compress and reconstruct PCA features

3. **attention.py** - Temporal Attention
   - **Purpose**: Apply self-attention over temporal sequences
   - **Input**: `mirrornet_latents.npy`
   - **Output**: `mirror_attention_output.npy`
   - **Description**: Multi-head attention mechanism to create temporal relationships

4. **self.py** - Self-Reference Learning
   - **Purpose**: Develop compressed self-representation
   - **Input**: `mirror_attention_output.npy`
   - **Output**: `self_reference_vector.npy`
   - **Description**: GRU-based model that learns to predict final state from internal representation

5. **fusion.py** - Consciousness Fusion
   - **Purpose**: Combine self-reference with experience vectors
   - **Input**: `self_reference_vector.npy`, `mirror_attention_output.npy`
   - **Output**: `fused_consciousness_vectors.npy`
   - **Description**: Neural fusion layer creating unified consciousness representations

6. **extractor.py** - CLIP Feature Extraction (Parallel)
   - **Purpose**: Extract semantic features for correlation analysis
   - **Input**: Video files from `data/videos/*.mp4`
   - **Output**: `clip_features.npy`
   - **Description**: CLIP-based semantic feature extraction (runs in parallel for analysis)

7. **clustering.py** - Pattern Analysis
   - **Purpose**: Cluster and analyze consciousness patterns
   - **Input**: `fused_consciousness_vectors.npy`, `clip_features.npy`
   - **Output**: `clustering_results.npy`
   - **Description**: K-means clustering with CLIP correlation analysis

## Integration Components

### 1. Pipeline Runner (`pipeline_runner.py`)

- **Comprehensive pipeline execution** with error handling
- **Progress tracking** and logging
- **Flexible execution** (full pipeline or individual stages)
- **Output validation** for each stage
- **Results summary** generation

### 2. Streamlit Dashboard (`mirror_dashboard.py`)

- **Interactive pipeline control** with individual stage buttons
- **Real-time status monitoring** for all stages
- **Data visualization** for each pipeline output
- **Log viewing** and progress tracking
- **Auto-refresh** capabilities

### 3. Expected File Structure

```
mirror-prototype-learning/
├── data/
│   └── videos/          # Input video files (.mp4)
├── vectors/             # Vector storage (optional organization)
├── mirror.py           # Stage 1: Perception
├── encoder.py          # Stage 2: Autoencoder
├── attention.py        # Stage 3: Attention
├── self.py            # Stage 4: Self-reference
├── fusion.py          # Stage 5: Fusion
├── extractor.py       # Stage 6: CLIP extraction
├── clustering.py      # Stage 7: Clustering
├── pipeline_runner.py  # Pipeline orchestration
├── mirror_dashboard.py # Streamlit interface
└── README.md          # Documentation
```

## Usage Instructions

### 1. Prerequisites

- Place video files in `data/videos/` directory
- Ensure all required Python packages are installed
- Verify all pipeline scripts exist

### 2. Running the Pipeline

#### Full Pipeline

```bash
python pipeline_runner.py
```

#### Individual Stages

```bash
python pipeline_runner.py --stage mirror
python pipeline_runner.py --stage encoder
# ... etc
```

#### Starting from Specific Stage

```bash
python pipeline_runner.py --start-from encoder.py
```

### 3. Interactive Dashboard

```bash
streamlit run mirror_dashboard.py
```

The dashboard provides:

- **Real-time pipeline status**
- **Individual stage controls**
- **Data visualization** for each stage output
- **Log monitoring**

### 4. Output Analysis

Each stage produces specific outputs that can be analyzed:

- **PCA Analysis**: Video chunk clustering and feature distribution
- **Autoencoder Analysis**: Latent space and reconstruction quality
- **Attention Analysis**: Temporal attention patterns and correlations
- **Self-Reference Analysis**: Self-representation vector components
- **Consciousness Fusion**: Combined consciousness evolution over time
- **Semantic Analysis**: CLIP feature semantic understanding
- **Clustering Analysis**: Pattern discovery and consciousness clustering

## Key Features

### 1. Robust Error Handling

- **Timeout protection** (5 minutes per stage)
- **Prerequisite checking** (videos, scripts)
- **Output validation** for each stage
- **Graceful failure** with detailed error reporting

### 2. Progress Tracking

- **Timestamped logging** in `pipeline_log.txt`
- **JSON results** saved for each run
- **Real-time status** in Streamlit dashboard
- **File modification** tracking

### 3. Flexibility

- **Run individual stages** or full pipeline
- **Resume from any stage** if needed
- **Interactive control** via web interface
- **Batch processing** capabilities

### 4. Visualization

- **Real-time data plots** for each stage
- **PCA and t-SNE** visualizations
- **Correlation analysis** between stages
- **Cluster analysis** results

## Troubleshooting

### Common Issues

1. **Missing video files**: Ensure `.mp4` files exist in `data/videos/`
2. **Script not found**: Verify all pipeline scripts exist in root directory
3. **Memory issues**: Reduce video chunk size or use smaller videos
4. **Timeout errors**: Increase timeout in `pipeline_runner.py` if needed

### Debug Mode

- Check `pipeline_log.txt` for detailed execution logs
- Use `--stage` to run individual components for debugging
- Monitor file generation in real-time via dashboard

## Future Enhancements

- **Multi-video processing** for batch analysis
- **Parameter tuning** interface in dashboard
- **Model comparison** between different runs
- **Export capabilities** for research analysis
- **Integration with external tools** (TensorBoard, W&B)

This integration provides a complete, robust system for mirror prototype learning research with both automated execution and interactive analysis capabilities.
