FROM python:3.10-slim

# Install system-level dependencies (e.g. OpenCV needs libgl)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set default working dir
WORKDIR /workspace
