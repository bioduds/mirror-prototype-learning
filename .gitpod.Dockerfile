FROM python:3.10-slim

# Install system dependencies including git and OpenCV deps
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
