# RL for Real - Docker Image
# 
# Build:
#   docker build -t rl-for-real .
#
# Run (interactive with GPU):
#   docker run --gpus all -it --rm -v $(pwd)/outputs:/app/outputs rl-for-real
#
# Run training:
#   docker run --gpus all -it --rm -v $(pwd)/outputs:/app/outputs rl-for-real \
#       python train_grpo.py --config config/qwen_1.5B_syllogism.yaml

# Base image with CUDA 12.8 and Python 3.10
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install PyTorch with CUDA 12.8, then other dependencies
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    trl>=0.12.0 \
    peft>=0.10.0 \
    reasoning-gym>=0.1.19 \
    wandb>=0.16.0 \
    tqdm>=4.65.0 \
    pyyaml>=6.0

# Copy project files
COPY . .

# Create outputs directory
RUN mkdir -p outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "train_grpo.py", "--config", "config/qwen_1.5B_syllogism.yaml"]
