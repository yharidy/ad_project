FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install core packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python-is-python3 \
    git curl unzip nano sudo \
    libgl1-mesa-glx libglib2.0-0 libxext6 libxrender1 libsm6 ffmpeg \
    build-essential cmake g++ \
    && rm -rf /var/lib/apt/lists/*

# Set compiler flags for MMCV and other C++ libs
ENV CXXFLAGS="-std=c++17"

# Optional: non-root user
RUN useradd -ms /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER vscode
WORKDIR /workspace

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt