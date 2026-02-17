FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install Python, pip, and other essentials in a single layer
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as the default python command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python packages
RUN pip install --no-cache-dir \
    timm \
    torchmetrics \
    numpy \
    pandas \
    tqdm \
    scikit-learn \
    matplotlib \
    seaborn \
    ultralytics \
    onnx \
    onnxruntime \
    onnxslim \
    opencv-python

# Set the working directory inside the container
WORKDIR /workspace