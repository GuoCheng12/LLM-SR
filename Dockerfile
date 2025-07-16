# LLM-SR Docker Configuration
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set working directory
WORKDIR /workspace

# Copy requirements files
COPY requirements_minimal.txt .
COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
RUN pip install -r requirements_minimal.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p logs data/cache

# Set proper permissions
RUN chmod +x setup.sh run_server.sh run_llmsr.sh

# Expose ports for local LLM server
EXPOSE 5000

# Default command
CMD ["bash"]