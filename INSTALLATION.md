# LLM-SR Installation Guide

This guide provides multiple ways to install and set up LLM-SR for different use cases and platforms.

## Quick Start (Recommended)

### 1. Automated Installation

The easiest way to get started is using our automated setup script:

```bash
# Clone the repository
git clone https://github.com/your-repo/LLM-SR.git
cd LLM-SR

# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

This script will:
- Check system requirements
- Create a conda environment
- Install all dependencies
- Verify the installation

### 2. Manual Installation

If you prefer manual installation or need more control:

#### Prerequisites
- Python 3.11 or higher
- Conda or Miniconda
- CUDA 11.8+ (optional, for GPU support)

#### Step-by-step installation:

```bash
# 1. Create conda environment
conda create -n llmsr python=3.11.7
conda activate llmsr

# 2. Install PyTorch (choose one based on your system)
# For CUDA 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# 3. Install other dependencies
pip install -r requirements_minimal.txt

# 4. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Installation Options

### Option 1: Using Conda Environment File (Full Environment)

```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate llmsr
```

### Option 2: Using pip requirements (Minimal)

```bash
# Install only essential packages
pip install -r requirements_minimal.txt
```

### Option 3: Using full requirements (Complete)

```bash
# Install all packages (including optional ones)
pip install -r requirements.txt
```

## Docker Installation

### Using Docker Compose

```bash
# Build and run with docker-compose
docker-compose up --build
```

### Using Docker directly

```bash
# Build the image
docker build -t llmsr .

# Run the container
docker run -it --gpus all -v $(pwd):/workspace llmsr bash
```

## Platform-Specific Instructions

### Windows

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open Anaconda Prompt as Administrator
3. Follow the manual installation steps above

### macOS

1. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Miniconda:
   ```bash
   brew install miniconda
   ```
3. Run the setup script or follow manual installation

### Linux

1. Install Miniconda:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
2. Run the setup script or follow manual installation

## Verification

After installation, verify everything works:

```bash
# Activate environment
conda activate llmsr

# Test basic functionality
python -c "import llmsr; print('LLM-SR imported successfully')"

# Test GPU availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run a quick test
python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/test
```

## Common Issues and Solutions

### Issue 1: CUDA not found
```bash
# Solution: Install CUDA-compatible PyTorch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: Permission denied for setup.sh
```bash
# Solution: Make script executable
chmod +x setup.sh
```

### Issue 3: Conda environment conflicts
```bash
# Solution: Remove and recreate environment
conda env remove -n llmsr
conda create -n llmsr python=3.11.7
```

### Issue 4: Out of memory errors
```bash
# Solution: Reduce batch size or use CPU-only mode
# Edit config.py and set smaller values for:
# - samples_per_prompt
# - num_evaluators
```

## Environment Variables

Set these environment variables for optimal performance:

```bash
# For OpenAI API usage
export API_KEY=your_openai_api_key

# For local LLM server
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU

# For debugging
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Next Steps

After successful installation:

1. **For API usage**: Set your OpenAI API key and run examples
2. **For local LLM**: Start the local server with `bash run_server.sh`
3. **For development**: Check `CLAUDE.md` for development guidelines
4. **For examples**: See `run_llmsr.sh` for usage examples

## Support

If you encounter issues:
1. Check the [Common Issues](#common-issues-and-solutions) section
2. Ensure your system meets the requirements
3. Try the Docker installation as an alternative
4. Open an issue on GitHub with your error details

## System Requirements

### Minimum Requirements
- Python 3.11+
- 8GB RAM
- 10GB disk space

### Recommended Requirements
- Python 3.11+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 50GB+ disk space (for local LLM models)

### Tested Platforms
- Ubuntu 20.04/22.04
- CentOS 7/8
- macOS 12+
- Windows 10/11 (with WSL2 recommended)