#!/bin/bash

# LLM-SR Environment Setup Script
# This script helps users set up the environment for LLM-SR

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed. Please install Anaconda or Miniconda first."
        echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_status "Conda found: $(conda --version)"
}

# Check Python version
check_python() {
    python_version=$(python --version 2>&1 | awk '{print $2}')
    required_version="3.11.0"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_warning "Python version $python_version found. Recommended: Python >= 3.11"
    else
        print_status "Python version $python_version is compatible"
    fi
}

# Check CUDA availability
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "CUDA detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)"
        return 0
    else
        print_warning "CUDA not detected. CPU-only installation will be used."
        return 1
    fi
}

# Create conda environment
create_environment() {
    print_step "Creating conda environment 'llmsr'..."
    
    # Check if environment already exists
    if conda env list | grep -q "llmsr"; then
        print_warning "Environment 'llmsr' already exists."
        read -p "Do you want to remove it and recreate? (y/n): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n llmsr -y
        else
            print_status "Using existing environment."
            return 0
        fi
    fi
    
    # Create environment with Python 3.11
    conda create -n llmsr python=3.11.7 -y
    print_status "Environment 'llmsr' created successfully."
}

# Install dependencies
install_dependencies() {
    print_step "Installing dependencies..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate llmsr
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support if available
    if check_cuda; then
        print_status "Installing PyTorch with CUDA support..."
        pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch CPU-only version..."
        pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    print_status "Installing other dependencies..."
    pip install -r requirements_minimal.txt
    
    print_status "Dependencies installed successfully."
}

# Verify installation
verify_installation() {
    print_step "Verifying installation..."
    
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate llmsr
    
    # Test imports
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import numpy, pandas, scipy, matplotlib; print('Core packages imported successfully')"
    
    print_status "Installation verification completed."
}

# Display next steps
show_next_steps() {
    print_step "Installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the environment: conda activate llmsr"
    echo "2. Set up your API key (if using OpenAI): export API_KEY=your_api_key_here"
    echo "3. Run a test: python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/test"
    echo ""
    echo "For local LLM server:"
    echo "1. Start server: bash run_server.sh"
    echo "2. Run LLM-SR: bash run_llmsr.sh"
    echo ""
    echo "For help, see README.md or run: python main.py --help"
}

# Main installation process
main() {
    echo "=============================================="
    echo "    LLM-SR Environment Setup"
    echo "=============================================="
    
    print_step "Starting environment setup..."
    
    # Check requirements
    check_conda
    
    # Create environment and install dependencies
    create_environment
    install_dependencies
    
    # Verify installation
    verify_installation
    
    # Show next steps
    show_next_steps
}

# Run main function
main "$@"