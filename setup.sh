#!/bin/bash
# EC2 GPU Instance Setup Script (FP16 Models)
# Optimized for A100/A10G GPUs (g5.xlarge, p4d.24xlarge)
# Run this on Ubuntu 22.04 Deep Learning AMI

set -e

echo "==========================================="
echo "Voice Pipeline EC2 Setup (FP16)"
echo "==========================================="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver not found!"
    echo "Please use an AMI with pre-installed NVIDIA drivers:"
    echo "  - Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"
    exit 1
fi

echo "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Check GPU memory (recommend 24GB+)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU Memory: ${GPU_MEM} MiB"
if [ "$GPU_MEM" -lt 16000 ]; then
    echo "WARNING: GPU has less than 16GB VRAM. FP16 models may not fit."
    echo "Recommended: g5.xlarge (24GB A10G) or larger"
fi

# System dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-full \
    python3-venv \
    python3-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    curl

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and accelerate
echo "Installing Transformers and Accelerate..."
pip install transformers accelerate sentencepiece

# Install chatterbox-tts
echo "Installing Chatterbox TTS..."
pip install chatterbox-tts

# Install remaining dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Pre-download models
echo "Pre-downloading models (this may take a while)..."
python3 -c "
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Download Qwen3 4B Instruct
print('Downloading Qwen3-4B-Instruct-2507...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', trust_remote_code=True)
# Just download, don't load to GPU
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-4B-Instruct-2507',
    torch_dtype=torch.float16,
    device_map='cpu',
    trust_remote_code=True
)
del model
print('Qwen3 downloaded!')

# Download Parakeet ASR
print('Downloading Parakeet ASR...')
import onnx_asr
model = onnx_asr.load_model('smcleod/parakeet-tdt-0.6b-v2')
del model
print('Parakeet downloaded!')

# Chatterbox will download on first use
print('Models downloaded!')
"

echo "==========================================="
echo "Setup complete!"
echo ""
echo "GPU Memory Requirements:"
echo "  - ASR (Parakeet FP16): ~1.2GB"
echo "  - LLM (Qwen3 4B FP16): ~8GB"
echo "  - TTS (Chatterbox FP16): ~3GB"
echo "  - Total: ~12-13GB (fits on 16GB+)"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python server.py --port 8765"
echo ""
echo "To connect from your local machine:"
echo "  python client.py --server ws://<EC2_IP>:8765/ws"
echo "==========================================="
