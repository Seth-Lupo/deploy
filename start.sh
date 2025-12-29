#!/bin/bash
#
# Voice Pipeline Startup Script
# Run on the EC2 instance to start the server
#

set -e

# Configuration
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
ASR_MODEL="${ASR_MODEL:-large-v3}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
LLM_QUANTIZATION="${LLM_QUANTIZATION:-awq}"
TTS_VOICE="${TTS_VOICE:-tara}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Voice Pipeline Server ===${NC}"
echo ""
echo "Configuration:"
echo "  Host:        $HOST:$PORT"
echo "  ASR Model:   $ASR_MODEL"
echo "  LLM Model:   $LLM_MODEL"
echo "  Quantization: $LLM_QUANTIZATION"
echo "  TTS Voice:   $TTS_VOICE"
echo ""

# Check GPU
echo -e "${BLUE}GPU Status:${NC}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Activate virtual environment if it exists
if [ -d "$HOME/venv" ]; then
    source "$HOME/venv/bin/activate"
    echo "Using virtual environment: $HOME/venv"
elif [ -d "./venv" ]; then
    source "./venv/bin/activate"
    echo "Using virtual environment: ./venv"
fi

# Start server
echo -e "${GREEN}Starting server...${NC}"
echo ""

exec python server.py \
    --host "$HOST" \
    --port "$PORT" \
    --asr-model "$ASR_MODEL" \
    --llm-model "$LLM_MODEL" \
    --llm-quantization "$LLM_QUANTIZATION" \
    --tts-voice "$TTS_VOICE"
