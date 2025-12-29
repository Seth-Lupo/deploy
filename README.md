# Voice Pipeline - EC2 GPU Deployment (FP16)

A streaming voice conversation pipeline with ASR → LLM → TTS, designed for real-time conversations on EC2 GPU instances with FP16 precision.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Client (Your Mac)                          │
│                                                                      │
│  ┌──────────────┐    WebSocket    ┌──────────────────────────────┐  │
│  │  Microphone  │ ──────────────► │  Audio + Text + Metrics      │  │
│  │   Speaker    │ ◄────────────── │  Bidirectional Streaming     │  │
│  └──────────────┘                 └──────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EC2 GPU Instance (g5.xlarge+)                    │
│                                                                      │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────────┐  │
│  │       ASR       │   │       LLM       │   │        TTS        │  │
│  │    Parakeet     │──►│     Qwen3-4B    │──►│   Chatterbox      │  │
│  │   FP16 ONNX     │   │   FP16 (HF)     │   │     FP16          │  │
│  │                 │   │  (Transformers) │   │   (Streaming)     │  │
│  └─────────────────┘   └─────────────────┘   └───────────────────┘  │
│                                                                      │
│  Pipeline: Audio → Transcription → Sentence Stream → Audio Stream   │
└─────────────────────────────────────────────────────────────────────┘
```

## Models Used

| Component | Model | Precision | VRAM |
|-----------|-------|-----------|------|
| **ASR** | [smcleod/parakeet-tdt-0.6b-v2](https://huggingface.co/smcleod/parakeet-tdt-0.6b-v2) | FP16 ONNX | ~1.2GB |
| **LLM** | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | FP16 | ~8GB |
| **TTS** | [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) | FP16 | ~3GB |

**Total VRAM: ~12-13GB** (fits on g5.xlarge with 24GB A10G)

## Features

- **FP16 Precision**: Maximum quality with native HuggingFace models
- **Streaming Pipeline**: Sentences stream from LLM to TTS, audio streams to client
- **Low Latency**: First audio chunk typically in <1s after speech ends
- **Interrupt Support**: User can interrupt the assistant mid-response
- **Voice Cloning**: Clone any voice with a 10s reference clip
- **Concurrent Processing**: All stages run in parallel with async queues
- **Real-time Metrics**: Latency tracking sent to client

## Quick Start

### 1. Launch EC2 Instance

**Recommended**: `g5.xlarge` (24GB A10G GPU) with Ubuntu 22.04 Deep Learning AMI

```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@<EC2_IP>
```

### 2. Setup Server

```bash
# Clone or upload this directory
git clone <your-repo> voice-pipeline
cd voice-pipeline/deploy

# Run setup (installs dependencies, downloads models)
chmod +x setup.sh
./setup.sh

# Start server
source venv/bin/activate
python server.py --port 8765
```

### 3. Open Firewall

```bash
# In AWS Console or CLI, open port 8765 for your IP
aws ec2 authorize-security-group-ingress \
    --group-id <sg-xxx> \
    --protocol tcp \
    --port 8765 \
    --cidr <YOUR_IP>/32
```

### 4. Connect from Local Machine

```bash
# Install client dependencies
pip install sounddevice websockets numpy

# List audio devices
python client.py --list-devices

# Connect to server
python client.py --server ws://<EC2_IP>:8765/ws
```

## Usage

### Server Options

```bash
python server.py \
    --host 0.0.0.0 \
    --port 8765 \
    --system-prompt "You are a helpful assistant..." \
    --voice-reference /path/to/reference.wav \
    --llm-model-id Qwen/Qwen3-4B-Instruct-2507 \
    --llm-dtype float16 \
    --llm-temperature 0.7 \
    --asr-model smcleod/parakeet-tdt-0.6b-v2 \
    --asr-quantization fp16 \
    --tts-exaggeration 0.5
```

### Client Options

```bash
python client.py \
    --server ws://1.2.3.4:8765/ws \
    --input-device 0 \
    --output-device 1 \
    --no-metrics
```

## WebSocket Protocol

### Client → Server

```json
{"type": "audio_in", "data": {"audio": "<base64 PCM s16le>", "sample_rate": 16000}}
{"type": "text_in", "data": {"text": "Hello!"}}
{"type": "interrupt", "data": {}}
{"type": "reset", "data": {}}
{"type": "set_system_prompt", "data": {"prompt": "..."}}
{"type": "set_voice", "data": {"path": "/path/to/voice.wav"}}
```

### Server → Client

```json
{"type": "ready", "data": {"asr_sample_rate": 16000, "tts_sample_rate": 24000}}
{"type": "transcription", "data": {"text": "Hello", "latency_ms": 150}}
{"type": "llm_sentence", "data": {"text": "Hi there!", "latency_ms": 200}}
{"type": "audio_out", "data": {"audio": "<base64 PCM>", "sample_rate": 24000}}
{"type": "audio_end", "data": {}}
{"type": "state", "data": {"state": "LISTENING|PROCESSING|SPEAKING"}}
{"type": "metrics", "data": {"llm_first_sentence": 0.2, "tts_first_chunk": 0.1, ...}}
{"type": "interrupted", "data": {}}
```

## Quick Deploy

For rapid iteration:

```bash
# From your local machine
./deploy.sh ubuntu@<EC2_IP> ~/.ssh/your-key.pem
```

This syncs code and optionally restarts the server.

## GPU Memory Usage

| Instance | GPU | VRAM | Fits? |
|----------|-----|------|-------|
| g4dn.xlarge | T4 | 16GB | Tight (may OOM) |
| **g5.xlarge** | A10G | 24GB | **Recommended** |
| g5.2xlarge | A10G | 24GB | Recommended + headroom |
| p4d.24xlarge | A100 | 40GB | Overkill but fast |

### Memory Breakdown

- ASR (Parakeet FP16): ~1.2GB
- LLM (Qwen3 4B FP16): ~8GB
- TTS (Chatterbox FP16): ~3GB
- **Total**: ~12-13GB

## Troubleshooting

### "CUDA out of memory"

- Use g5.xlarge (24GB) instead of g4dn.xlarge (16GB)
- Or reduce LLM to bfloat16 (slightly smaller)

### High latency

- Check network latency: `ping <EC2_IP>`
- Monitor GPU utilization: `nvidia-smi -l 1`
- Reduce ASR silence threshold for faster detection

### No audio output

- Check client audio devices: `python client.py --list-devices`
- Ensure TTS model loaded: check server logs
- Verify WebSocket connection: check client logs

## Files

```
deploy/
├── asr.py          # Parakeet ONNX FP16 speech recognition
├── llm.py          # Qwen3 4B FP16 via Transformers
├── tts.py          # Chatterbox FP16 streaming synthesis
├── pipeline.py     # Streaming pipeline orchestration
├── server.py       # WebSocket server
├── client.py       # Audio client
├── requirements.txt
├── setup.sh        # EC2 setup script
├── deploy.sh       # Quick deploy script
└── README.md
```

## License

MIT
