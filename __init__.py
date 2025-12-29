"""
Voice Pipeline - Streaming ASR → LLM → TTS

A high-performance voice conversation pipeline designed for EC2 GPU instances.

Components:
- ASR: Parakeet TDT 0.6B V2 Int8 (ONNX)
- LLM: Qwen3 4B Instruct (GGUF via llama.cpp)
- TTS: Chatterbox Turbo with streaming

Designed for EC2 GPU instances (g4dn.xlarge, g5.xlarge, etc.)
"""

from .asr import StreamingASR, ASRConfig, ParakeetASR, StreamingASRBuffer
from .llm import StreamingLLM, LLMConfig, QwenLLM, SentenceBuffer
from .tts import StreamingTTS, TTSConfig, ChatterboxTTS, AudioPlayer
from .pipeline import VoicePipeline, PipelineConfig, PipelineState, PipelineMetrics

__version__ = "1.0.0"
__all__ = [
    "StreamingASR",
    "ASRConfig",
    "ParakeetASR",
    "StreamingASRBuffer",
    "StreamingLLM",
    "LLMConfig",
    "QwenLLM",
    "SentenceBuffer",
    "StreamingTTS",
    "TTSConfig",
    "ChatterboxTTS",
    "AudioPlayer",
    "VoicePipeline",
    "PipelineConfig",
    "PipelineState",
    "PipelineMetrics",
]
