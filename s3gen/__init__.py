"""
S3Gen Vocoder - Standalone Implementation
Extracted from Chatterbox TTS (Apache 2.0 License)
Converts speech tokens to audio waveforms
"""

from .s3gen import S3Token2Wav
from .hifigan import HiFTGenerator

__all__ = ['S3Token2Wav', 'HiFTGenerator']
