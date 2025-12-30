"""
ASR Module - Parakeet TDT 0.6B V2 (FP16/FP32)
High-performance speech recognition using NVIDIA's Parakeet model
Optimized for A100/A10G GPUs
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    """Configuration for ASR module"""
    model: str = "istupakov/parakeet-tdt-0.6b-v2-onnx"  # Public ONNX version
    quantization: str = "fp16"  # fp16, fp32, or int8
    sample_rate: int = 16000
    # VAD settings
    use_vad: bool = True
    vad_threshold: float = 0.5
    # Silence detection
    silence_threshold: float = 0.01
    min_speech_ms: int = 200
    max_silence_ms: int = 400
    # Processing
    max_audio_seconds: float = 30.0
    chunk_size_ms: int = 100


class VADProcessor:
    """
    Voice Activity Detection using energy-based detection
    Falls back to simple energy detection if Silero VAD unavailable
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model = None
        self._use_silero = False

    def load(self):
        """Try to load Silero VAD, fall back to energy-based"""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self._model = model
            self._use_silero = True
            logger.info("Silero VAD loaded")
        except Exception as e:
            logger.warning(f"Silero VAD not available, using energy-based: {e}")
            self._use_silero = False

    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio chunk contains speech"""
        if len(audio) == 0:
            return False

        if self._use_silero and self._model is not None:
            # Silero VAD requires exactly 512 samples for 16kHz
            chunk_size = 512 if self.sample_rate == 16000 else 256
            if len(audio) >= chunk_size:
                import torch
                # Use last chunk_size samples
                audio_chunk = audio[-chunk_size:]
                audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
                try:
                    confidence = self._model(audio_tensor, self.sample_rate).item()
                    return confidence > self.threshold
                except Exception:
                    pass  # Fall back to energy-based

        # Energy-based fallback
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > self.threshold * 0.1

    def get_speech_probability(self, audio: np.ndarray) -> float:
        """Get speech probability for audio chunk"""
        if len(audio) == 0:
            return 0.0

        if self._use_silero and self._model is not None:
            # Silero VAD requires exactly 512 samples for 16kHz
            chunk_size = 512 if self.sample_rate == 16000 else 256
            if len(audio) >= chunk_size:
                import torch
                audio_chunk = audio[-chunk_size:]
                audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
                try:
                    return self._model(audio_tensor, self.sample_rate).item()
                except Exception:
                    pass  # Fall back to energy-based

        # Energy-based fallback
        energy = np.sqrt(np.mean(audio ** 2))
        return min(energy / 0.1, 1.0)


class ParakeetASR:
    """
    Parakeet ASR using ONNX runtime
    FP16/FP32 inference optimized for A100/A10G
    """

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2)

    def load(self):
        """Load Parakeet model"""
        if self._model is not None:
            return

        import onnx_asr

        logger.info(f"Loading Parakeet ASR: {self.config.model}")
        logger.info(f"Quantization: {self.config.quantization}")

        # Force CUDA execution provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        logger.info(f"Using ONNX providers: {providers}")

        # Load with specified quantization
        # For FP16/FP32, we use the non-int8 model
        if self.config.quantization == "int8":
            self._model = onnx_asr.load_model(
                self.config.model,
                quantization="int8",
                providers=providers
            )
        else:
            # FP16/FP32 - load without quantization
            self._model = onnx_asr.load_model(
                self.config.model,
                quantization=None,  # Uses default FP16/FP32
                providers=providers
            )

        # Warmup
        self._warmup()
        logger.info("Parakeet ASR loaded")

    def _warmup(self):
        """Warmup the model"""
        dummy_audio = np.zeros(self.config.sample_rate, dtype=np.float32)
        _ = self._transcribe_sync(dummy_audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription"""
        if self._model is None:
            self.load()

        try:
            result = self._model.recognize(audio)
            return result if isinstance(result, str) else result.get('text', '')
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    async def transcribe(self, audio: np.ndarray) -> str:
        """Async transcription"""
        if self._model is None:
            self.load()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._transcribe_sync,
            audio
        )

    def transcribe_with_timestamps(self, audio: np.ndarray) -> dict:
        """Transcribe with word-level timestamps"""
        if self._model is None:
            self.load()

        try:
            model_with_ts = self._model.with_timestamps()
            result = model_with_ts.recognize(audio)
            return result
        except Exception as e:
            logger.error(f"Timestamp transcription error: {e}")
            return {"text": "", "words": []}

    def shutdown(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)


class StreamingASRBuffer:
    """
    Buffer for streaming ASR with endpoint detection
    Accumulates audio and detects speech boundaries
    """

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self._buffer: deque = deque()
        self._total_samples = 0
        self._speech_start: Optional[float] = None
        self._last_speech: Optional[float] = None
        self._vad = VADProcessor(
            sample_rate=self.config.sample_rate,
            threshold=self.config.vad_threshold
        )

    def load(self):
        """Load VAD model"""
        self._vad.load()

    def add_audio(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Add audio chunk and return complete utterance if endpoint detected
        Returns None if more audio is needed
        """
        if len(audio) == 0:
            return None

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        now = time.time()
        is_speech = self._vad.is_speech(audio)

        if is_speech:
            if self._speech_start is None:
                self._speech_start = now
            self._last_speech = now

        self._buffer.append(audio)
        self._total_samples += len(audio)

        # Check for endpoint
        complete_audio = self._check_endpoint(now)
        return complete_audio

    def _check_endpoint(self, now: float) -> Optional[np.ndarray]:
        """Check if utterance is complete"""
        # No speech yet
        if self._speech_start is None:
            # Prevent buffer overflow with silence
            max_samples = int(self.config.sample_rate * 2)  # 2 seconds of silence max
            while self._total_samples > max_samples and len(self._buffer) > 1:
                removed = self._buffer.popleft()
                self._total_samples -= len(removed)
            return None

        # Check for max duration
        speech_duration = now - self._speech_start
        if speech_duration > self.config.max_audio_seconds:
            return self._extract_utterance()

        # Check for silence after speech
        if self._last_speech is not None:
            silence_duration = (now - self._last_speech) * 1000  # ms
            speech_duration_ms = (now - self._speech_start) * 1000

            if (silence_duration > self.config.max_silence_ms and
                    speech_duration_ms > self.config.min_speech_ms):
                return self._extract_utterance()

        return None

    def _extract_utterance(self) -> np.ndarray:
        """Extract complete utterance from buffer"""
        audio = np.concatenate(list(self._buffer))
        self.reset()
        return audio

    def reset(self):
        """Reset buffer state"""
        self._buffer.clear()
        self._total_samples = 0
        self._speech_start = None
        self._last_speech = None

    def get_buffered_audio(self) -> Optional[np.ndarray]:
        """Get all buffered audio without resetting"""
        if not self._buffer:
            return None
        return np.concatenate(list(self._buffer))

    def force_endpoint(self) -> Optional[np.ndarray]:
        """Force extraction of current buffer"""
        if self._speech_start is not None and self._buffer:
            return self._extract_utterance()
        return None


class StreamingASR:
    """
    Complete streaming ASR pipeline
    Combines buffering, VAD, and transcription
    """

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self._asr = ParakeetASR(self.config)
        self._buffer = StreamingASRBuffer(self.config)
        self._on_transcription: Optional[Callable[[str, float], Any]] = None

    def set_callback(self, on_transcription: Callable[[str, float], Any]):
        """Set callback for transcription results"""
        self._on_transcription = on_transcription

    def load(self):
        """Load models"""
        self._asr.load()
        self._buffer.load()

    async def process_audio(self, audio: np.ndarray) -> Optional[str]:
        """
        Process audio chunk
        Returns transcription if utterance complete, None otherwise
        """
        complete_audio = self._buffer.add_audio(audio)

        if complete_audio is not None:
            start_time = time.time()
            transcription = await self._asr.transcribe(complete_audio)
            latency = time.time() - start_time

            if transcription and self._on_transcription:
                self._on_transcription(transcription, latency)

            return transcription

        return None

    async def flush(self) -> Optional[str]:
        """Force process any remaining audio"""
        remaining = self._buffer.force_endpoint()
        if remaining is not None:
            start_time = time.time()
            transcription = await self._asr.transcribe(remaining)
            latency = time.time() - start_time

            if transcription and self._on_transcription:
                self._on_transcription(transcription, latency)

            return transcription

        return None

    def reset(self):
        """Reset the buffer"""
        self._buffer.reset()

    def shutdown(self):
        """Cleanup"""
        self._asr.shutdown()
