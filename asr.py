"""
ASR Module - Parakeet TDT 0.6B V2 with TensorRT
High-performance speech recognition using NVIDIA's Parakeet model
Optimized for A100/A10G/L4 GPUs with TensorRT acceleration
"""

import asyncio
import logging
import time
import os
from dataclasses import dataclass
from typing import Optional, Callable, Any
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    """Configuration for ASR module"""
    model_path: str = "/workspace/models/parakeet-tdt-0.6b-v2"
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


class ParakeetTRTASR:
    """
    Parakeet ASR using TensorRT engines
    Uses encoder.trt and decoder_joint.trt for inference
    """

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self._encoder = None
        self._decoder = None
        self._preprocessor = None
        self._tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=2)

    def load(self):
        """Load TensorRT engines and preprocessor"""
        if self._encoder is not None:
            return

        from trt_runtime import TRTEngine

        encoder_path = os.path.join(self.config.model_path, "trt", "encoder.trt")
        decoder_path = os.path.join(self.config.model_path, "trt", "decoder_joint.trt")

        logger.info(f"Loading Parakeet TRT encoder: {encoder_path}")
        self._encoder = TRTEngine(encoder_path)
        self._encoder.load()

        # Log expected input shapes for debugging
        for name in self._encoder.input_names:
            binding = self._encoder.bindings[name]
            logger.info(f"  Encoder input '{name}': shape={binding.shape}, dtype={binding.dtype}")

        logger.info(f"Loading Parakeet TRT decoder: {decoder_path}")
        self._decoder = TRTEngine(decoder_path)
        self._decoder.load()

        # Log decoder input shapes
        for name in self._decoder.input_names:
            binding = self._decoder.bindings[name]
            logger.info(f"  Decoder input '{name}': shape={binding.shape}, dtype={binding.dtype}")

        # Detect expected mel bins from encoder input shape
        audio_binding = self._encoder.bindings.get("audio_signal")
        if audio_binding:
            expected_shape = audio_binding.shape
            if len(expected_shape) >= 2:
                self._expected_mel_bins = expected_shape[1] if expected_shape[1] > 0 else 80
            else:
                self._expected_mel_bins = 80
        else:
            self._expected_mel_bins = 80
        logger.info(f"Expected mel bins: {self._expected_mel_bins}")

        # Load preprocessor for mel spectrogram extraction
        self._load_preprocessor()

        # Load tokenizer/vocabulary
        self._load_tokenizer()

        # Warmup
        self._warmup()
        logger.info("Parakeet TRT ASR loaded")

    def _load_preprocessor(self):
        """Load audio preprocessor for mel spectrogram extraction"""
        # Use detected mel bins or default to 80
        n_mels = getattr(self, '_expected_mel_bins', 80)

        # Check for static time dimension issue
        audio_binding = self._encoder.bindings.get("audio_signal")
        if audio_binding and len(audio_binding.shape) >= 3:
            time_dim = audio_binding.shape[2]
            if time_dim > 0 and time_dim == 1:
                logger.warning(
                    f"ASR encoder has static time dimension of 1. "
                    f"This suggests the engine was built incorrectly or for streaming mode. "
                    f"Please rebuild with dynamic shapes: [1, {n_mels}, -1]"
                )

        try:
            import torch
            import torchaudio

            # Use detected mel bins
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=512,
                win_length=int(0.025 * self.config.sample_rate),  # 25ms
                hop_length=int(0.01 * self.config.sample_rate),   # 10ms
                n_mels=n_mels,
                f_min=0,
                f_max=8000,
            )
            self._preprocessor = "torchaudio"
            self._n_mels = n_mels
            logger.info(f"Using torchaudio preprocessor with {n_mels} mel bins")
        except ImportError:
            # Fallback to librosa
            try:
                import librosa
                self._preprocessor = "librosa"
                self._n_mels = n_mels
                logger.info(f"Using librosa preprocessor with {n_mels} mel bins")
            except ImportError:
                raise RuntimeError("Neither torchaudio nor librosa available for preprocessing")

    def _load_tokenizer(self):
        """Load vocabulary for decoding"""
        # Parakeet TDT uses SentencePiece tokenizer
        # Try to load from model directory
        vocab_path = os.path.join(self.config.model_path, "tokenizer.model")
        if os.path.exists(vocab_path):
            try:
                import sentencepiece as spm
                self._tokenizer = spm.SentencePieceProcessor()
                self._tokenizer.Load(vocab_path)
                logger.info("Loaded SentencePiece tokenizer")
                return
            except Exception as e:
                logger.warning(f"Failed to load SentencePiece: {e}")

        # Fallback: try to download from HuggingFace
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "nvidia/parakeet-tdt-0.6b-v2",
                trust_remote_code=True
            )
            logger.info("Loaded HuggingFace tokenizer")
        except Exception as e:
            logger.warning(f"Failed to load HF tokenizer: {e}")
            # Use simple character-based decoding as last resort
            self._tokenizer = None

    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to mel spectrogram features"""
        n_mels = getattr(self, '_n_mels', 80)

        if self._preprocessor == "torchaudio":
            import torch
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            mel = self._mel_transform(audio_tensor)
            # Log mel
            mel = torch.log(mel.clamp(min=1e-10))
            # Normalize
            mel = (mel - mel.mean()) / (mel.std() + 1e-10)
            return mel.numpy()
        else:
            # Librosa fallback
            import librosa
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config.sample_rate,
                n_fft=512,
                hop_length=int(0.01 * self.config.sample_rate),
                win_length=int(0.025 * self.config.sample_rate),
                n_mels=n_mels,
                fmin=0,
                fmax=8000,
            )
            mel = np.log(np.maximum(mel, 1e-10))
            mel = (mel - mel.mean()) / (mel.std() + 1e-10)
            return mel[np.newaxis, ...]

    def _decode_tokens(self, token_ids: np.ndarray) -> str:
        """Decode token IDs to text"""
        # Remove padding and special tokens
        token_ids = token_ids.flatten()
        token_ids = token_ids[token_ids > 0]  # Remove padding

        if self._tokenizer is not None:
            if hasattr(self._tokenizer, 'decode'):
                return self._tokenizer.decode(token_ids.tolist())
            elif hasattr(self._tokenizer, 'DecodeIds'):
                return self._tokenizer.DecodeIds(token_ids.tolist())

        # Fallback: return token IDs as string (for debugging)
        return " ".join(map(str, token_ids.tolist()))

    def _warmup(self):
        """Warmup the engines"""
        try:
            dummy_audio = np.zeros(self.config.sample_rate, dtype=np.float32)
            _ = self._transcribe_sync(dummy_audio)
            logger.info("ASR warmup complete")
        except Exception as e:
            logger.warning(f"ASR warmup failed: {e}")
            logger.warning("ASR may work on first actual inference or needs engine rebuild")

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription using TRT engines"""
        if self._encoder is None:
            self.load()

        try:
            # Preprocess to mel spectrogram
            mel = self._preprocess_audio(audio)

            # Encoder inference
            # Input shape: [batch, mel_bins, time]
            encoder_out = self._encoder.infer({"audio_signal": mel})
            encoded = encoder_out.get("encoded", list(encoder_out.values())[0])
            encoded_len = np.array([[encoded.shape[-1]]], dtype=np.int64)

            # Decoder inference (greedy decoding)
            # TDT decoder takes encoded features and outputs token logits
            decoder_out = self._decoder.infer({
                "encoder_output": encoded,
                "encoder_output_length": encoded_len
            })

            # Get predicted tokens
            logits = decoder_out.get("logits", list(decoder_out.values())[0])
            token_ids = np.argmax(logits, axis=-1)

            # Decode to text
            text = self._decode_tokens(token_ids)
            return text.strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    async def transcribe(self, audio: np.ndarray) -> str:
        """Async transcription"""
        if self._encoder is None:
            self.load()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._transcribe_sync,
            audio
        )

    def shutdown(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=False)
        if self._encoder:
            self._encoder.shutdown()
        if self._decoder:
            self._decoder.shutdown()


# Alias for compatibility
ParakeetASR = ParakeetTRTASR


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
        self._asr = ParakeetTRTASR(self.config)
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
