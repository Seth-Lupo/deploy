"""
TTS Module - Chatterbox TURBO with Streaming
Real-time audio generation using Turbo's 2-step CFM with chunked output
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Callable, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS module"""
    device: str = "cuda"
    sample_rate: int = 24000
    # Voice reference for cloning
    voice_reference: Optional[str] = None
    # Generation settings
    exaggeration: float = 0.0  # Turbo ignores this
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    # Streaming settings
    chunk_size: int = 25  # Speech tokens per chunk (balance latency vs quality)
    context_window: int = 15  # Context for continuity (smaller = less overlap)


class StreamingChatterboxTurboTTS:
    """
    Chatterbox TURBO TTS with TRUE STREAMING
    - Uses Turbo's 2-step CFM for 5x faster inference
    - Streams audio chunks as they're generated
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._stop_event = Event()

    def load(self):
        """Load Chatterbox Turbo model with streaming support"""
        if self._model is not None:
            return

        logger.info("Loading Chatterbox TURBO TTS with streaming...")

        from tts_turbo_streaming import ChatterboxTurboStreamingTTS
        self._model = ChatterboxTurboStreamingTTS.from_pretrained(device=self.config.device)
        logger.info("Loaded Chatterbox TURBO TTS with streaming")

        # Warmup
        self._warmup()
        logger.info("Chatterbox TURBO TTS ready")

    def _warmup(self):
        """Warmup the model with multiple generations to compile all code paths"""
        warmup_texts = [
            "Hello there.",
            "This is a warmup test for the text to speech system.",
            "One more sentence to fully compile.",
        ]
        try:
            for i, text in enumerate(warmup_texts):
                for chunk, metrics in self._model.generate_stream(
                    text,
                    chunk_size=self.config.chunk_size,
                    print_metrics=False
                ):
                    pass  # Just run through to warmup
                logger.info(f"TTS warmup {i+1}/{len(warmup_texts)} complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    @property
    def sample_rate(self) -> int:
        """Get sample rate from model"""
        if self._model is not None:
            return getattr(self._model, 'sr', self.config.sample_rate)
        return self.config.sample_rate

    def stop(self):
        """Stop current generation"""
        self._stop_event.set()

    def _generate_stream_sync(self, text: str):
        """Synchronous streaming generator"""
        kwargs = {}
        if self.config.voice_reference:
            kwargs['audio_prompt_path'] = self.config.voice_reference

        for audio_chunk, metrics in self._model.generate_stream(
            text,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            chunk_size=self.config.chunk_size,
            context_window=self.config.context_window,
            print_metrics=False,
            **kwargs
        ):
            if self._stop_event.is_set():
                break
            yield audio_chunk, metrics

    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncIterator[Tuple[np.ndarray, dict]]:
        """
        Async streaming synthesis
        Yields (audio_chunk, metrics) tuples as they're generated
        """
        if self._model is None:
            self.load()

        self._stop_event.clear()

        # Run the sync generator in executor and yield chunks
        import queue
        import threading

        chunk_queue = queue.Queue()
        done_event = threading.Event()

        def producer():
            try:
                for audio_chunk, metrics in self._generate_stream_sync(text):
                    chunk_queue.put((audio_chunk, metrics))
                    if self._stop_event.is_set():
                        break
            except Exception as e:
                logger.error(f"Stream producer error: {e}")
            finally:
                done_event.set()

        # Start producer thread
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        # Yield chunks as they arrive
        while not done_event.is_set() or not chunk_queue.empty():
            try:
                audio_chunk, metrics = chunk_queue.get(timeout=0.05)

                # Convert to numpy
                if hasattr(audio_chunk, 'numpy'):
                    audio_np = audio_chunk.squeeze().cpu().numpy()
                elif hasattr(audio_chunk, 'cpu'):
                    audio_np = audio_chunk.squeeze().cpu().numpy()
                else:
                    audio_np = np.array(audio_chunk)

                yield (audio_np.astype(np.float32), {"rtf": getattr(metrics, 'rtf', 0.0)})
            except queue.Empty:
                await asyncio.sleep(0.01)

        thread.join(timeout=1.0)

    async def synthesize(self, text: str) -> np.ndarray:
        """Full synthesis (collects all chunks)"""
        chunks = []
        async for audio_chunk, _ in self.synthesize_stream(text):
            chunks.append(audio_chunk)

        if chunks:
            return np.concatenate(chunks)
        return np.array([], dtype=np.float32)

    def shutdown(self):
        """Cleanup"""
        self._stop_event.set()
        self._executor.shutdown(wait=False)


class StreamingTTS:
    """
    High-level streaming TTS wrapper
    Provides async interface with callbacks
    Uses Chatterbox TURBO for fast inference
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._tts = StreamingChatterboxTurboTTS(self.config)
        self._on_audio: Optional[Callable[[np.ndarray, dict], Any]] = None
        self._on_complete: Optional[Callable[[], Any]] = None

    def set_callbacks(
        self,
        on_audio: Optional[Callable[[np.ndarray, dict], Any]] = None,
        on_complete: Optional[Callable[[], Any]] = None
    ):
        """Set callbacks"""
        self._on_audio = on_audio
        self._on_complete = on_complete

    def load(self):
        """Load model"""
        self._tts.load()

    @property
    def sample_rate(self) -> int:
        """Get sample rate"""
        return self._tts.sample_rate

    def set_voice(self, reference_path: str):
        """Set voice reference for cloning"""
        self.config.voice_reference = reference_path
        self._tts.config.voice_reference = reference_path

    def stop(self):
        """Stop generation"""
        self._tts.stop()

    async def speak(self, text: str) -> AsyncIterator[np.ndarray]:
        """
        Synthesize text to audio, streaming chunks
        """
        async for audio_chunk, metrics in self._tts.synthesize_stream(text):
            if self._on_audio:
                self._on_audio(audio_chunk, metrics)
            yield audio_chunk

        if self._on_complete:
            self._on_complete()

    async def speak_full(self, text: str) -> np.ndarray:
        """Synthesize complete audio"""
        return await self._tts.synthesize(text)

    def shutdown(self):
        """Cleanup"""
        self._tts.shutdown()
