"""
Voice Pipeline - Streaming ASR → LLM → TTS
Concurrent, efficient pipeline with proper data structures for streaming
Optimized for A100/A10G GPUs with FP16 models
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict
from enum import Enum, auto
import numpy as np

from asr import StreamingASR, ASRConfig
from llm import StreamingLLM, LLMConfig
from tts import StreamingTTS, TTSConfig

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline state machine"""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()


@dataclass
class PipelineConfig:
    """Configuration for the voice pipeline"""
    # ASR settings
    asr_model: str = "istupakov/parakeet-tdt-0.6b-v2-onnx"
    asr_quantization: str = "fp16"
    asr_sample_rate: int = 16000
    asr_vad_threshold: float = 0.5
    asr_silence_ms: int = 400

    # LLM settings
    llm_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_dtype: str = "float16"
    llm_max_tokens: int = 150
    llm_temperature: float = 0.6

    # TTS settings
    tts_device: str = "cuda"
    tts_voice_reference: Optional[str] = None
    tts_exaggeration: float = 0.5

    # System prompt
    system_prompt: str = """You are a fast, helpful voice assistant. Keep responses extremely brief - one sentence when possible. Never use emojis, asterisks, or special formatting. Speak naturally and directly. Get to the point immediately."""

    # Pipeline settings
    enable_interrupts: bool = True
    audio_queue_size: int = 20


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance"""
    asr_latency: float = 0.0
    llm_first_token: float = 0.0
    llm_first_sentence: float = 0.0
    tts_first_chunk: float = 0.0
    total_latency: float = 0.0
    sentences_generated: int = 0
    audio_chunks_sent: int = 0


class VoicePipeline:
    """
    Complete voice pipeline with streaming between models

    Flow:
    Audio In → ASR (utterance detection) → LLM (sentence streaming) → TTS (audio streaming) → Audio Out

    Uses asyncio queues and events for concurrent, efficient streaming
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._state = PipelineState.IDLE
        self._metrics = PipelineMetrics()

        # Components
        self._asr: Optional[StreamingASR] = None
        self._llm: Optional[StreamingLLM] = None
        self._tts: Optional[StreamingTTS] = None

        # Async primitives
        self._sentence_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.audio_queue_size)
        self._interrupt_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()

        # Worker tasks
        self._tts_worker_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_transcription: Optional[Callable[[str, float], Any]] = None
        self._on_llm_sentence: Optional[Callable[[str, float], Any]] = None
        self._on_audio_chunk: Optional[Callable[[np.ndarray, dict], Any]] = None
        self._on_state_change: Optional[Callable[[PipelineState], Any]] = None
        self._on_metrics: Optional[Callable[[PipelineMetrics], Any]] = None

    def set_callbacks(
        self,
        on_transcription: Optional[Callable[[str, float], Any]] = None,
        on_llm_sentence: Optional[Callable[[str, float], Any]] = None,
        on_audio_chunk: Optional[Callable[[np.ndarray, dict], Any]] = None,
        on_state_change: Optional[Callable[[PipelineState], Any]] = None,
        on_metrics: Optional[Callable[[PipelineMetrics], Any]] = None,
    ):
        """Set callbacks for pipeline events"""
        self._on_transcription = on_transcription
        self._on_llm_sentence = on_llm_sentence
        self._on_audio_chunk = on_audio_chunk
        self._on_state_change = on_state_change
        self._on_metrics = on_metrics

    def _set_state(self, state: PipelineState):
        """Update pipeline state"""
        if self._state != state:
            self._state = state
            logger.debug(f"Pipeline state: {state.name}")
            if self._on_state_change:
                self._on_state_change(state)

    async def initialize(self):
        """Initialize all pipeline components"""
        logger.info("Initializing voice pipeline...")

        # Initialize ASR
        asr_config = ASRConfig(
            model=self.config.asr_model,
            quantization=self.config.asr_quantization,
            sample_rate=self.config.asr_sample_rate,
            vad_threshold=self.config.asr_vad_threshold,
            max_silence_ms=self.config.asr_silence_ms,
        )
        self._asr = StreamingASR(asr_config)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._asr.load)
        logger.info("ASR initialized")

        # Initialize LLM
        llm_config = LLMConfig(
            model_id=self.config.llm_model_id,
            dtype=self.config.llm_dtype,
            max_new_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            system_prompt=self.config.system_prompt,
        )
        self._llm = StreamingLLM(llm_config)
        await loop.run_in_executor(None, self._llm.load)
        logger.info("LLM initialized")

        # Initialize TTS
        tts_config = TTSConfig(
            device=self.config.tts_device,
            voice_reference=self.config.tts_voice_reference,
            exaggeration=self.config.tts_exaggeration,
        )
        self._tts = StreamingTTS(tts_config)
        await loop.run_in_executor(None, self._tts.load)
        logger.info("TTS initialized")

        # Start TTS worker
        self._tts_worker_task = asyncio.create_task(self._tts_worker())

        self._set_state(PipelineState.IDLE)
        logger.info("Voice pipeline ready")

    async def _tts_worker(self):
        """
        Background worker that processes sentences from queue and generates audio
        Runs continuously, streaming audio as sentences arrive
        """
        while not self._shutdown_event.is_set():
            try:
                # Wait for sentence with timeout
                try:
                    sentence = await asyncio.wait_for(
                        self._sentence_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                if sentence is None:
                    # End of response signal
                    await self._audio_queue.put(None)
                    continue

                if self._interrupt_event.is_set():
                    continue

                # Generate audio for this sentence
                tts_start = time.time()
                first_chunk = True
                logger.info(f"TTS worker: generating audio for '{sentence[:30]}...'")

                async for audio_chunk in self._tts.speak(sentence):
                    if self._interrupt_event.is_set():
                        break

                    if first_chunk:
                        self._metrics.tts_first_chunk = time.time() - tts_start
                        first_chunk = False

                    # Put audio in output queue
                    logger.info(f"TTS worker: got audio chunk shape={audio_chunk.shape}, putting in queue")
                    try:
                        self._audio_queue.put_nowait(audio_chunk)
                        self._metrics.audio_chunks_sent += 1
                        logger.info(f"TTS worker: audio chunk queued, queue size={self._audio_queue.qsize()}")
                    except asyncio.QueueFull:
                        # Drop if queue full
                        logger.warning("Audio queue full, dropping chunk")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TTS worker error: {e}")

    async def process_audio(self, audio: np.ndarray) -> Optional[str]:
        """
        Process incoming audio chunk
        Returns transcription if utterance complete
        """
        if self._state == PipelineState.SPEAKING and not self.config.enable_interrupts:
            return None

        # Check for interrupt during speech
        if self._state == PipelineState.SPEAKING and self.config.enable_interrupts:
            # Detect if user is speaking (potential interrupt)
            energy = np.sqrt(np.mean(audio ** 2))
            if energy > 0.02:  # Threshold for interrupt detection
                await self.interrupt()
                return None

        self._set_state(PipelineState.LISTENING)
        transcription = await self._asr.process_audio(audio)

        if transcription:
            self._metrics.asr_latency = time.time()  # Will be used for total latency calc
            if self._on_transcription:
                self._on_transcription(transcription, 0.0)
            return transcription

        return None

    async def respond(self, user_input: str):
        """
        Generate response to user input
        Streams sentences to TTS which streams audio chunks
        """
        self._set_state(PipelineState.PROCESSING)
        self._interrupt_event.clear()
        self._metrics = PipelineMetrics()

        response_start = time.time()
        first_sentence = True

        try:
            async for sentence in self._llm.respond(user_input):
                if self._interrupt_event.is_set():
                    break

                # Track metrics
                if first_sentence:
                    self._metrics.llm_first_sentence = time.time() - response_start
                    first_sentence = False

                self._metrics.sentences_generated += 1

                # Notify callback
                if self._on_llm_sentence:
                    self._on_llm_sentence(sentence, time.time() - response_start)

                # Queue for TTS (TTS will stream audio chunks)
                await self._sentence_queue.put(sentence)

                self._set_state(PipelineState.SPEAKING)

            # Signal end of response
            await self._sentence_queue.put(None)

            # Calculate total latency
            self._metrics.total_latency = time.time() - response_start

            if self._on_metrics:
                self._on_metrics(self._metrics)

        except asyncio.CancelledError:
            await self.interrupt()
            raise

    async def get_audio_chunk(self) -> Optional[np.ndarray]:
        """
        Get next audio chunk from output queue
        Returns None when response complete
        """
        try:
            chunk = await asyncio.wait_for(
                self._audio_queue.get(),
                timeout=0.1
            )
            if chunk is not None and self._on_audio_chunk:
                self._on_audio_chunk(chunk, {})
            return chunk
        except asyncio.TimeoutError:
            return np.array([], dtype=np.float32)

    async def interrupt(self):
        """Interrupt current response"""
        logger.info("Interrupt triggered")
        self._set_state(PipelineState.INTERRUPTED)
        self._interrupt_event.set()

        # Stop LLM generation
        if self._llm:
            self._llm.stop()

        # Stop TTS
        if self._tts:
            self._tts.stop()

        # Clear queues
        while not self._sentence_queue.empty():
            try:
                self._sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Reset ASR
        if self._asr:
            self._asr.reset()

        self._set_state(PipelineState.IDLE)

    async def flush_asr(self) -> Optional[str]:
        """Flush any remaining audio in ASR buffer"""
        if self._asr:
            return await self._asr.flush()
        return None

    def reset_conversation(self):
        """Reset conversation history"""
        if self._llm:
            self._llm.reset()

    def set_system_prompt(self, prompt: str):
        """Update system prompt"""
        self.config.system_prompt = prompt
        if self._llm:
            self._llm.set_system_prompt(prompt)

    def set_voice(self, reference_path: str):
        """Set voice reference for TTS cloning"""
        self.config.tts_voice_reference = reference_path
        if self._tts:
            self._tts.set_voice(reference_path)

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state"""
        return self._state

    @property
    def tts_sample_rate(self) -> int:
        """Get TTS sample rate"""
        return self._tts.sample_rate if self._tts else 24000

    @property
    def asr_sample_rate(self) -> int:
        """Get ASR sample rate"""
        return self.config.asr_sample_rate

    async def shutdown(self):
        """Shutdown pipeline"""
        logger.info("Shutting down pipeline...")
        self._shutdown_event.set()

        if self._tts_worker_task:
            self._tts_worker_task.cancel()
            try:
                await self._tts_worker_task
            except asyncio.CancelledError:
                pass

        if self._asr:
            self._asr.shutdown()
        if self._llm:
            self._llm.shutdown()
        if self._tts:
            self._tts.shutdown()

        logger.info("Pipeline shutdown complete")


class ConversationSession:
    """
    High-level conversation session
    Manages a complete voice conversation loop
    """

    def __init__(self, pipeline: VoicePipeline):
        self._pipeline = pipeline
        self._running = False
        self._response_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the conversation session"""
        await self._pipeline.initialize()
        self._running = True

    async def process_audio(self, audio: np.ndarray):
        """Process incoming audio, automatically triggering responses"""
        transcription = await self._pipeline.process_audio(audio)

        if transcription:
            # Start response generation
            if self._response_task and not self._response_task.done():
                await self._pipeline.interrupt()

            self._response_task = asyncio.create_task(
                self._pipeline.respond(transcription)
            )

    async def get_audio(self) -> Optional[np.ndarray]:
        """Get next audio chunk for playback"""
        return await self._pipeline.get_audio_chunk()

    async def interrupt(self):
        """Interrupt current response"""
        await self._pipeline.interrupt()

    async def stop(self):
        """Stop the session"""
        self._running = False
        await self._pipeline.shutdown()
