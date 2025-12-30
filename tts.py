"""
TTS Module - Chatterbox TURBO with TensorRT + PyTorch Vocoder
Uses TensorRT for T3 transformer components
Keeps S3Gen vocoder in PyTorch for audio synthesis
"""

import asyncio
import logging
import time
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Callable, Any, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np

logger = logging.getLogger(__name__)


class AttrDict(dict):
    """Dictionary with attribute access"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@dataclass
class TTSConfig:
    """Configuration for TTS module"""
    model_path: str = "/workspace/models/chatterbox-turbo"
    device: str = "cuda"
    sample_rate: int = 24000
    # Voice reference for cloning
    voice_reference: Optional[str] = None
    # Generation settings
    exaggeration: float = 0.0
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    # Streaming settings
    chunk_size: int = 25
    context_window: int = 15


class ChatterboxTRTTTS:
    """
    Chatterbox TURBO TTS with TensorRT acceleration
    - T3 components (transformer, embeddings, heads) use TensorRT
    - S3Gen vocoder uses PyTorch for audio synthesis
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.sr = self.config.sample_rate

        # TRT engines for T3
        self._speech_emb_engine = None
        self._text_emb_engine = None
        self._speech_head_engine = None
        self._transformer_engine = None

        # PyTorch components
        self._s3gen = None
        self._tokenizer = None
        self._conds = None

        self._executor = ThreadPoolExecutor(max_workers=2)
        self._stop_event = Event()

        # Token constants from Chatterbox
        self._start_speech_token = 6561
        self._stop_speech_token = 6562

    def load(self):
        """Load TRT engines and PyTorch vocoder"""
        if self._transformer_engine is not None:
            return

        import torch
        from trt_runtime import TRTEngine

        trt_path = os.path.join(self.config.model_path, "trt")
        pytorch_path = os.path.join(self.config.model_path, "pytorch")

        # Load TRT engines
        logger.info("Loading TensorRT T3 engines...")

        self._speech_emb_engine = TRTEngine(os.path.join(trt_path, "speech_emb.trt"))
        self._speech_emb_engine.load()
        logger.info("  speech_emb.trt loaded")

        self._text_emb_engine = TRTEngine(os.path.join(trt_path, "text_emb.trt"))
        self._text_emb_engine.load()
        logger.info("  text_emb.trt loaded")

        self._speech_head_engine = TRTEngine(os.path.join(trt_path, "speech_head.trt"))
        self._speech_head_engine.load()
        logger.info("  speech_head.trt loaded")

        self._transformer_engine = TRTEngine(os.path.join(trt_path, "transformer_fp16.trt"))
        self._transformer_engine.load()
        logger.info("  transformer_fp16.trt loaded")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Load S3Gen vocoder (PyTorch)
        logger.info("Loading S3Gen vocoder (PyTorch)...")
        self._load_s3gen(pytorch_path)

        # Warmup
        self._warmup()
        logger.info("Chatterbox TRT TTS ready")

    def _load_s3gen(self, pytorch_path: str):
        """Load S3Gen vocoder from safetensors"""
        import torch
        from safetensors.torch import load_file

        # Use standalone S3Gen implementation
        try:
            from s3gen import S3Token2Wav

            s3gen_weights = load_file(os.path.join(pytorch_path, "s3gen.safetensors"))
            self._s3gen = S3Token2Wav()
            self._s3gen.load_state_dict(s3gen_weights, strict=False)
            self._s3gen.to(self.config.device).eval()
            logger.info("Loaded standalone S3Gen vocoder")
            return
        except Exception as e:
            logger.warning(f"Failed to load standalone S3Gen: {e}")

        # Fallback: Try chatterbox package if available
        try:
            from chatterbox.models.s3gen import S3Gen

            s3gen_weights = load_file(os.path.join(pytorch_path, "s3gen.safetensors"))
            self._s3gen = S3Gen()
            self._s3gen.load_state_dict(s3gen_weights, strict=False)
            self._s3gen.to(self.config.device).eval()
            logger.info("Loaded S3Gen from chatterbox package")
        except Exception as e:
            logger.error(f"Could not load S3Gen vocoder: {e}")
            self._s3gen = None

    def prepare_conditionals(self, audio_path: str, exaggeration: float = 0.0):
        """Prepare conditioning from reference audio"""
        if self._s3gen is None:
            logger.warning("S3Gen not loaded, cannot prepare conditionals")
            return

        import torch
        import torchaudio

        try:
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            if sr != 24000:
                wav = torchaudio.functional.resample(wav, sr, 24000)
            wav = wav.mean(0) if wav.dim() > 1 else wav

            # Extract mel spectrogram for conditioning
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=24000,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
            ).to(self.config.device)

            mel = mel_transform(wav.to(self.config.device).unsqueeze(0))
            mel = torch.log(mel.clamp(min=1e-5))

            # Create conditioning dict
            self._conds = AttrDict({
                'gen': {
                    'mel': mel,
                    'embedding': torch.zeros(1, 192, device=self.config.device),  # Placeholder
                }
            })

            logger.info(f"Prepared conditionals from: {audio_path}")

        except Exception as e:
            logger.warning(f"Failed to prepare conditionals: {e}")
            self._conds = None

    def _warmup(self):
        """Warmup TRT engines"""
        # Warmup each engine
        for name, engine in [
            ("speech_emb", self._speech_emb_engine),
            ("text_emb", self._text_emb_engine),
            ("speech_head", self._speech_head_engine),
            ("transformer", self._transformer_engine),
        ]:
            if engine:
                engine.warmup()
                logger.info(f"  {name} warmed up")

    def _generate_speech_tokens(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = 1000,
        chunk_size: int = 25,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate speech tokens using TRT engines
        Yields chunks of speech tokens
        """
        import torch

        # Tokenize text
        text_tokens = self._tokenizer.encode(text, return_tensors="np")

        # Get text embeddings via TRT
        text_embeds = self._text_emb_engine.infer({"tokens": text_tokens})
        text_embeds = list(text_embeds.values())[0]

        # Initialize with start token
        speech_tokens = [self._start_speech_token]
        chunk_buffer = []

        # Get initial speech embedding
        speech_input = np.array([[self._start_speech_token]], dtype=np.int64)
        speech_embeds = self._speech_emb_engine.infer({"tokens": speech_input})
        speech_embeds = list(speech_embeds.values())[0]

        # Combine text and speech embeddings
        combined_embeds = np.concatenate([text_embeds, speech_embeds], axis=1)

        # Initial transformer pass
        hidden = self._transformer_engine.infer({"inputs_embeds": combined_embeds.astype(np.float32)})
        hidden = list(hidden.values())[0]

        # Get logits for next token
        last_hidden = hidden[:, -1:, :]
        logits = self._speech_head_engine.infer({"hidden": last_hidden.astype(np.float32)})
        logits = list(logits.values())[0]

        # Sample next token
        next_token = self._sample_token(logits[0, 0], temperature, top_k, top_p)
        speech_tokens.append(next_token)
        chunk_buffer.append(next_token)

        # Autoregressive generation
        for i in range(max_gen_len):
            if self._stop_event.is_set():
                break

            # Get embedding for current token
            current_input = np.array([[next_token]], dtype=np.int64)
            current_embed = self._speech_emb_engine.infer({"tokens": current_input})
            current_embed = list(current_embed.values())[0]

            # Update combined embeddings
            combined_embeds = np.concatenate([combined_embeds, current_embed], axis=1)

            # Transformer pass
            hidden = self._transformer_engine.infer({"inputs_embeds": combined_embeds.astype(np.float32)})
            hidden = list(hidden.values())[0]

            # Get next token logits
            last_hidden = hidden[:, -1:, :]
            logits = self._speech_head_engine.infer({"hidden": last_hidden.astype(np.float32)})
            logits = list(logits.values())[0]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in speech_tokens:
                    if token < logits.shape[-1]:
                        logits[0, 0, token] /= repetition_penalty

            # Sample
            next_token = self._sample_token(logits[0, 0], temperature, top_k, top_p)
            speech_tokens.append(next_token)
            chunk_buffer.append(next_token)

            # Check for EOS
            if next_token == self._stop_speech_token:
                if chunk_buffer:
                    yield np.array(chunk_buffer[:-1], dtype=np.int64)  # Exclude EOS
                break

            # Yield chunk
            if len(chunk_buffer) >= chunk_size:
                yield np.array(chunk_buffer, dtype=np.int64)
                chunk_buffer = []

        # Yield remaining
        if chunk_buffer:
            yield np.array(chunk_buffer, dtype=np.int64)

    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> int:
        """Sample next token from logits"""
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices = np.argsort(logits)[-top_k:]
            mask = np.ones_like(logits) * -np.inf
            mask[indices] = 0
            logits = logits + mask

        # Softmax
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()

        # Top-p filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_indices])
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            mask_indices = sorted_indices[cutoff_idx:]
            probs[mask_indices] = 0
            probs = probs / probs.sum()

        # Sample
        return int(np.random.choice(len(probs), p=probs))

    def _tokens_to_audio(self, tokens: np.ndarray) -> np.ndarray:
        """Convert speech tokens to audio using S3Gen (PyTorch)"""
        if self._s3gen is None:
            logger.error("S3Gen vocoder not loaded")
            return np.array([], dtype=np.float32)

        import torch

        # Filter out invalid tokens
        valid_tokens = tokens[tokens < 6561]
        if len(valid_tokens) == 0:
            return np.array([], dtype=np.float32)

        tokens_tensor = torch.from_numpy(valid_tokens).long().to(self.config.device)

        with torch.no_grad():
            # Get reference dict for conditioning
            ref_dict = None
            if self._conds is not None and hasattr(self._conds, 'gen'):
                ref_dict = self._conds.gen

            # S3Gen inference with 2-step CFM (Turbo mode)
            if hasattr(self._s3gen, 'inference'):
                wav, _ = self._s3gen.inference(
                    speech_tokens=tokens_tensor,
                    ref_dict=ref_dict,
                    n_cfm_timesteps=2,
                )
            elif hasattr(self._s3gen, 'forward'):
                # Direct forward pass
                wav = self._s3gen(tokens_tensor)
            else:
                logger.error("S3Gen has no inference method")
                return np.array([], dtype=np.float32)

            wav = wav.squeeze().cpu().numpy()

        return wav

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        chunk_size: int = 25,
        context_window: int = 15,
        print_metrics: bool = False,
    ) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """
        Stream audio chunks as they're generated
        Uses TRT for T3 + PyTorch for S3Gen

        Yields:
            Tuple of (audio_chunk, metrics_dict)
        """
        start_time = time.time()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

        self._stop_event.clear()

        all_tokens = []
        chunk_count = 0
        total_audio_duration = 0.0
        first_chunk_time = None

        for token_chunk in self._generate_speech_tokens(
            text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
        ):
            if self._stop_event.is_set():
                break

            # Get context from previous tokens
            if all_tokens and context_window > 0:
                context = np.concatenate(all_tokens)[-context_window:]
                tokens_with_context = np.concatenate([context, token_chunk])
            else:
                tokens_with_context = token_chunk
                context = np.array([], dtype=np.int64)

            # Convert to audio
            audio = self._tokens_to_audio(tokens_with_context)

            if len(audio) > 0:
                # Crop out context samples
                if len(context) > 0:
                    samples_per_token = len(audio) / len(tokens_with_context)
                    skip_samples = int(len(context) * samples_per_token)
                    audio = audio[skip_samples:]

                if len(audio) > 0:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                        if print_metrics:
                            logger.info(f"Time to first chunk: {first_chunk_time:.3f}s")

                    chunk_count += 1
                    audio_duration = len(audio) / self.sr
                    total_audio_duration += audio_duration

                    metrics = {
                        "chunk": chunk_count,
                        "latency_first": first_chunk_time,
                        "audio_duration": audio_duration,
                    }

                    yield audio.astype(np.float32), metrics

            all_tokens.append(token_chunk)

        # Final metrics
        total_time = time.time() - start_time
        if print_metrics and total_audio_duration > 0:
            rtf = total_time / total_audio_duration
            logger.info(f"Total time: {total_time:.3f}s, Audio: {total_audio_duration:.3f}s, RTF: {rtf:.3f}")

    def stop(self):
        """Stop current generation"""
        self._stop_event.set()

    def shutdown(self):
        """Cleanup resources"""
        self._stop_event.set()
        self._executor.shutdown(wait=False)

        # Shutdown TRT engines
        for engine in [
            self._speech_emb_engine,
            self._text_emb_engine,
            self._speech_head_engine,
            self._transformer_engine,
        ]:
            if engine:
                engine.shutdown()


class StreamingChatterboxTurboTTS:
    """
    Async wrapper for Chatterbox TRT TTS
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._stop_event = Event()

    def load(self):
        """Load TTS model"""
        if self._model is not None:
            return

        logger.info("Loading Chatterbox TRT TTS...")
        self._model = ChatterboxTRTTTS(self.config)
        self._model.load()

        # Warmup
        self._warmup()
        logger.info("Chatterbox TRT TTS ready")

    def _warmup(self):
        """Warmup the model"""
        warmup_texts = [
            "Hello there.",
            "This is a warmup test.",
        ]
        try:
            for text in warmup_texts:
                for _ in self._model.generate_stream(text, chunk_size=self.config.chunk_size):
                    pass
            logger.info("TTS warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    @property
    def sample_rate(self) -> int:
        """Get sample rate"""
        return self.config.sample_rate

    def stop(self):
        """Stop generation"""
        self._stop_event.set()
        if self._model:
            self._model.stop()

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
        Yields (audio_chunk, metrics) tuples
        """
        if self._model is None:
            self.load()

        self._stop_event.clear()

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

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while not done_event.is_set() or not chunk_queue.empty():
            try:
                audio_chunk, metrics = chunk_queue.get(timeout=0.05)
                yield (audio_chunk.astype(np.float32), metrics)
            except queue.Empty:
                await asyncio.sleep(0.01)

        thread.join(timeout=1.0)

    async def synthesize(self, text: str) -> np.ndarray:
        """Full synthesis"""
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
        if self._model:
            self._model.shutdown()


class StreamingTTS:
    """
    High-level streaming TTS wrapper
    Provides async interface with callbacks
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
