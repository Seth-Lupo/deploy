"""
TTS Module - Chatterbox TURBO with TensorRT GPT2 Transformer
Uses TensorRT for T3 GPT2 transformer (autoregressive)
Keeps embeddings, heads, and S3Gen in PyTorch
"""

import asyncio
import logging
import time
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Callable, Any, Tuple, Generator, List
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS module"""
    model_path: str = "/workspace/models/chatterbox-turbo-trt"
    device: str = "cuda"
    sample_rate: int = 24000
    voice_reference: Optional[str] = None
    # Generation settings
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    # Streaming settings
    chunk_size: int = 25
    context_window: int = 15
    max_gen_len: int = 1000


class GPT2TRTRunner:
    """
    TensorRT runner for GPT2 transformer with KV cache
    """

    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self._engine = None
        self._context = None
        self._stream = None
        self._bindings = {}
        self._n_layer = 24
        self._n_head = 16
        self._head_dim = 64

    def load(self):
        """Load TRT engine"""
        from trt_runtime import TRTEngine

        logger.info(f"Loading GPT2 TRT engine: {self.engine_path}")
        self._engine = TRTEngine(self.engine_path)
        self._engine.load()
        logger.info("GPT2 TRT engine loaded")

    def run(
        self,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray,
        past_key_values: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Run GPT2 transformer

        Args:
            inputs_embeds: [batch, seq_len, 1024]
            attention_mask: [batch, total_len]
            past_key_values: List of (key, value) tuples for each layer

        Returns:
            last_hidden_state: [batch, seq_len, 1024]
            present_key_values: Updated KV cache
        """
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]

        # Initialize empty KV cache if not provided
        if past_key_values is None:
            past_len = 1  # TRT engine expects min 1
            past_key_values = [
                (
                    np.zeros((batch_size, self._n_head, past_len, self._head_dim), dtype=np.float32),
                    np.zeros((batch_size, self._n_head, past_len, self._head_dim), dtype=np.float32),
                )
                for _ in range(self._n_layer)
            ]

        # Build input dict
        inputs = {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask.astype(np.int64),
        }

        # Add KV cache inputs
        for i, (k, v) in enumerate(past_key_values):
            inputs[f"past_key_values.{i}.key"] = k.astype(np.float32)
            inputs[f"past_key_values.{i}.value"] = v.astype(np.float32)

        # Run inference
        outputs = self._engine.infer(inputs)

        # Extract outputs
        last_hidden_state = outputs["last_hidden_state"]

        # Extract updated KV cache
        present_key_values = []
        for i in range(self._n_layer):
            k = outputs[f"present.{i}.key"]
            v = outputs[f"present.{i}.value"]
            present_key_values.append((k, v))

        return last_hidden_state, present_key_values

    def shutdown(self):
        if self._engine:
            self._engine.shutdown()


class ChatterboxHybridTTS:
    """
    Chatterbox TURBO TTS with hybrid TRT/PyTorch
    - GPT2 transformer uses TensorRT
    - Embeddings, heads, S3Gen use PyTorch
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.sr = self.config.sample_rate

        # TRT for transformer
        self._gpt2_trt = None

        # PyTorch components
        self._t3 = None  # Full T3 model for embeddings/heads
        self._s3gen = None
        self._conds = None

        self._executor = ThreadPoolExecutor(max_workers=2)
        self._stop_event = Event()

    def load(self):
        """Load TRT engine and PyTorch components"""
        if self._gpt2_trt is not None:
            return

        import torch

        # Load TRT GPT2 transformer
        trt_path = os.path.join(self.config.model_path, "gpt2_transformer.trt")
        if os.path.exists(trt_path):
            self._gpt2_trt = GPT2TRTRunner(trt_path)
            self._gpt2_trt.load()
        else:
            logger.error(f"GPT2 TRT engine not found: {trt_path}")
            raise FileNotFoundError(f"Missing TRT engine: {trt_path}")

        # Load Chatterbox PyTorch model for embeddings/heads/S3Gen
        logger.info("Loading Chatterbox PyTorch components...")
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            full_model = ChatterboxTurboTTS.from_pretrained(device=self.config.device)
            self._t3 = full_model.t3
            self._s3gen = full_model.s3gen

            # Set to eval mode
            self._t3.eval()
            self._s3gen.eval()

            logger.info("Chatterbox PyTorch components loaded")
        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
            raise

        # Warmup
        self._warmup()
        logger.info("Chatterbox Hybrid TTS ready")

    def _warmup(self):
        """Warmup TRT engine"""
        import torch

        logger.info("Warming up TRT engine...")

        # Warmup with small sequence
        batch_size = 1
        seq_len = 10
        past_len = 1

        inputs_embeds = np.random.randn(batch_size, seq_len, 1024).astype(np.float32)
        attention_mask = np.ones((batch_size, seq_len + past_len), dtype=np.int64)

        for _ in range(3):
            self._gpt2_trt.run(inputs_embeds, attention_mask)

        logger.info("TRT warmup complete")

    def prepare_conditionals(self, audio_path: str):
        """Prepare conditioning from reference audio"""
        import torch
        import torchaudio

        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != 24000:
                wav = torchaudio.functional.resample(wav, sr, 24000)
            wav = wav.mean(0) if wav.dim() > 1 else wav
            wav = wav.to(self.config.device)

            # Use S3Gen to extract conditioning
            with torch.no_grad():
                # Get speaker embedding
                ref_dict = self._s3gen.speaker_encoder(wav.unsqueeze(0))
                self._conds = {'ref_wav': wav, 'ref_sr': 24000}

            logger.info(f"Prepared conditionals from: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to prepare conditionals: {e}")
            self._conds = None

    def _generate_speech_tokens_hybrid(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        max_gen_len: int = 1000,
        chunk_size: int = 25,
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate speech tokens using TRT transformer + PyTorch embeddings/heads
        """
        import torch
        import torch.nn.functional as F

        device = self.config.device

        # Get T3 components
        text_emb = self._t3.text_emb
        speech_emb = self._t3.speech_emb
        speech_head = self._t3.speech_head

        # Tokenize text using T3's method
        # T3 uses its own text tokenization
        with torch.no_grad():
            # Prepare text input - T3 expects specific format
            # Use the model's prepare method if available
            if hasattr(self._t3, 'prepare_input_embeds'):
                # Get text embeddings through T3
                text_tokens = self._t3.hp.text_tokenizer.encode(text)
                text_tokens = torch.tensor([text_tokens], device=device, dtype=torch.long)
                text_embeds = text_emb(text_tokens)
            else:
                # Fallback: use GPT2 tokenizer
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                text_tokens = tokenizer.encode(text, return_tensors="pt").to(device)
                text_embeds = text_emb(text_tokens)

        # Speech token constants
        start_speech_token = 6561
        stop_speech_token = 6562
        vocab_size = 6563

        # Initialize speech tokens
        speech_tokens = [start_speech_token]
        chunk_buffer = []

        # Get initial speech embedding
        with torch.no_grad():
            speech_input = torch.tensor([[start_speech_token]], device=device, dtype=torch.long)
            speech_embeds = speech_emb(speech_input)

            # Combine embeddings
            combined_embeds = torch.cat([text_embeds, speech_embeds], dim=1)

        # Convert to numpy for TRT
        combined_np = combined_embeds.cpu().numpy()
        seq_len = combined_np.shape[1]
        attention_mask = np.ones((1, seq_len), dtype=np.int64)

        # Initial TRT forward pass
        hidden_np, kv_cache = self._gpt2_trt.run(combined_np, attention_mask)

        # Get logits from last hidden state using PyTorch head
        with torch.no_grad():
            hidden_torch = torch.from_numpy(hidden_np[:, -1:, :]).to(device)
            logits = speech_head(hidden_torch)
            logits = logits.squeeze().cpu().numpy()

        # Sample first token
        next_token = self._sample_token(logits, temperature, top_k, top_p, speech_tokens, repetition_penalty)
        speech_tokens.append(next_token)
        chunk_buffer.append(next_token)

        # Autoregressive generation with KV cache
        for i in range(max_gen_len):
            if self._stop_event.is_set():
                break

            # Check for EOS
            if next_token == stop_speech_token:
                if chunk_buffer:
                    yield np.array(chunk_buffer[:-1], dtype=np.int64)
                break

            # Yield chunk
            if len(chunk_buffer) >= chunk_size:
                yield np.array(chunk_buffer, dtype=np.int64)
                chunk_buffer = []

            # Get embedding for current token
            with torch.no_grad():
                current_input = torch.tensor([[next_token]], device=device, dtype=torch.long)
                current_embed = speech_emb(current_input)
                current_np = current_embed.cpu().numpy()

            # Update attention mask for KV cache
            past_len = kv_cache[0][0].shape[2]
            attention_mask = np.ones((1, past_len + 1), dtype=np.int64)

            # TRT forward with KV cache (single token)
            hidden_np, kv_cache = self._gpt2_trt.run(current_np, attention_mask, kv_cache)

            # Get logits
            with torch.no_grad():
                hidden_torch = torch.from_numpy(hidden_np[:, -1:, :]).to(device)
                logits = speech_head(hidden_torch)
                logits = logits.squeeze().cpu().numpy()

            # Sample next token
            next_token = self._sample_token(logits, temperature, top_k, top_p, speech_tokens, repetition_penalty)
            speech_tokens.append(next_token)
            chunk_buffer.append(next_token)

        # Yield remaining
        if chunk_buffer and chunk_buffer[-1] != stop_speech_token:
            yield np.array(chunk_buffer, dtype=np.int64)

    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        prev_tokens: List[int],
        repetition_penalty: float,
    ) -> int:
        """Sample next token with repetition penalty"""
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token in set(prev_tokens[-50:]):  # Look at last 50 tokens
                if token < len(logits):
                    logits[token] /= repetition_penalty

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0 and top_k < len(logits):
            indices = np.argsort(logits)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[indices] = logits[indices]
            logits = mask

        # Softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / (probs.sum() + 1e-10)

        # Top-p filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_indices])
            cutoff_idx = np.searchsorted(cumsum, top_p) + 1
            mask_indices = sorted_indices[cutoff_idx:]
            probs[mask_indices] = 0
            probs = probs / (probs.sum() + 1e-10)

        # Sample
        return int(np.random.choice(len(probs), p=probs))

    def _tokens_to_audio(self, tokens: np.ndarray) -> np.ndarray:
        """Convert speech tokens to audio using S3Gen (PyTorch)"""
        if self._s3gen is None:
            logger.error("S3Gen not loaded")
            return np.array([], dtype=np.float32)

        import torch

        # Filter out invalid tokens
        valid_tokens = tokens[(tokens >= 0) & (tokens < 6561)]
        if len(valid_tokens) == 0:
            return np.array([], dtype=np.float32)

        tokens_tensor = torch.from_numpy(valid_tokens).long().to(self.config.device)

        with torch.no_grad():
            # S3Gen inference
            ref_wav = self._conds.get('ref_wav') if self._conds else None
            ref_sr = self._conds.get('ref_sr') if self._conds else None

            wav = self._s3gen(
                speech_tokens=tokens_tensor.unsqueeze(0),
                ref_wav=ref_wav.unsqueeze(0) if ref_wav is not None else None,
                ref_sr=ref_sr,
                finalize=True,
                n_cfm_timesteps=2,  # Turbo mode
            )

            wav = wav.squeeze().cpu().numpy()

        return wav

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        chunk_size: int = 25,
        context_window: int = 15,
        print_metrics: bool = False,
    ) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """Stream audio chunks"""
        start_time = time.time()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path)

        self._stop_event.clear()

        all_tokens = []
        chunk_count = 0
        total_audio_duration = 0.0
        first_chunk_time = None

        for token_chunk in self._generate_speech_tokens_hybrid(
            text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            chunk_size=chunk_size,
            max_gen_len=self.config.max_gen_len,
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
        """Stop generation"""
        self._stop_event.set()

    def shutdown(self):
        """Cleanup"""
        self._stop_event.set()
        self._executor.shutdown(wait=False)
        if self._gpt2_trt:
            self._gpt2_trt.shutdown()


class StreamingTTS:
    """High-level streaming TTS wrapper"""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._model = ChatterboxHybridTTS(self.config)
        self._on_audio: Optional[Callable[[np.ndarray, dict], Any]] = None
        self._on_complete: Optional[Callable[[], Any]] = None
        self._stop_event = Event()

    def set_callbacks(
        self,
        on_audio: Optional[Callable[[np.ndarray, dict], Any]] = None,
        on_complete: Optional[Callable[[], Any]] = None
    ):
        self._on_audio = on_audio
        self._on_complete = on_complete

    def load(self):
        self._model.load()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def set_voice(self, reference_path: str):
        self.config.voice_reference = reference_path

    def stop(self):
        self._stop_event.set()
        self._model.stop()

    async def speak(self, text: str) -> AsyncIterator[np.ndarray]:
        """Synthesize text to audio, streaming chunks"""
        self._stop_event.clear()

        import queue
        import threading

        chunk_queue = queue.Queue()
        done_event = threading.Event()

        def producer():
            try:
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
                    **kwargs
                ):
                    if self._stop_event.is_set():
                        break
                    chunk_queue.put((audio_chunk, metrics))
            except Exception as e:
                logger.error(f"TTS producer error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                done_event.set()

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        while not done_event.is_set() or not chunk_queue.empty():
            try:
                audio_chunk, metrics = chunk_queue.get(timeout=0.05)
                if self._on_audio:
                    self._on_audio(audio_chunk, metrics)
                yield audio_chunk
            except queue.Empty:
                await asyncio.sleep(0.01)

        thread.join(timeout=1.0)

        if self._on_complete:
            self._on_complete()

    async def speak_full(self, text: str) -> np.ndarray:
        """Synthesize complete audio"""
        chunks = []
        async for chunk in self.speak(text):
            chunks.append(chunk)
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    def shutdown(self):
        self._stop_event.set()
        self._model.shutdown()
