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
    exaggeration: float = 0.0
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

        # Log available inputs/outputs
        logger.info(f"  Inputs: {self._engine.input_names}")
        logger.info(f"  Outputs: {self._engine.output_names}")
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
            attention_mask: [batch, seq_len] - for current tokens only
            past_key_values: List of (key, value) tuples for each layer

        Returns:
            last_hidden_state: [batch, seq_len, 1024]
            present_key_values: Updated KV cache
        """
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]

        # Determine past length from KV cache
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
        else:
            past_len = 1  # TRT engine was built with min past_len=1

        # Initialize minimal KV cache if not provided (TRT requires these inputs)
        if past_key_values is None:
            past_key_values = [
                (
                    np.zeros((batch_size, self._n_head, past_len, self._head_dim), dtype=np.float32),
                    np.zeros((batch_size, self._n_head, past_len, self._head_dim), dtype=np.float32),
                )
                for _ in range(self._n_layer)
            ]

        # Attention mask: TRT engine expects [batch, past_len + seq_len]
        total_len = past_len + seq_len
        attention_mask = np.ones((batch_size, total_len), dtype=np.int64)

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
        try:
            outputs = self._engine.infer(inputs)
        except Exception as e:
            logger.error(f"TRT inference failed: {e}")
            logger.error(f"  inputs_embeds shape: {inputs_embeds.shape}")
            logger.error(f"  attention_mask shape: {attention_mask.shape}")
            logger.error(f"  past_len: {past_len}, seq_len: {seq_len}")
            raise

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
        import torch.nn as nn

        # Load TRT GPT2 transformer
        trt_path = os.path.join(self.config.model_path, "gpt2_transformer.trt")
        if os.path.exists(trt_path):
            self._gpt2_trt = GPT2TRTRunner(trt_path)
            self._gpt2_trt.load()
        else:
            logger.error(f"GPT2 TRT engine not found: {trt_path}")
            raise FileNotFoundError(f"Missing TRT engine: {trt_path}")

        # Load T3 components from safetensors/weights
        logger.info("Loading T3 embeddings and heads...")
        self._load_t3_components()

        # Load S3Gen vocoder
        logger.info("Loading S3Gen vocoder...")
        self._load_s3gen()

        # Load text tokenizer
        logger.info("Loading tokenizer...")
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Warmup
        self._warmup()
        logger.info("Chatterbox Hybrid TTS ready")

    def _load_t3_components(self):
        """Load T3 embedding layers and output heads from weights"""
        import torch
        import torch.nn as nn

        device = self.config.device

        # T3 config from the model
        vocab_size_text = 50276  # GPT2 vocab
        vocab_size_speech = 6563  # Speech tokens (6561 + start + stop)
        hidden_size = 1024

        # Create embedding layers
        self._text_emb = nn.Embedding(vocab_size_text, hidden_size).to(device)
        self._speech_emb = nn.Embedding(vocab_size_speech, hidden_size).to(device)
        self._speech_head = nn.Linear(hidden_size, vocab_size_speech, bias=False).to(device)

        # Try to load weights from various sources
        weight_files = [
            os.path.join(self.config.model_path, "t3_components.pt"),
            os.path.join(self.config.model_path, "t3.safetensors"),
            os.path.join(self.config.model_path, "model.safetensors"),
        ]

        weights_loaded = False
        for wf in weight_files:
            if os.path.exists(wf):
                try:
                    if wf.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(wf)
                    else:
                        state_dict = torch.load(wf, map_location=device, weights_only=True)

                    # Map weights - handle different naming conventions
                    if 'text_emb.weight' in state_dict:
                        self._text_emb.weight.data = state_dict['text_emb.weight']
                    if 'speech_emb.weight' in state_dict:
                        self._speech_emb.weight.data = state_dict['speech_emb.weight']
                    if 'speech_head.weight' in state_dict:
                        self._speech_head.weight.data = state_dict['speech_head.weight']

                    weights_loaded = True
                    logger.info(f"Loaded T3 weights from: {wf}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {wf}: {e}")

        if not weights_loaded:
            # Initialize with random weights - will need proper weights for good output
            logger.warning("No T3 weights found, using random initialization (quality will be poor)")
            nn.init.normal_(self._text_emb.weight, std=0.02)
            nn.init.normal_(self._speech_emb.weight, std=0.02)
            nn.init.normal_(self._speech_head.weight, std=0.02)

        self._text_emb.eval()
        self._speech_emb.eval()
        self._speech_head.eval()

    def _load_s3gen(self):
        """Load S3Gen vocoder"""
        import torch
        import sys

        # Add deploy directory to path for s3gen import
        deploy_dir = os.path.dirname(os.path.abspath(__file__))
        if deploy_dir not in sys.path:
            sys.path.insert(0, deploy_dir)

        # Try standalone S3Gen first
        try:
            from s3gen import S3Token2Wav

            weight_files = [
                os.path.join(self.config.model_path, "s3gen.safetensors"),
                os.path.join(self.config.model_path, "s3gen_meanflow.safetensors"),
                os.path.join(self.config.model_path, "s3gen.pt"),
            ]

            for wf in weight_files:
                if os.path.exists(wf):
                    self._s3gen = S3Token2Wav()
                    if wf.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(wf)
                    else:
                        state_dict = torch.load(wf, map_location=self.config.device, weights_only=False)
                    self._s3gen.load_state_dict(state_dict, strict=False)
                    self._s3gen.to(self.config.device).eval()
                    logger.info(f"Loaded S3Gen from: {wf}")
                    return
        except ImportError:
            logger.warning("s3gen module not found")
        except Exception as e:
            logger.warning(f"Failed to load S3Gen: {e}")

        logger.error("S3Gen vocoder not loaded - TTS will not produce audio")
        self._s3gen = None

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

        # Tokenize text using GPT2 tokenizer
        with torch.no_grad():
            text_tokens = self._tokenizer.encode(text, return_tensors="pt").to(device)
            text_embeds = self._text_emb(text_tokens)

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
            speech_embeds = self._speech_emb(speech_input)

            # Combine embeddings
            combined_embeds = torch.cat([text_embeds, speech_embeds], dim=1)

        # Autoregressive generation (no KV cache for simplicity - recompute each step)
        # This is slower but more reliable with TRT
        all_embeds = combined_embeds  # [1, seq_len, 1024]

        for i in range(max_gen_len):
            if self._stop_event.is_set():
                break

            # Convert to numpy for TRT
            embeds_np = all_embeds.cpu().numpy()
            seq_len = embeds_np.shape[1]
            attention_mask = np.ones((1, seq_len), dtype=np.int64)

            # TRT forward (no KV cache)
            hidden_np, _ = self._gpt2_trt.run(embeds_np, attention_mask, None)

            # Get logits from last hidden state
            with torch.no_grad():
                hidden_torch = torch.from_numpy(hidden_np[:, -1:, :]).to(device)
                logits = self._speech_head(hidden_torch)
                logits = logits.squeeze().cpu().numpy()

            # Sample next token
            next_token = self._sample_token(logits, temperature, top_k, top_p, speech_tokens, repetition_penalty)
            speech_tokens.append(next_token)
            chunk_buffer.append(next_token)

            # Check for EOS
            if next_token == stop_speech_token:
                if chunk_buffer:
                    yield np.array(chunk_buffer[:-1], dtype=np.int64)
                break

            # Yield chunk
            if len(chunk_buffer) >= chunk_size:
                yield np.array(chunk_buffer, dtype=np.int64)
                chunk_buffer = []

            # Append new token embedding for next iteration
            with torch.no_grad():
                new_token_input = torch.tensor([[next_token]], device=device, dtype=torch.long)
                new_embed = self._speech_emb(new_token_input)
                all_embeds = torch.cat([all_embeds, new_embed], dim=1)

            # Limit sequence length to avoid OOM
            if all_embeds.shape[1] > 500:
                logger.warning("Sequence too long, truncating")
                break

        # Yield remaining
        if chunk_buffer and (not chunk_buffer or chunk_buffer[-1] != stop_speech_token):
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
