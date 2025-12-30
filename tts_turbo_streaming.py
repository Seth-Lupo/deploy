"""
ChatterboxTurboTTS with Streaming Support
Extends the official ChatterboxTurboTTS with streaming audio output
"""

import time
from dataclasses import dataclass
from typing import Generator, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)

import logging
logger = logging.getLogger(__name__)

# Import from official chatterbox-tts package
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm, S3GEN_SIL


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0


class ChatterboxTurboStreamingTTS(ChatterboxTurboTTS):
    """
    ChatterboxTurboTTS with TRUE STREAMING
    - Uses Turbo's 2-step CFM for 5x faster inference
    - Streams audio chunks as they're generated
    """

    def inference_turbo_stream(
        self,
        t3_cond,
        text_tokens: torch.Tensor,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = 1000,
        chunk_size: int = 25,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Streaming version of inference_turbo that yields speech token chunks

        Args:
            t3_cond: T3 conditioning
            text_tokens: Tokenized input text
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty
            max_gen_len: Maximum generation length
            chunk_size: Number of tokens per yielded chunk

        Yields:
            torch.Tensor: Chunk of speech tokens
        """
        # Setup logits processors
        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        # Prepare initial embeddings
        speech_start_token = self.t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.t3.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        generated_speech_tokens = []
        chunk_buffer = []

        # Initial forward pass
        llm_outputs = self.t3.tfmr(
            inputs_embeds=embeds,
            use_cache=True
        )

        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_hidden = hidden_states[:, -1:]
        speech_logits = self.t3.speech_head(speech_hidden)

        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens.append(next_speech_token)
        chunk_buffer.append(next_speech_token)
        current_speech_token = next_speech_token

        # Generation loop
        for i in range(max_gen_len):
            current_speech_embed = self.t3.speech_emb(current_speech_token)

            llm_outputs = self.t3.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True
            )

            hidden_states = llm_outputs[0]
            past_key_values = llm_outputs.past_key_values
            speech_logits = self.t3.speech_head(hidden_states)

            input_ids = torch.cat(generated_speech_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])

            if torch.all(processed_logits == -float("inf")):
                logger.warning("All logits are -inf, stopping")
                break

            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)

            generated_speech_tokens.append(next_speech_token)
            chunk_buffer.append(next_speech_token)
            current_speech_token = next_speech_token

            # Check for EOS
            if torch.all(next_speech_token == self.t3.hp.stop_speech_token):
                if chunk_buffer:
                    yield torch.cat(chunk_buffer, dim=1)
                    chunk_buffer = []  # Clear to prevent double-yield
                break

            # Yield chunk when buffer is full
            if len(chunk_buffer) >= chunk_size:
                yield torch.cat(chunk_buffer, dim=1)
                chunk_buffer = []

        # Yield any remaining tokens (only if we didn't hit EOS)
        if chunk_buffer:
            yield torch.cat(chunk_buffer, dim=1)

    def _process_token_buffer(
        self,
        token_buffer,
        all_tokens_so_far,
        context_window: int,
        start_time: float,
        metrics: StreamingMetrics,
        print_metrics: bool,
        fade_duration: float = 0.02,
    ):
        """
        Process a buffer of speech tokens into audio using Turbo's 2-step CFM

        Args:
            token_buffer: List of token tensors to process
            all_tokens_so_far: All tokens processed so far (for context)
            context_window: Number of context tokens to include
            start_time: Generation start time
            metrics: StreamingMetrics object to update
            print_metrics: Whether to print metrics
            fade_duration: Fade-in duration in seconds

        Returns:
            Tuple of (audio_tensor, audio_duration, success)
        """
        # Combine buffered chunks
        new_tokens = torch.cat(token_buffer, dim=-1) if len(token_buffer) > 1 else token_buffer[0]

        # Flatten if needed
        if new_tokens.dim() > 1:
            new_tokens = new_tokens.squeeze(0)

        # Build tokens with context window
        if len(all_tokens_so_far) > 0 and context_window > 0:
            if isinstance(all_tokens_so_far, list):
                all_tokens_flat = torch.cat(all_tokens_so_far, dim=-1)
                if all_tokens_flat.dim() > 1:
                    all_tokens_flat = all_tokens_flat.squeeze(0)
            else:
                all_tokens_flat = all_tokens_so_far
                if all_tokens_flat.dim() > 1:
                    all_tokens_flat = all_tokens_flat.squeeze(0)

            context_tokens = (
                all_tokens_flat[-context_window:]
                if len(all_tokens_flat) > context_window
                else all_tokens_flat
            )
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)

            # Count valid tokens in context vs new for accurate cropping
            context_valid = (context_tokens < 6561).sum().item()
            new_valid = (new_tokens < 6561).sum().item()
        else:
            tokens_to_process = new_tokens
            context_valid = 0
            new_valid = (new_tokens < 6561).sum().item()

        # Remove invalid tokens (OOV)
        clean_tokens = tokens_to_process[tokens_to_process < 6561].to(self.device)
        if len(clean_tokens) == 0:
            return None, 0.0, False

        # Run S3Gen with Turbo's 2-step CFM
        wav, _ = self.s3gen.inference(
            speech_tokens=clean_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,  # TURBO: 2 steps instead of 10!
        )
        wav = wav.squeeze(0).detach().cpu().numpy()

        # Crop out context samples using valid token counts
        if context_valid > 0:
            samples_per_token = len(wav) / len(clean_tokens)
            skip_samples = int(context_valid * samples_per_token)
            # Add small overlap to prevent gaps (5ms worth)
            overlap_samples = int(0.005 * self.sr)
            skip_samples = max(0, skip_samples - overlap_samples)
            audio_chunk = wav[skip_samples:]
        else:
            audio_chunk = wav

        if len(audio_chunk) == 0:
            return None, 0.0, False

        # Apply fade-in for smooth boundaries
        fade_samples = int(fade_duration * self.sr)
        if fade_samples > 0 and fade_samples < len(audio_chunk):
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=audio_chunk.dtype)
            audio_chunk[:fade_samples] *= fade_in

        # Watermark and return
        audio_duration = len(audio_chunk) / self.sr
        watermarked_chunk = audio_chunk  # Watermarking disabled for speed
        audio_tensor = torch.from_numpy(watermarked_chunk).unsqueeze(0)

        # Update metrics
        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            if print_metrics:
                logger.info(f"Latency to first chunk: {metrics.latency_to_first_chunk:.3f}s")

        metrics.chunk_count += 1
        return audio_tensor, audio_duration, True

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.0,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        chunk_size: int = 40,
        context_window: int = 15,
        fade_duration: float = 0.02,
        print_metrics: bool = False,
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        """
        Stream audio chunks as they're generated using Turbo's fast inference

        Args:
            text: Input text to synthesize
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration (ignored in Turbo)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            repetition_penalty: Repetition penalty
            chunk_size: Number of speech tokens per chunk
            context_window: Context window for continuity
            fade_duration: Fade-in duration for smooth chunk boundaries
            print_metrics: Whether to print timing metrics

        Yields:
            Tuple of (audio_chunk, metrics)
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        total_audio_length = 0.0
        all_tokens_processed = []

        with torch.inference_mode():
            for token_chunk in self.inference_turbo_stream(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                chunk_size=chunk_size,
            ):
                # Process chunk to audio
                audio_tensor, audio_duration, success = self._process_token_buffer(
                    [token_chunk],
                    all_tokens_processed,
                    context_window,
                    start_time,
                    metrics,
                    print_metrics,
                    fade_duration,
                )

                if success:
                    total_audio_length += audio_duration
                    yield audio_tensor, metrics

                # Update processed tokens
                all_tokens_processed.append(token_chunk)

        # Final metrics
        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = total_audio_length
        if total_audio_length > 0:
            metrics.rtf = metrics.total_generation_time / total_audio_length
            if print_metrics:
                logger.info(f"Total generation time: {metrics.total_generation_time:.3f}s")
                logger.info(f"Total audio duration: {metrics.total_audio_duration:.3f}s")
                logger.info(f"RTF: {metrics.rtf:.3f}")
                logger.info(f"Chunks: {metrics.chunk_count}")
