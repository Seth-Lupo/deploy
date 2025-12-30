"""
LLM Module - Qwen3 4B Instruct via Transformers (FP16)
Streaming text generation using HuggingFace transformers
Optimized for A100/A10G GPUs
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Event, Thread

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM module"""
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    device: str = "cuda"
    dtype: str = "float16"  # float16, bfloat16, or float32
    # Generation settings
    max_new_tokens: int = 150
    temperature: float = 0.6
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.1
    do_sample: bool = True
    # Sentence splitting - only split after this many chars (0 = never split mid-response)
    min_chars_before_split: int = 300
    # Chat settings
    system_prompt: str = """You are a fast, helpful voice assistant. Keep responses extremely brief - one sentence when possible. Never use emojis, asterisks, or special formatting. Speak naturally and directly. Get to the point immediately."""


class SentenceBuffer:
    """
    Buffer that accumulates tokens and yields complete sentences
    Only splits on sentence boundaries after min_chars threshold
    """

    def __init__(self, min_chars_before_split: int = 300):
        self._buffer = ""
        self._min_chars = min_chars_before_split
        # Only split on actual sentence endings
        self._end_patterns = re.compile(r'[.!?]+[\s"\')\]]*')
        # Thinking tags pattern (for Qwen3)
        self._thinking_pattern = re.compile(r'<think>.*?</think>', re.DOTALL)

    def add_token(self, token: str) -> Optional[str]:
        """
        Add token and return complete sentence if available
        Only splits if buffer has accumulated >= min_chars
        """
        self._buffer += token

        # Remove thinking tags
        self._buffer = self._thinking_pattern.sub('', self._buffer)

        # Only check for split if we have enough characters
        if len(self._buffer) < self._min_chars:
            return None

        # Check for sentence boundary
        match = self._end_patterns.search(self._buffer)
        if match:
            end_pos = match.end()
            sentence = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[end_pos:].lstrip()

            if len(sentence) > 0:
                return sentence

        return None

    def flush(self) -> Optional[str]:
        """Return any remaining text"""
        text = self._buffer.strip()
        self._buffer = ""
        return text if text else None

    def reset(self):
        """Clear buffer"""
        self._buffer = ""


class QwenLLM:
    """
    Qwen3 LLM using HuggingFace Transformers
    FP16 inference optimized for A100/A10G
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._model = None
        self._tokenizer = None
        self._streamer = None
        self._stop_event = Event()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def load(self):
        """Load model and tokenizer"""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.config.model_id}")
        logger.info(f"Device: {self.config.device}, Dtype: {self.config.dtype}")

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float16)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True
        )

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self._model.eval()

        # Warmup
        self._warmup()
        logger.info("Qwen LLM loaded and ready")

    def _warmup(self):
        """Warmup the model with a simple generation"""
        try:
            messages = [{"role": "user", "content": "Hi"}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                _ = self._model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            logger.info("LLM warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def _generate_streaming(
        self,
        messages: List[dict],
        output_queue: Queue,
    ):
        """Streaming generation in a separate thread"""
        import torch
        from transformers import TextIteratorStreamer

        self._stop_event.clear()

        try:
            # Apply chat template
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking for voice responses
            )

            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

            # Create streamer
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Generation kwargs
            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "do_sample": self.config.do_sample,
                "pad_token_id": self._tokenizer.eos_token_id,
            }

            # Start generation in a thread
            def generate():
                with torch.no_grad():
                    self._model.generate(**gen_kwargs)

            gen_thread = Thread(target=generate)
            gen_thread.start()

            # Stream tokens
            for token in streamer:
                if self._stop_event.is_set():
                    break
                if token:
                    output_queue.put(token)

            gen_thread.join(timeout=1.0)

        except Exception as e:
            logger.error(f"Generation error: {e}")

        output_queue.put(None)  # Signal completion

    async def generate_stream(
        self,
        messages: List[dict],
    ) -> AsyncIterator[str]:
        """
        Async streaming generation
        Yields tokens as they're generated
        """
        if self._model is None:
            self.load()

        output_queue = Queue()

        # Run generation in executor
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            self._executor,
            self._generate_streaming,
            messages,
            output_queue
        )

        # Yield tokens as they arrive
        while True:
            await asyncio.sleep(0.001)
            try:
                token = output_queue.get_nowait()
                if token is None:
                    break
                yield token
            except Empty:
                if future.done():
                    # Drain remaining
                    while True:
                        try:
                            token = output_queue.get_nowait()
                            if token is None:
                                break
                            yield token
                        except Empty:
                            break
                    break

    def stop(self):
        """Stop current generation"""
        self._stop_event.set()

    def shutdown(self):
        """Cleanup resources"""
        self._stop_event.set()
        self._executor.shutdown(wait=False)
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None


class StreamingLLM:
    """
    High-level streaming LLM interface
    Manages conversation history and streams sentences
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._llm = QwenLLM(self.config)
        self._sentence_buffer = SentenceBuffer(self.config.min_chars_before_split)
        self._conversation: List[dict] = []
        self._on_token: Optional[Callable[[str], Any]] = None
        self._on_sentence: Optional[Callable[[str], Any]] = None

    def set_callbacks(
        self,
        on_token: Optional[Callable[[str], Any]] = None,
        on_sentence: Optional[Callable[[str], Any]] = None
    ):
        """Set callbacks for streaming events"""
        self._on_token = on_token
        self._on_sentence = on_sentence

    def load(self):
        """Load the model"""
        self._llm.load()

    def set_system_prompt(self, prompt: str):
        """Update system prompt"""
        self.config.system_prompt = prompt

    def reset(self):
        """Reset conversation history"""
        self._conversation = []
        self._sentence_buffer.reset()

    async def respond(self, user_input: str) -> AsyncIterator[str]:
        """
        Generate response to user input
        Yields complete sentences as they're generated
        """
        # Build messages
        messages = []

        # Add system prompt
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })

        # Add conversation history
        messages.extend(self._conversation)

        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })

        # Track for history
        self._conversation.append({
            "role": "user",
            "content": user_input
        })

        # Generate response
        full_response = ""
        self._sentence_buffer.reset()

        async for token in self._llm.generate_stream(messages):
            if self._on_token:
                self._on_token(token)

            full_response += token

            # Check for complete sentence
            sentence = self._sentence_buffer.add_token(token)
            if sentence:
                if self._on_sentence:
                    self._on_sentence(sentence)
                yield sentence

        # Flush any remaining text
        remaining = self._sentence_buffer.flush()
        if remaining:
            if self._on_sentence:
                self._on_sentence(remaining)
            yield remaining

        # Add to history
        if full_response:
            self._conversation.append({
                "role": "assistant",
                "content": full_response
            })

        # Trim history if too long (keep last 10 turns)
        if len(self._conversation) > 20:
            self._conversation = self._conversation[-20:]

    def stop(self):
        """Stop current generation"""
        self._llm.stop()

    def shutdown(self):
        """Cleanup"""
        self._llm.shutdown()


# Import for backwards compatibility
import torch
