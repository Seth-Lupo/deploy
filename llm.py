"""
LLM Module - Qwen3 4B Instruct with TensorRT-LLM (INT8)
Streaming text generation using TensorRT-LLM
Optimized for A100/A10G/L4 GPUs
"""

import asyncio
import logging
import re
import time
import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Event, Thread

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM module"""
    engine_path: str = "/workspace/models/qwen3-4b-int8wo-engine"
    tokenizer_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    device: str = "cuda"
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
    system_prompt: str = """You are a fast, helpful voice assistant. Keep responses extremely brief - one sentence when possible.

CRITICAL RULES:
- NEVER use emojis, asterisks, or special formatting
- ALWAYS write numbers as words (say "twenty three" not "23")
- For phone numbers, spell each digit: "five five five, one two three, four five six seven"
- Speak naturally and directly. Get to the point immediately.

You can use these natural speech sounds sparingly for expressiveness:
[clear throat] [sigh] [shush] [cough] [groan] [sniff] [gasp] [chuckle] [laugh]

Example: "[chuckle] That's a great question." or "Hmm, [sigh] let me think about that." """


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


class QwenTRTLLM:
    """
    Qwen3 LLM using TensorRT-LLM
    INT8 weight-only quantized for optimal performance
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._runner = None
        self._tokenizer = None
        self._stop_event = Event()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def load(self):
        """Load TensorRT-LLM engine and tokenizer"""
        if self._runner is not None:
            return

        logger.info(f"Loading TensorRT-LLM engine: {self.config.engine_path}")

        # Load tokenizer from HuggingFace
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path,
            trust_remote_code=True
        )

        # Load TensorRT-LLM runner
        try:
            from tensorrt_llm.runtime import ModelRunnerCpp
            from tensorrt_llm.bindings import GptJsonConfig

            # Load engine config
            config_path = os.path.join(self.config.engine_path, "config.json")
            engine_path = os.path.join(self.config.engine_path, "rank0.engine")

            logger.info(f"Loading engine from: {engine_path}")

            self._runner = ModelRunnerCpp.from_dir(
                engine_dir=self.config.engine_path,
                rank=0,
            )

            logger.info("TensorRT-LLM ModelRunnerCpp loaded")

        except ImportError:
            # Fallback to Python runner
            logger.warning("ModelRunnerCpp not available, trying Python runner")
            try:
                from tensorrt_llm.runtime import ModelRunner

                self._runner = ModelRunner.from_dir(
                    engine_dir=self.config.engine_path,
                    rank=0,
                )
                logger.info("TensorRT-LLM ModelRunner loaded")
            except Exception as e:
                logger.error(f"Failed to load TensorRT-LLM: {e}")
                raise

        # Warmup
        self._warmup()
        logger.info("Qwen TRT-LLM loaded and ready")

    def _warmup(self):
        """Warmup the engine with a simple generation"""
        try:
            messages = [{"role": "user", "content": "Hi"}]
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            input_ids = self._tokenizer(text, return_tensors="pt").input_ids

            # Convert to list properly
            input_ids_list = input_ids[0].tolist() if hasattr(input_ids[0], 'tolist') else list(input_ids[0])

            # Run warmup generation
            outputs = self._runner.generate(
                batch_input_ids=[input_ids_list],
                max_new_tokens=10,
                end_id=self._tokenizer.eos_token_id,
                pad_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                temperature=1.0,
                top_p=1.0,
                top_k=1,
            )
            logger.info("TRT-LLM warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def _generate_streaming(
        self,
        messages: List[dict],
        output_queue: Queue,
    ):
        """Streaming generation using TensorRT-LLM"""
        self._stop_event.clear()

        try:
            # Apply chat template
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            input_ids = self._tokenizer(text, return_tensors="pt").input_ids
            input_ids_list = input_ids[0].tolist()

            # TensorRT-LLM streaming generation
            # Use streaming callback if available
            if hasattr(self._runner, 'generate_async'):
                # Async streaming generation
                async def stream_tokens():
                    async for output in self._runner.generate_async(
                        batch_input_ids=[input_ids_list],
                        max_new_tokens=self.config.max_new_tokens,
                        end_id=self._tokenizer.eos_token_id,
                        pad_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                        temperature=self.config.temperature if self.config.do_sample else 1.0,
                        top_p=self.config.top_p if self.config.do_sample else 1.0,
                        top_k=self.config.top_k if self.config.do_sample else 1,
                        streaming=True,
                    ):
                        if self._stop_event.is_set():
                            break
                        # Decode new tokens
                        new_tokens = output[0][len(input_ids_list):]
                        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                        output_queue.put(text)

                # Run in event loop
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(stream_tokens())
                loop.close()

            else:
                # Non-streaming fallback - generate all at once then simulate streaming
                outputs = self._runner.generate(
                    batch_input_ids=[input_ids_list],
                    max_new_tokens=self.config.max_new_tokens,
                    end_id=self._tokenizer.eos_token_id,
                    pad_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                    temperature=self.config.temperature if self.config.do_sample else 1.0,
                    top_p=self.config.top_p if self.config.do_sample else 1.0,
                    top_k=self.config.top_k if self.config.do_sample else 1,
                    repetition_penalty=self.config.repetition_penalty,
                )

                # Get output tokens
                output_ids = outputs[0][0] if isinstance(outputs[0], list) else outputs[0]
                if hasattr(output_ids, 'tolist'):
                    output_ids = output_ids.tolist()

                # Decode only new tokens
                new_tokens = output_ids[len(input_ids_list):]
                full_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Simulate streaming by yielding words
                words = full_text.split(' ')
                for i, word in enumerate(words):
                    if self._stop_event.is_set():
                        break
                    token = word if i == 0 else ' ' + word
                    output_queue.put(token)

        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()

        output_queue.put(None)  # Signal completion

    async def generate_stream(
        self,
        messages: List[dict],
    ) -> AsyncIterator[str]:
        """
        Async streaming generation
        Yields tokens as they're generated
        """
        if self._runner is None:
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
        if self._runner is not None:
            del self._runner
            self._runner = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None


# Alias for compatibility
QwenLLM = QwenTRTLLM


class StreamingLLM:
    """
    High-level streaming LLM interface
    Manages conversation history and streams sentences
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._llm = QwenTRTLLM(self.config)
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
