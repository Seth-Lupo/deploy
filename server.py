"""
WebSocket Server - Voice Pipeline API with TensorRT
Real-time bidirectional audio streaming with text and metrics
Optimized for A100/A10G/L4 GPUs with TensorRT acceleration
"""

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional, Set, Dict, Any
from contextlib import asynccontextmanager
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pipeline import VoicePipeline, PipelineConfig, PipelineState, PipelineMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration for TensorRT pipeline"""
    host: str = "0.0.0.0"
    port: int = 8765
    # Pipeline config
    system_prompt: Optional[str] = None
    voice_reference: Optional[str] = None
    # Model base path
    model_base_path: str = "/workspace/models"
    # ASR config - Parakeet TRT
    asr_model_path: str = "/workspace/models/parakeet-tdt-0.6b-v2"
    # LLM config - Qwen TRT-LLM
    llm_engine_path: str = "/workspace/models/qwen3-4b-int8wo-engine"
    llm_tokenizer_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_temperature: float = 0.7
    # TTS config - Chatterbox TRT
    tts_model_path: str = "/workspace/models/chatterbox-turbo"
    tts_exaggeration: float = 0.5


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast to all connections"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        for ws in disconnected:
            self.active_connections.discard(ws)


class VoiceSession:
    """
    Voice session for a single WebSocket connection
    Handles bidirectional audio streaming with TensorRT acceleration
    """

    def __init__(self, websocket: WebSocket, config: ServerConfig):
        self.websocket = websocket
        self.config = config
        self._pipeline: Optional[VoicePipeline] = None
        self._running = False
        self._audio_sender_task: Optional[asyncio.Task] = None
        self._response_task: Optional[asyncio.Task] = None
        self._first_audio_sent = False

    async def start(self):
        """Initialize and start the session"""
        # Create pipeline config with TRT paths
        pipeline_config = PipelineConfig(
            model_base_path=self.config.model_base_path,
            asr_model_path=self.config.asr_model_path,
            llm_engine_path=self.config.llm_engine_path,
            llm_tokenizer_path=self.config.llm_tokenizer_path,
            llm_temperature=self.config.llm_temperature,
            tts_model_path=self.config.tts_model_path,
            tts_exaggeration=self.config.tts_exaggeration,
        )

        if self.config.system_prompt:
            pipeline_config.system_prompt = self.config.system_prompt

        if self.config.voice_reference:
            pipeline_config.tts_voice_reference = self.config.voice_reference

        # Initialize pipeline
        self._pipeline = VoicePipeline(pipeline_config)

        # Set callbacks
        self._pipeline.set_callbacks(
            on_transcription=self._handle_transcription,
            on_llm_sentence=self._handle_llm_sentence,
            on_state_change=self._handle_state_change,
            on_metrics=self._handle_metrics,
        )

        await self._pipeline.initialize()
        self._running = True

        # Send ready message
        await self._send_message("ready", {
            "asr_sample_rate": self._pipeline.asr_sample_rate,
            "tts_sample_rate": self._pipeline.tts_sample_rate,
        })

        # Start audio sender task
        self._audio_sender_task = asyncio.create_task(self._audio_sender())

    async def _send_message(self, msg_type: str, data: dict):
        """Send a message to the client"""
        try:
            await self.websocket.send_json({
                "type": msg_type,
                "data": data,
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"Send error: {e}")

    async def _send_audio(self, audio: np.ndarray):
        """Send audio chunk to client"""
        try:
            # Convert to bytes (16-bit PCM)
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            await self.websocket.send_json({
                "type": "audio_out",
                "data": {
                    "audio": audio_b64,
                    "sample_rate": self._pipeline.tts_sample_rate,
                    "format": "pcm_s16le"
                },
                "timestamp": time.time()
            })
        except Exception as e:
            logger.error(f"Audio send error: {e}")

    def _handle_transcription(self, text: str, latency: float):
        """Handle transcription callback"""
        asyncio.create_task(self._send_message("transcription", {
            "text": text,
            "latency_ms": latency * 1000
        }))

    def _handle_llm_sentence(self, sentence: str, latency: float):
        """Handle LLM sentence callback"""
        asyncio.create_task(self._send_message("llm_sentence", {
            "text": sentence,
            "latency_ms": latency
        }))

    def _handle_state_change(self, state: PipelineState):
        """Handle state change callback"""
        asyncio.create_task(self._send_message("state", {
            "state": state.name
        }))

    def _handle_metrics(self, metrics: PipelineMetrics):
        """Handle metrics callback"""
        metrics_dict = {
            "llm_first_token_ms": metrics.llm_first_token_ms,
            "llm_first_sentence_ms": metrics.llm_first_sentence_ms,
            "tts_first_chunk_ms": metrics.tts_first_chunk_ms,
            "first_audio_sent_ms": metrics.first_audio_sent_ms,
            "total_response_ms": metrics.total_response_ms,
            "llm_tokens": metrics.llm_tokens,
            "sentences_generated": metrics.sentences_generated,
            "audio_chunks_sent": metrics.audio_chunks_sent,
            "total_audio_duration_ms": metrics.total_audio_duration_ms,
            "llm_tokens_per_sec": metrics.llm_tokens_per_sec,
            "tts_rtf": metrics.tts_rtf,
        }
        asyncio.create_task(self._send_message("metrics", metrics_dict))

    async def _audio_sender(self):
        """Background task to send audio chunks"""
        while self._running:
            try:
                chunk = await self._pipeline.get_audio_chunk()

                if chunk is None:
                    # End of response - reset first audio flag for next response
                    self._first_audio_sent = False
                    await self._send_message("audio_end", {})
                    continue

                if len(chunk) > 0:
                    # Mark first audio sent time in pipeline metrics
                    if not self._first_audio_sent:
                        self._pipeline.mark_first_audio_sent()
                        self._first_audio_sent = True

                    await self._send_audio(chunk)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio sender error: {e}")
                await asyncio.sleep(0.01)

    async def handle_message(self, message: dict):
        """Handle incoming WebSocket message"""
        msg_type = message.get("type")
        data = message.get("data", {})

        if msg_type == "audio_in":
            await self._handle_audio_in(data)
        elif msg_type == "text_in":
            await self._handle_text_in(data)
        elif msg_type == "interrupt":
            await self._handle_interrupt()
        elif msg_type == "reset":
            await self._handle_reset()
        elif msg_type == "set_system_prompt":
            self._handle_set_system_prompt(data)
        elif msg_type == "set_voice":
            self._handle_set_voice(data)
        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_audio_in(self, data: dict):
        """Handle incoming audio"""
        audio_b64 = data.get("audio")
        if not audio_b64:
            return

        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Process through pipeline
        transcription = await self._pipeline.process_audio(audio_float)

        if transcription:
            # Reset first audio flag for new response
            self._first_audio_sent = False

            # Start response generation
            if self._response_task and not self._response_task.done():
                await self._pipeline.interrupt()

            self._response_task = asyncio.create_task(
                self._pipeline.respond(transcription)
            )

    async def _handle_text_in(self, data: dict):
        """Handle text input (bypass ASR)"""
        text = data.get("text")
        if not text:
            return

        # Send as transcription
        await self._send_message("transcription", {
            "text": text,
            "latency_ms": 0
        })

        # Reset first audio flag for new response
        self._first_audio_sent = False

        # Start response
        if self._response_task and not self._response_task.done():
            await self._pipeline.interrupt()

        self._response_task = asyncio.create_task(
            self._pipeline.respond(text)
        )

    async def _handle_interrupt(self):
        """Handle interrupt command"""
        await self._pipeline.interrupt()
        await self._send_message("interrupted", {})

    async def _handle_reset(self):
        """Handle reset command"""
        self._pipeline.reset_conversation()
        await self._send_message("reset", {})

    def _handle_set_system_prompt(self, data: dict):
        """Handle system prompt update"""
        prompt = data.get("prompt")
        if prompt:
            self._pipeline.set_system_prompt(prompt)

    def _handle_set_voice(self, data: dict):
        """Handle voice reference update"""
        path = data.get("path")
        if path:
            self._pipeline.set_voice(path)

    async def stop(self):
        """Stop the session"""
        self._running = False

        if self._audio_sender_task:
            self._audio_sender_task.cancel()
            try:
                await self._audio_sender_task
            except asyncio.CancelledError:
                pass

        if self._response_task:
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

        if self._pipeline:
            await self._pipeline.shutdown()


# Global state
manager = ConnectionManager()
server_config = ServerConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("TensorRT Voice Pipeline Server starting...")
    yield
    logger.info("Voice pipeline server shutting down...")


app = FastAPI(
    title="Voice Pipeline Server (TensorRT)",
    description="Real-time voice conversation API with TensorRT acceleration",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "voice-pipeline-trt",
        "connections": len(manager.active_connections)
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "connections": len(manager.active_connections),
        "config": {
            "asr_model_path": server_config.asr_model_path,
            "llm_engine_path": server_config.llm_engine_path,
            "tts_model_path": server_config.tts_model_path,
            "has_voice_reference": server_config.voice_reference is not None,
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for voice conversations"""
    await manager.connect(websocket)
    session = VoiceSession(websocket, server_config)

    try:
        await session.start()

        while True:
            try:
                message = await websocket.receive_json()
                await session.handle_message(message)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")
                continue

    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await session.stop()
        manager.disconnect(websocket)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Voice Pipeline Server (TensorRT)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--system-prompt", default=None, help="System prompt")
    parser.add_argument("--voice-reference", default=None, help="Voice reference audio file")
    # Model paths
    parser.add_argument("--model-base-path", default="/workspace/models",
                       help="Base path for models")
    # ASR
    parser.add_argument("--asr-model-path", default="/workspace/models/parakeet-tdt-0.6b-v2",
                       help="Path to Parakeet TRT model")
    # LLM
    parser.add_argument("--llm-engine-path", default="/workspace/models/qwen3-4b-int8wo-engine",
                       help="Path to TensorRT-LLM engine")
    parser.add_argument("--llm-tokenizer-path", default="Qwen/Qwen3-4B-Instruct-2507",
                       help="HuggingFace tokenizer path")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                       help="LLM temperature")
    # TTS
    parser.add_argument("--tts-model-path", default="/workspace/models/chatterbox-turbo",
                       help="Path to Chatterbox TRT model")
    parser.add_argument("--tts-exaggeration", type=float, default=0.5,
                       help="TTS emotion exaggeration")

    args = parser.parse_args()

    global server_config
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        system_prompt=args.system_prompt,
        voice_reference=args.voice_reference,
        model_base_path=args.model_base_path,
        asr_model_path=args.asr_model_path,
        llm_engine_path=args.llm_engine_path,
        llm_tokenizer_path=args.llm_tokenizer_path,
        llm_temperature=args.llm_temperature,
        tts_model_path=args.tts_model_path,
        tts_exaggeration=args.tts_exaggeration,
    )

    logger.info(f"Starting TensorRT server on {args.host}:{args.port}")
    logger.info(f"ASR: {args.asr_model_path}")
    logger.info(f"LLM: {args.llm_engine_path}")
    logger.info(f"TTS: {args.tts_model_path}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
