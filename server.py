"""
WebSocket Server - Voice Pipeline API
Real-time bidirectional audio streaming with text and metrics
Optimized for A100/A10G GPUs with FP16 models
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
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8765
    # Pipeline config
    system_prompt: Optional[str] = None
    voice_reference: Optional[str] = None
    # ASR config
    asr_model: str = "istupakov/parakeet-tdt-0.6b-v2-onnx"
    asr_quantization: str = "fp16"
    # LLM config
    llm_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    llm_dtype: str = "float16"
    llm_temperature: float = 0.7
    # TTS config
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
    Handles bidirectional audio streaming
    """

    def __init__(self, websocket: WebSocket, config: ServerConfig):
        self.websocket = websocket
        self.config = config
        self._pipeline: Optional[VoicePipeline] = None
        self._running = False
        self._audio_sender_task: Optional[asyncio.Task] = None
        self._response_task: Optional[asyncio.Task] = None

    async def start(self):
        """Initialize and start the session"""
        # Create pipeline config
        pipeline_config = PipelineConfig(
            asr_model=self.config.asr_model,
            asr_quantization=self.config.asr_quantization,
            llm_model_id=self.config.llm_model_id,
            llm_dtype=self.config.llm_dtype,
            llm_temperature=self.config.llm_temperature,
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
            "latency_ms": latency * 1000
        }))

    def _handle_state_change(self, state: PipelineState):
        """Handle state change callback"""
        asyncio.create_task(self._send_message("state", {
            "state": state.name
        }))

    def _handle_metrics(self, metrics: PipelineMetrics):
        """Handle metrics callback"""
        asyncio.create_task(self._send_message("metrics", asdict(metrics)))

    async def _audio_sender(self):
        """Background task to send audio chunks"""
        while self._running:
            try:
                chunk = await self._pipeline.get_audio_chunk()

                if chunk is None:
                    # End of response
                    logger.info("Audio sender: got None, sending audio_end")
                    await self._send_message("audio_end", {})
                    continue

                if len(chunk) > 0:
                    logger.info(f"Audio sender: sending chunk len={len(chunk)}")
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
    logger.info("Voice pipeline server starting...")
    yield
    logger.info("Voice pipeline server shutting down...")


app = FastAPI(
    title="Voice Pipeline Server",
    description="Real-time voice conversation API with FP16 models",
    version="2.0.0",
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
        "service": "voice-pipeline",
        "connections": len(manager.active_connections)
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "connections": len(manager.active_connections),
        "config": {
            "asr_model": server_config.asr_model,
            "llm_model": server_config.llm_model_id,
            "llm_dtype": server_config.llm_dtype,
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

    parser = argparse.ArgumentParser(description="Voice Pipeline Server (FP16)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--system-prompt", default=None, help="System prompt")
    parser.add_argument("--voice-reference", default=None, help="Voice reference audio file")
    # ASR
    parser.add_argument("--asr-model", default="istupakov/parakeet-tdt-0.6b-v2-onnx",
                       help="ASR model ID")
    parser.add_argument("--asr-quantization", default="fp16",
                       choices=["fp16", "fp32", "int8"], help="ASR quantization")
    # LLM
    parser.add_argument("--llm-model-id", default="Qwen/Qwen3-4B-Instruct-2507",
                       help="LLM model ID")
    parser.add_argument("--llm-dtype", default="float16",
                       choices=["float16", "bfloat16", "float32"], help="LLM dtype")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                       help="LLM temperature")
    # TTS
    parser.add_argument("--tts-exaggeration", type=float, default=0.5,
                       help="TTS emotion exaggeration")

    args = parser.parse_args()

    global server_config
    server_config = ServerConfig(
        host=args.host,
        port=args.port,
        system_prompt=args.system_prompt,
        voice_reference=args.voice_reference,
        asr_model=args.asr_model,
        asr_quantization=args.asr_quantization,
        llm_model_id=args.llm_model_id,
        llm_dtype=args.llm_dtype,
        llm_temperature=args.llm_temperature,
        tts_exaggeration=args.tts_exaggeration,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"ASR: {args.asr_model} ({args.asr_quantization})")
    logger.info(f"LLM: {args.llm_model_id} ({args.llm_dtype})")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
