#!/usr/bin/env python3
"""
Voice Client - Connect to Voice Pipeline Server
Real-time audio conversation with text display and metrics
"""

import asyncio
import base64
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional
from collections import deque
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Client configuration"""
    server_url: str = "ws://localhost:8765/ws"
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    chunk_duration_ms: int = 100
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    show_metrics: bool = True


class AudioInput:
    """Microphone input handler"""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        device: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.device = device
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stream: Optional[sd.InputStream] = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input"""
        if status:
            logger.warning(f"Input status: {status}")
        # Put audio in queue (copy to avoid buffer reuse issues)
        audio = indata[:, 0].copy()
        try:
            self._queue.put_nowait(audio)
        except asyncio.QueueFull:
            pass  # Drop if queue full

    async def start(self):
        """Start audio input"""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self._audio_callback
        )
        self._stream.start()
        logger.info(f"Audio input started (device: {self.device or 'default'})")

    async def get_chunk(self) -> Optional[np.ndarray]:
        """Get next audio chunk"""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            return None

    async def stop(self):
        """Stop audio input"""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class AudioOutput:
    """Speaker output handler"""

    def __init__(
        self,
        sample_rate: int = 24000,
        device: Optional[int] = None,
        buffer_size: int = 20
    ):
        self.sample_rate = sample_rate
        self.device = device
        self._chunk_queue: deque = deque(maxlen=buffer_size)
        self._buffer = np.array([], dtype=np.float32)  # Continuous buffer
        self._stream: Optional[sd.OutputStream] = None
        self._running = False

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio output"""
        if status:
            logger.warning(f"Output status: {status}")

        # Fill buffer from queue if needed
        while len(self._buffer) < frames and self._chunk_queue:
            chunk = self._chunk_queue.popleft()
            self._buffer = np.concatenate([self._buffer, chunk])

        # Output from buffer
        if len(self._buffer) >= frames:
            outdata[:, 0] = self._buffer[:frames]
            self._buffer = self._buffer[frames:]
        elif len(self._buffer) > 0:
            outdata[:len(self._buffer), 0] = self._buffer
            outdata[len(self._buffer):, 0] = 0
            self._buffer = np.array([], dtype=np.float32)
        else:
            outdata.fill(0)

    async def start(self):
        """Start audio output"""
        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            device=self.device,
            callback=self._audio_callback
        )
        self._stream.start()
        logger.info(f"Audio output started (device: {self.device or 'default'})")

    def play(self, audio: np.ndarray):
        """Queue audio for playback"""
        self._chunk_queue.append(audio)

    async def stop(self):
        """Stop audio output"""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def clear(self):
        """Clear audio buffer"""
        self._chunk_queue.clear()
        self._buffer = np.array([], dtype=np.float32)


class VoiceClient:
    """
    Voice client for connecting to pipeline server
    Handles bidirectional audio streaming
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        self.config = config or ClientConfig()
        self._ws = None
        self._running = False
        self._audio_in: Optional[AudioInput] = None
        self._audio_out: Optional[AudioOutput] = None
        self._state = "disconnected"

        # Tasks
        self._send_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None

    async def connect(self):
        """Connect to the server"""
        logger.info(f"Connecting to {self.config.server_url}...")

        try:
            self._ws = await websockets.connect(
                self.config.server_url,
                ping_interval=20,
                ping_timeout=30
            )
            logger.info("Connected to server")
            self._state = "connected"

            # Wait for ready message (pipeline init can take 2+ minutes on first run)
            ready_msg = await asyncio.wait_for(self._ws.recv(), timeout=180)
            ready_data = json.loads(ready_msg)

            if ready_data.get("type") == "ready":
                data = ready_data.get("data", {})
                self.config.input_sample_rate = data.get("asr_sample_rate", 16000)
                self.config.output_sample_rate = data.get("tts_sample_rate", 24000)
                logger.info(f"Server ready. ASR: {self.config.input_sample_rate}Hz, TTS: {self.config.output_sample_rate}Hz")

            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def start(self):
        """Start the voice conversation"""
        if not self._ws:
            if not await self.connect():
                return

        self._running = True

        # Initialize audio
        self._audio_in = AudioInput(
            sample_rate=self.config.input_sample_rate,
            chunk_ms=self.config.chunk_duration_ms,
            device=self.config.input_device
        )
        self._audio_out = AudioOutput(
            sample_rate=self.config.output_sample_rate,
            device=self.config.output_device
        )

        await self._audio_in.start()
        await self._audio_out.start()

        # Start tasks
        self._send_task = asyncio.create_task(self._send_loop())
        self._receive_task = asyncio.create_task(self._receive_loop())

        print("\n" + "="*50)
        print("Voice conversation started!")
        print("Speak into your microphone...")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")

        try:
            await asyncio.gather(self._send_task, self._receive_task)
        except asyncio.CancelledError:
            pass

    async def _send_loop(self):
        """Send audio to server"""
        while self._running:
            try:
                audio = await self._audio_in.get_chunk()
                if audio is None:
                    continue

                # Convert to bytes
                audio_int16 = (audio * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

                # Send to server
                await self._ws.send(json.dumps({
                    "type": "audio_in",
                    "data": {
                        "audio": audio_b64,
                        "sample_rate": self.config.input_sample_rate
                    }
                }))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send error: {e}")
                await asyncio.sleep(0.1)

    async def _receive_loop(self):
        """Receive messages from server"""
        while self._running:
            try:
                message = await self._ws.recv()
                data = json.loads(message)
                await self._handle_message(data)

            except asyncio.CancelledError:
                break
            except websockets.exceptions.ConnectionClosed:
                logger.error("Connection closed")
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")

    async def _handle_message(self, message: dict):
        """Handle incoming message"""
        msg_type = message.get("type")
        data = message.get("data", {})

        if msg_type == "audio_out":
            await self._handle_audio(data)
        elif msg_type == "transcription":
            self._handle_transcription(data)
        elif msg_type == "llm_sentence":
            self._handle_llm_sentence(data)
        elif msg_type == "state":
            self._handle_state(data)
        elif msg_type == "metrics":
            self._handle_metrics(data)
        elif msg_type == "audio_end":
            pass  # End of response audio
        elif msg_type == "interrupted":
            print("\n[Interrupted]")
        elif msg_type == "error":
            logger.error(f"Server error: {data}")

    async def _handle_audio(self, data: dict):
        """Handle audio output"""
        audio_b64 = data.get("audio")
        if not audio_b64:
            return

        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Play
        self._audio_out.play(audio_float)

    def _handle_transcription(self, data: dict):
        """Handle transcription"""
        text = data.get("text", "")
        latency = data.get("latency_ms", 0)
        print(f"\n\033[94m[You]\033[0m {text}")
        if self.config.show_metrics and latency > 0:
            print(f"  \033[90m(ASR: {latency:.0f}ms)\033[0m")

    def _handle_llm_sentence(self, data: dict):
        """Handle LLM sentence"""
        text = data.get("text", "")
        latency = data.get("latency_ms", 0)
        print(f"\033[92m[Assistant]\033[0m {text}")
        if self.config.show_metrics:
            print(f"  \033[90m(LLM: {latency:.0f}ms)\033[0m")

    def _handle_state(self, data: dict):
        """Handle state change"""
        state = data.get("state", "")
        self._state = state
        if state == "LISTENING":
            print("\033[93m[Listening...]\033[0m")
        elif state == "PROCESSING":
            print("\033[93m[Processing...]\033[0m")
        elif state == "SPEAKING":
            pass  # Don't print for speaking

    def _handle_metrics(self, data: dict):
        """Handle metrics"""
        if self.config.show_metrics:
            print(f"\033[90m[Metrics] First sentence: {data.get('llm_first_sentence', 0)*1000:.0f}ms, "
                  f"First audio: {data.get('tts_first_chunk', 0)*1000:.0f}ms, "
                  f"Total: {data.get('total_latency', 0)*1000:.0f}ms\033[0m")

    async def send_text(self, text: str):
        """Send text instead of audio (bypass ASR)"""
        if self._ws:
            await self._ws.send(json.dumps({
                "type": "text_in",
                "data": {"text": text}
            }))

    async def interrupt(self):
        """Interrupt current response"""
        if self._ws:
            await self._ws.send(json.dumps({"type": "interrupt"}))
            self._audio_out.clear()

    async def reset(self):
        """Reset conversation"""
        if self._ws:
            await self._ws.send(json.dumps({"type": "reset"}))

    async def stop(self):
        """Stop the client"""
        self._running = False

        if self._send_task:
            self._send_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()

        if self._audio_in:
            await self._audio_in.stop()
        if self._audio_out:
            await self._audio_out.stop()

        if self._ws:
            await self._ws.close()


def list_devices():
    """List available audio devices"""
    print("\nAvailable Audio Devices:")
    print("=" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        in_ch = device['max_input_channels']
        out_ch = device['max_output_channels']
        default = ""
        if i == sd.default.device[0]:
            default += " [DEFAULT INPUT]"
        if i == sd.default.device[1]:
            default += " [DEFAULT OUTPUT]"
        if in_ch > 0 or out_ch > 0:
            print(f"  {i}: {device['name']}")
            print(f"      In: {in_ch} ch, Out: {out_ch} ch{default}")
    print()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Voice Pipeline Client")
    parser.add_argument("--server", default="ws://localhost:8765/ws",
                       help="Server WebSocket URL")
    parser.add_argument("--input-device", type=int, default=None,
                       help="Input audio device ID")
    parser.add_argument("--output-device", type=int, default=None,
                       help="Output audio device ID")
    parser.add_argument("--list-devices", action="store_true",
                       help="List available audio devices")
    parser.add_argument("--no-metrics", action="store_true",
                       help="Hide timing metrics")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    config = ClientConfig(
        server_url=args.server,
        input_device=args.input_device,
        output_device=args.output_device,
        show_metrics=not args.no_metrics,
    )

    client = VoiceClient(config)

    try:
        await client.start()
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        await client.stop()
        print("Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
