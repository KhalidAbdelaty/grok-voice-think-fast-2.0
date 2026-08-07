"""WebSocket client for the Grok Speech to Speech API.

Handles the connection, session config, audio in/out, and cost tracking.
Microphone and speaker I/O are left as callables so this module runs the
same way in a terminal test, a FastAPI backend, or a real audio pipeline.
"""
import base64
import json

import websockets

from config import REALTIME_URL, REALTIME_URL_BASE, SAMPLE_RATE, XAI_API_KEY


class CostTracker:
    """Tracks the two billing meters: audio minutes and text-input events."""

    def __init__(self):
        self.audio_seconds = 0.0
        self.text_events = 0

    def add_audio_bytes(self, num_bytes: int, sample_rate: int = SAMPLE_RATE):
        # 16-bit mono PCM: 2 bytes per sample.
        self.audio_seconds += num_bytes / 2 / sample_rate

    @property
    def audio_usd(self) -> float:
        return self.audio_seconds / 60 * 0.08  # grok-voice-think-fast-2.0 rate

    @property
    def text_usd(self) -> float:
        return self.text_events * 0.004

    @property
    def total_usd(self) -> float:
        return self.audio_usd + self.text_usd


class VoiceClient:
    def __init__(self, resume_conversation_id: str | None = None, model: str | None = None):
        self.ws = None
        self.conversation_id = resume_conversation_id
        self.model = model
        self.cost = CostTracker()

    async def connect(self, resumption: bool = False):
        if not XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY is not set. Copy .env.example to .env "
                               "and paste your key in.")
        url = REALTIME_URL if not self.model else f"{REALTIME_URL_BASE}?model={self.model}"
        if self.conversation_id:
            url += f"&conversation_id={self.conversation_id}"
        self.ws = await websockets.connect(
            url, additional_headers={"Authorization": f"Bearer {XAI_API_KEY}"}
        )
        if resumption:
            await self.send({"type": "session.update", "session": {"resumption": {"enabled": True}}})
        return self.ws

    async def send(self, event: dict):
        # Anything other than a bare audio item counts as a text-input event
        # for billing, except function_call_output and response.create.
        item = event.get("item", {})
        if event.get("type") == "conversation.item.create" and item.get("type") not in (
            "function_call_output",
            "input_audio",
            "audio",
        ):
            self.cost.text_events += 1
        await self.ws.send(json.dumps(event))

    async def configure_session(self, config: dict):
        await self.send({"type": "session.update", "session": config})

    async def send_audio_chunk(self, pcm_bytes: bytes):
        self.cost.add_audio_bytes(len(pcm_bytes))
        await self.send(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm_bytes).decode()}
        )

    async def commit_audio(self):
        """Close the input buffer by hand. Only needed when turn_detection is
        null; with server VAD the server commits the turn for you."""
        await self.send({"type": "input_audio_buffer.commit"})

    async def send_text(self, text: str):
        await self.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def request_response(self):
        await self.send({"type": "response.create"})

    async def send_tool_result(self, call_id: str, output: dict):
        await self.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output),
                },
            }
        )

    async def events(self):
        async for message in self.ws:
            event = json.loads(message)
            if event["type"] == "conversation.created":
                self.conversation_id = event["conversation"]["id"]
            if event["type"] == "response.output_audio.delta":
                self.cost.add_audio_bytes(len(base64.b64decode(event["delta"])))
            yield event

    async def close(self):
        """Close the socket, and only ever from the loop that opened it.

        A WebSocket is bound to its event loop. Closing it from a second loop
        (the shape you get from calling asyncio.run twice) raises instead of
        closing, which then reads as a failed turn even though the turn
        worked. Closing is also what ends the session server-side: drop the
        loop without it and the session lingers against the concurrency cap.
        """
        if self.ws is None:
            return
        ws, self.ws = self.ws, None
        try:
            await ws.close()
        except Exception:  # noqa: BLE001 - already gone is a fine outcome
            pass
