"""Session config, system prompt, and the event loop for the support agent.

This is the piece that turns a bare WebSocket connection into an agent:
it configures the session, plays audio deltas in order, runs tools, waits
for playback before continuing after a tool result, and clears the local
queue the moment the caller barges in.
"""
import asyncio
import base64
import json

from tools import ORDER_TOOLS, execute
from voice_client import VoiceClient

SYSTEM_PROMPT = """
You are a voice support agent for an online store.

How to talk: short sentences, one question at a time, the way a person
speaks on the phone. No lists, no headings, no reading out fields the caller
did not ask for. If a lookup will take a second, say so in a few words.

Finding the order: you need an order number before you touch anything. If
the caller does not have it, ask for the email or phone on the order and use
find_orders. If several come back, read out the numbers with their status
and let the caller pick.

Before you change anything: read back the order number and exactly what you
are about to do, then wait for a clear yes. This applies to changing
delivery instructions, cancelling, and opening a ticket. Never assume the
caller meant yes because they kept talking.

When a tool refuses: it tells you why and what to offer instead. Explain the
reason in one sentence and offer the alternative. Do not apologise twice, do
not repeat the refusal, and never suggest something you have no tool for.

Hand over to a person with transfer_to_human for refunds beyond a plain
cancellation, payment or billing problems, account access, anything you are
unsure about, or a caller who is annoyed or asks for a human. Say you are
transferring them, then stop talking.

Never invent: no delivery dates, no refund timelines, no policies, and no
promises about when someone will call back. If you do not know, say you do
not know and offer a ticket or a transfer.
""".strip()


# 100 ms of 24 kHz mono PCM16 per append. Small enough that the server starts
# working before the caller stops talking, big enough to avoid a flood of frames.
AUDIO_CHUNK_BYTES = 4800


def build_session_config(
    voice="eve",
    turn_detection=None,
    instructions=SYSTEM_PROMPT,
    language_hint=None,
    speed=1.0,
):
    # Naming the transcription model is what turns on the cumulative
    # `conversation.item.input_audio_transcription.updated` events. Without
    # it you only get the final event, and a long sentence can land in the
    # log half-written even though the model heard all of it.
    transcription = {"model": "grok-transcribe"}
    if language_hint:
        transcription["language_hint"] = language_hint
    audio_input = {
        "format": {"type": "audio/pcm", "rate": 24000},
        "transcription": transcription,
    }
    return {
        "voice": voice,
        "instructions": instructions,
        "turn_detection": turn_detection,
        "tools": ORDER_TOOLS,
        "resumption": {"enabled": True},
        "audio": {
            "input": audio_input,
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "speed": speed},
        },
    }


class PlaybackQueue:
    """Tracks how much queued audio is still 'playing' so the client knows
    when it is safe to send response.create after a tool result, and lets
    a barge-in clear whatever is left instantly."""

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.queued_seconds = 0.0

    def push(self, pcm_bytes: bytes):
        self.queued_seconds += len(pcm_bytes) / 2 / self.sample_rate

    async def wait_until_drained(self):
        # In a real app this polls the actual speaker buffer. Here we just
        # wait out however many seconds of audio are still queued.
        await asyncio.sleep(self.queued_seconds)
        self.queued_seconds = 0.0

    def clear(self):
        self.queued_seconds = 0.0


class SupportAgent:
    def __init__(self, resume_conversation_id=None, session_config=None, model=None):
        self.client = VoiceClient(resume_conversation_id, model=model)
        self.playback = PlaybackQueue()
        self.transcript = []
        self.tool_calls = []      # (name, arguments, result) per call, for the UI
        self.event_log = []       # every event type in arrival order
        self.session_model = None  # what the server says we actually connected to
        self.session_config = session_config
        self.on_audio_chunk = None  # optional callback(bytes) for real playback

    async def run_turn(
        self,
        user_text: str | None = None,
        manual_turns: bool = True,
        user_audio: bytes | None = None,
    ):
        """Send one user turn, as text or as PCM16 audio, and drive the event
        loop until the agent finishes speaking, tool calls included."""
        resuming = self.client.conversation_id is not None
        # Send resumption.enabled in the one and only session.update below.
        # A second, separate session.update right after the first appears to
        # race the server's replay of cached turns on a resumed connection.
        await self.client.connect(resumption=False)
        turn_detection = None if manual_turns else {"type": "server_vad"}
        config = self.session_config or build_session_config()
        config["turn_detection"] = turn_detection
        await self.client.configure_session(config)

        try:
            await self._drive_turn(user_text, user_audio, resuming)
        finally:
            # Close here, on the loop that opened the socket. Returning with
            # it open leaves the session alive server-side, and the team only
            # gets 10 concurrent ones.
            await self.client.close()

        return self.client.conversation_id

    async def _drive_turn(self, user_text, user_audio, resuming):
        pending_calls = 0
        turn_sent = False
        async for event in self.client.events():
            etype = event["type"]
            self.event_log.append(etype)

            if etype == "session.created":
                # An unrecognized model string falls back silently, so read
                # back what the server actually gave us.
                self.session_model = event.get("session", {}).get("model")

            elif etype == "session.updated" and not turn_sent:
                # On a resumed connection, the cached turns replay as
                # conversation.item.added events around this point. Give
                # them a moment to land before asking a question that
                # depends on them, or the model answers with no memory
                # of the earlier turns.
                if resuming:
                    await asyncio.sleep(1.0)
                if user_audio:
                    # Stream the caller's audio in chunks, then close the turn
                    # by hand because this session runs without server VAD.
                    for i in range(0, len(user_audio), AUDIO_CHUNK_BYTES):
                        await self.client.send_audio_chunk(
                            user_audio[i:i + AUDIO_CHUNK_BYTES]
                        )
                    await self.client.commit_audio()
                else:
                    await self.client.send_text(user_text)
                await self.client.request_response()
                turn_sent = True

            elif etype == "input_audio_buffer.speech_started":
                # Caller barged in: drop whatever is still queued locally.
                self.playback.clear()

            elif etype == "response.output_audio.delta":
                chunk = base64.b64decode(event["delta"])
                self.playback.push(chunk)
                if self.on_audio_chunk:
                    self.on_audio_chunk(chunk)

            elif etype == "response.output_audio_transcript.done":
                self.transcript.append(("assistant", event.get("transcript", "")))

            elif etype.startswith("conversation.item.input_audio_transcription."):
                # What the model heard the caller say. Worth showing next to
                # the reply: most "the agent ignored me" reports are really
                # transcription misses.
                text = event.get("transcript", "")
                if etype.endswith(".completed") and text:
                    self.transcript.append(("caller", text))

            elif etype == "response.function_call_arguments.done":
                pending_calls += 1
                name = event["name"]
                call_id = event["call_id"]
                args = json.loads(event["arguments"])
                result = execute(name, args)
                self.tool_calls.append((name, args, result))
                await self.client.send_tool_result(call_id, result)
                pending_calls -= 1

            elif etype == "response.done":
                if pending_calls == 0 and self._had_function_call(event):
                    # All outputs are in. Wait for the current turn's audio
                    # to finish before asking for the follow-up, or the two
                    # responses overlap.
                    await self.playback.wait_until_drained()
                    await self.client.request_response()
                    continue
                break

            elif etype == "error":
                self.transcript.append(("error", event.get("error", {}).get("message", "")))
                break

    @staticmethod
    def _had_function_call(done_event: dict) -> bool:
        item = done_event.get("response", {}).get("output", [])
        return any(part.get("type") == "function_call" for part in item)

    async def close(self):
        await self.client.close()
