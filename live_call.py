"""One live call over one WebSocket, driven from a background thread.

The scripted path in `assistant.py` opens a connection, takes a turn, and
closes it. A live call cannot work that way: the caller talks whenever they
want, so the socket has to stay open and something has to keep reading from
it while Streamlit reruns the page underneath.

So this module owns an event loop on its own thread. Streamlit's thread only
ever pushes microphone audio in and reads a snapshot out, both under a lock.
Everything that touches the WebSocket happens on the call's own loop, which
is also what makes a clean close possible.
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time

from config import SAMPLE_RATE
from tools import execute
from voice_client import VoiceClient

# 100 ms of 24 kHz mono PCM16 per append.
CHUNK_BYTES = 4800

# Mic frames queue up as ~20 ms items (the aiortc default ptime). Capping the
# queue bounds how far behind real time a slow network can push the caller's
# audio; past this, drop the oldest buffered chunk rather than let latency
# grow without limit, since a stale frame is worse than a missing one.
MAX_QUEUED_MIC_CHUNKS = 250  # ~5 s of audio

# Event types that arrive hundreds of times per turn. Counting them beats
# listing them: an event panel with 400 identical rows tells you nothing.
NOISY_EVENTS = {
    "response.output_audio.delta",
    "response.output_audio_transcript.delta",
    "conversation.item.input_audio_transcription.updated",
}


class LiveCall:
    """A single conversation held open across many turns.

    Start it once, push microphone audio for as long as the caller talks, and
    read `snapshot()` on every Streamlit rerun to render the page.
    """

    def __init__(
        self,
        session_config: dict,
        model: str | None = None,
        conversation_id: str | None = None,
        on_output_pcm=None,
        on_barge_in=None,
    ):
        self.session_config = session_config
        self.on_output_pcm = on_output_pcm or (lambda pcm: None)
        self.on_barge_in = on_barge_in or (lambda: None)

        self._client = VoiceClient(conversation_id, model=model)
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._send_queue: asyncio.Queue | None = None
        self._lock = threading.Lock()

        # Everything below is read from Streamlit's thread under _lock.
        self._transcript: list[dict] = []
        self._tool_calls: list[dict] = []
        self._event_counts: list[list] = []  # [direction, type, count]
        self._session_model = None
        self._conversation_id = conversation_id
        self._error = None
        self._handoff = None
        self._speaking = False
        self._thinking = False
        # response.done means the server finished *generating* audio, not
        # that the browser finished *playing* it: PcmAudioSource holds a FIFO
        # that can be seconds deep. This tracks the perf_counter() deadline
        # by which everything pushed so far will have finished playing, so
        # barge-in detection and the echo gate key off actual audible
        # playback instead of the server's generation state.
        self._playback_until = 0.0
        self._reply_started = None
        self._last_reply_latency = None
        # Input transcription arrives in pieces: "updated" carries the
        # cumulative text so far and can correct itself, "completed" is final.
        # Both refer to the same item, so the transcript keeps one line per
        # utterance and rewrites it in place as better text arrives. Appending
        # each event would duplicate the line; keeping only the first would
        # leave it half-written.
        self._caller_lines: dict[str, int] = {}   # item_id -> transcript index
        self._item_text: dict[str, str] = {}      # item_id -> best text so far
        # Voice activity detection splits one spoken sentence into several
        # items when the caller pauses, so consecutive fragments are stitched
        # into the line already on screen instead of starting a new one.
        self._open_caller_line: int | None = None
        self._last_caller_at = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 20.0):
        """Open the socket and block until the session is configured."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("the call did not connect in time")
        if self._error:
            raise RuntimeError(self._error)
        return self

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as exc:  # noqa: BLE001 - surfaced through _error
            self._record_error(f"{type(exc).__name__}: {exc}")
        finally:
            # Release anyone still waiting on start() before the loop dies.
            self._ready.set()
            self._stopped.set()
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001 - nothing useful to do here
                pass

    async def _main(self):
        self._send_queue = asyncio.Queue(maxsize=MAX_QUEUED_MIC_CHUNKS)
        await self._client.connect()
        await self._client.configure_session(self.session_config)
        sender = asyncio.create_task(self._sender())
        try:
            await self._receiver()
        finally:
            sender.cancel()
            # Close on the loop that opened the socket. Closing from another
            # loop is what makes a "successful turn" report as a failure.
            await self._client.close()

    async def _sender(self):
        """Drain microphone audio onto the socket without blocking the mic."""
        while True:
            pcm = await self._send_queue.get()
            if pcm is None:
                return
            try:
                for i in range(0, len(pcm), CHUNK_BYTES):
                    await self._client.send_audio_chunk(pcm[i:i + CHUNK_BYTES])
            except Exception as exc:  # noqa: BLE001 - a dead socket ends the call
                self._record_error(f"send failed: {exc}")
                return

    async def _receiver(self):
        async for event in self._client.events():
            etype = event["type"]
            self._count_event("server", etype)

            if etype == "session.created":
                # An unrecognized model string falls back silently, so read
                # back what the server actually gave us.
                with self._lock:
                    self._session_model = event.get("session", {}).get("model")

            elif etype == "session.updated":
                self._ready.set()

            elif etype == "conversation.created":
                with self._lock:
                    self._conversation_id = self._client.conversation_id

            elif etype == "input_audio_buffer.speech_started":
                # Only an interruption if something was playing. The server
                # fires this whenever it hears speech, including when the
                # caller simply takes their turn, and labelling all of those
                # as interruptions buries the real ones. "Playing" has to
                # mean audible in the browser, not just still generating:
                # response.done can land seconds before PcmAudioSource's FIFO
                # actually empties, so this checks the playback deadline too,
                # not only the raw streaming flag.
                with self._lock:
                    now = time.perf_counter()
                    was_playing = self._is_playing_locked(now)
                    self._speaking = False
                    self._thinking = False
                    self._playback_until = now
                if was_playing:
                    self.on_barge_in()
                    self._note("system", "Caller interrupted. Playback flushed.")

            elif etype == "response.created":
                # Generating, but nothing audible yet. That gap is what the
                # page shows as "thinking" instead of a dead-looking pause.
                with self._lock:
                    self._thinking = True
                    self._reply_started = time.perf_counter()

            elif etype == "response.output_audio.delta":
                pcm = base64.b64decode(event["delta"])
                with self._lock:
                    self._speaking = True
                    self._thinking = False
                    now = time.perf_counter()
                    # Extend the playback deadline by this chunk's duration,
                    # queuing after whatever is already queued rather than
                    # from "now" so back-to-back deltas add up correctly.
                    self._playback_until = max(self._playback_until, now) + pcm_seconds(pcm)
                self.on_output_pcm(pcm)

            elif etype == "response.output_audio_transcript.done":
                self._note("agent", event.get("transcript", ""))

            elif etype in (
                "conversation.item.input_audio_transcription.updated",
                "conversation.item.input_audio_transcription.completed",
            ):
                self._note_caller(event, final=etype.endswith(".completed"))

            elif etype == "response.function_call_arguments.done":
                await self._handle_tool_call(event)

            elif etype == "response.done":
                with self._lock:
                    self._speaking = False
                    self._thinking = False
                    if self._reply_started is not None:
                        self._last_reply_latency = time.perf_counter() - self._reply_started
                        self._reply_started = None
                if self._had_function_call(event):
                    # The tool results are in. Wait for whatever audio is
                    # already queued to finish playing before asking for the
                    # spoken follow-up, or the two responses' audio overlaps
                    # in the browser.
                    await self._wait_for_playback_drain()
                    self._count_event("client", "response.create")
                    await self._client.request_response()

            elif etype == "error":
                self._record_error(event.get("error", {}).get("message", "unknown error"))

    def _is_playing_locked(self, now: float | None = None) -> bool:
        """Whether the browser is still expected to be audibly playing agent
        speech: either the server is actively streaming deltas right now, or
        there is buffered audio queued for playback that has not finished.
        Caller must already hold ``self._lock``.
        """
        if now is None:
            now = time.perf_counter()
        return self._speaking or now < self._playback_until

    async def _wait_for_playback_drain(self):
        """Sleep out whatever audio is still queued for browser playback."""
        with self._lock:
            remaining = self._playback_until - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def _handle_tool_call(self, event):
        name = event["name"]
        try:
            args = json.loads(event["arguments"])
        except json.JSONDecodeError:
            args = {}
        result = execute(name, args)
        with self._lock:
            self._tool_calls.append({"name": name, "arguments": args, "result": result})
            if name == "transfer_to_human":
                # The page shows a banner and the caller is done with us. The
                # socket stays open so the agent can say its handover line.
                self._handoff = args.get("reason") or "Transferring to a human agent."
        self._count_event("client", "conversation.item.create (function_call_output)")
        await self._client.send_tool_result(event["call_id"], result)

    @staticmethod
    def _had_function_call(done_event: dict) -> bool:
        output = done_event.get("response", {}).get("output", [])
        return any(part.get("type") == "function_call" for part in output)

    def stop(self, timeout: float = 5.0):
        """Close the socket on its own loop and wait for the thread to finish."""
        if self._loop is None or self._stopped.is_set():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
        except RuntimeError:
            pass  # loop already gone
        self._stopped.wait(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_speaking(self) -> bool:
        """Whether the agent's speech is still audible right now: streaming,
        or buffered audio queued for playback that has not finished yet.

        Read from the media callback on every frame, so it takes the lock
        briefly rather than building a whole snapshot.
        """
        with self._lock:
            return self._is_playing_locked()

    # ------------------------------------------------------------------
    # Input from Streamlit / the WebRTC thread
    # ------------------------------------------------------------------

    def send_audio(self, pcm: bytes):
        """Queue microphone audio. Safe to call from the aiortc thread."""
        if not pcm or self._loop is None or self._stopped.is_set():
            return
        self._loop.call_soon_threadsafe(self._enqueue_mic_audio, pcm)

    def _enqueue_mic_audio(self, pcm: bytes):
        """Runs on the call's own loop. Drops the oldest buffered chunk
        instead of growing latency without bound when the network, or the
        sender, falls behind real time."""
        try:
            self._send_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._send_queue.put_nowait(pcm)
            except asyncio.QueueFull:
                pass

    def send_text(self, text: str):
        """Type a turn instead of speaking it, for when there is no mic."""
        if self._loop is None or self._stopped.is_set():
            return
        self._note("caller", text)
        self._count_event("client", "conversation.item.create (message)")
        self._count_event("client", "response.create")
        asyncio.run_coroutine_threadsafe(self._say(text), self._loop)

    async def _say(self, text: str):
        await self._client.send_text(text)
        await self._client.request_response()

    # ------------------------------------------------------------------
    # State shared with the page
    # ------------------------------------------------------------------

    def _note(self, role: str, text: str):
        if not text:
            return
        with self._lock:
            # Never stack the same system line twice in a row.
            if role == "system" and self._transcript:
                last = self._transcript[-1]
                if last["role"] == "system" and last["text"] == text:
                    return
            self._transcript.append({"role": role, "text": text})
            # Any line written here closes the open spoken one, so the next
            # thing the caller says starts fresh rather than extending it.
            # Typed turns land here too and are not stitched into.
            self._open_caller_line = None

    # A pause shorter than this reads as one sentence, not two turns.
    MERGE_WINDOW_S = 4.0

    def _note_caller(self, event: dict, final: bool = False):
        """Keep one readable line per thing the caller said.

        Two problems to solve at once. Reports for the same item arrive
        repeatedly and grow, so the line is rewritten rather than appended.
        And a single sentence can be split across items when the caller
        pauses, so a fragment that lands right after the previous one joins
        it instead of starting a new line.
        """
        text = (event.get("transcript") or "").strip()
        if not text:
            return
        item_id = event.get("item_id") or f"anon-{len(self._transcript)}"
        now = time.monotonic()

        with self._lock:
            known = self._item_text.get(item_id)
            # Later reports can correct themselves, but they should never
            # shorten a line back into the partial it grew out of.
            if known is not None and len(text) <= len(known) and not final:
                return
            self._item_text[item_id] = text

            index = self._caller_lines.get(item_id)
            if index is None:
                recent = (now - self._last_caller_at) < self.MERGE_WINDOW_S
                if self._open_caller_line is not None and recent:
                    index = self._open_caller_line
                else:
                    index = len(self._transcript)
                    self._transcript.append({"role": "caller", "text": ""})
                    self._open_caller_line = index
                self._caller_lines[item_id] = index

            # Rebuild the line from every fragment pointing at it, in order.
            parts = [
                self._item_text[key]
                for key, spot in self._caller_lines.items()
                if spot == index and self._item_text.get(key)
            ]
            self._transcript[index]["text"] = " ".join(parts)
            self._last_caller_at = now

    def _record_error(self, message: str):
        with self._lock:
            self._error = message
        self._ready.set()

    def _count_event(self, direction: str, etype: str):
        """Collapse repeats into a count so the panel stays readable."""
        with self._lock:
            if self._event_counts and self._event_counts[-1][:2] == [direction, etype]:
                self._event_counts[-1][2] += 1
            else:
                self._event_counts.append([direction, etype, 1])
                if len(self._event_counts) > 200:
                    del self._event_counts[0]

    def snapshot(self) -> dict:
        """A consistent copy of everything the page renders."""
        with self._lock:
            cost = self._client.cost
            return {
                "transcript": list(self._transcript),
                "tool_calls": list(self._tool_calls),
                "events": [list(e) for e in self._event_counts],
                "session_model": self._session_model,
                "conversation_id": self._conversation_id,
                "error": self._error,
                "handoff": self._handoff,
                "speaking": self._is_playing_locked(),
                "thinking": self._thinking,
                "reply_latency": self._last_reply_latency,
                "audio_seconds": cost.audio_seconds,
                "text_events": cost.text_events,
                "audio_usd": cost.audio_usd,
                "text_usd": cost.text_usd,
                "total_usd": cost.total_usd,
            }


def pcm_seconds(pcm: bytes, rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / 2 / rate
