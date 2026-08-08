"""Offline tests for the live-call path, against a fake Speech to Speech server.

These run without an API key and without spending anything. The fake server
replays the event sequences the real one sends, driven by commands in the
text turns, so each scenario is explicit: a tool call, speech detected while
nothing is playing, a genuine barge-in, and a sentence that voice activity
detection chopped into pieces.

Run: python test_live_call.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time

import websockets

import config

SAMPLE_RATE = 24000
PCM_100MS = b"\x10\x00" * 2400
sessions_opened = 0
sessions_closed = 0
# Wall-clock markers the fake server drops for the client-timing tests below.
# A plain module-level dict is enough: fake_server runs on the asyncio loop
# thread and run_checks() polls it from a worker thread, same pattern as
# sessions_opened/sessions_closed above.
event_times: dict = {}


async def fake_server(ws):
    """Speak the parts of the protocol the client depends on.

    The client's text turns double as commands, which keeps each scenario in
    the test readable instead of hidden behind timing.
    """
    global sessions_opened, sessions_closed
    sessions_opened += 1
    await ws.send(json.dumps({"type": "session.created",
                              "session": {"model": "grok-voice-think-fast-2.0"}}))
    await ws.send(json.dumps({"type": "conversation.created",
                              "conversation": {"id": "conv_fake_123"}}))
    command = None
    # Which flow (if any) is waiting for a clean, no-function-call follow-up
    # response.create - so the second round-trip of a tool turn ends the
    # turn instead of reporting another function_call and recursing forever.
    pending_followup: str | None = None
    try:
        async for raw in ws:
            event = json.loads(raw)
            etype = event.get("type")

            if etype == "session.update":
                await ws.send(json.dumps({"type": "session.updated"}))

            elif etype == "conversation.item.create":
                item = event.get("item", {})
                if item.get("type") == "message":
                    content = item.get("content", [{}])[0].get("text", "")
                    command = content.strip()

            elif etype == "response.create":
                if pending_followup == "tool_once":
                    # The client waited out the queued playback and is now
                    # asking for the spoken follow-up to the earlier tool
                    # call. Record when that happened and end the turn
                    # cleanly instead of handing out another tool call.
                    pending_followup = None
                    event_times["followup_request_at"] = time.monotonic()
                    await ws.send(json.dumps({"type": "response.done",
                                              "response": {"output": []}}))
                elif pending_followup == "default_tool":
                    # Same idea for the default tool-turn scenario below: its
                    # response.done also reports a function_call, so without
                    # this the client's follow-up request.create would loop
                    # back into that same branch forever in the background.
                    pending_followup = None
                    await ws.send(json.dumps({"type": "response.done",
                                              "response": {"output": []}}))
                elif command == "TOOL_ONCE":
                    # A tool call whose audio takes real time to play out in
                    # the browser. The client must not ask for the spoken
                    # follow-up until that time has passed, or the two
                    # responses' audio overlaps.
                    await ws.send(json.dumps({"type": "response.created"}))
                    await ws.send(json.dumps({
                        "type": "response.function_call_arguments.done",
                        "name": "check_order_status", "call_id": "call_drain",
                        "arguments": json.dumps({"order_number": "ORD-1042"}),
                    }))
                    for _ in range(3):  # 300 ms of audio queued for playback
                        await ws.send(json.dumps({
                            "type": "response.output_audio.delta",
                            "delta": base64.b64encode(PCM_100MS).decode(),
                        }))
                    await ws.send(json.dumps({
                        "type": "response.done",
                        "response": {"output": [{"type": "function_call"}]},
                    }))
                    event_times["tool_done_at"] = time.monotonic()
                    pending_followup = "tool_once"
                elif command == "BARGE_IN_AFTER_DONE":
                    # Audio finishes generating (response.done) well before
                    # it finishes playing in the browser. A barge-in that
                    # arrives in that gap must still flush playback.
                    await ws.send(json.dumps({"type": "response.created"}))
                    for _ in range(3):  # 300 ms of audio queued for playback
                        await ws.send(json.dumps({
                            "type": "response.output_audio.delta",
                            "delta": base64.b64encode(PCM_100MS).decode(),
                        }))
                    await ws.send(json.dumps({"type": "response.done",
                                              "response": {"output": []}}))
                    await asyncio.sleep(0.05)  # still mid-playback client-side
                    await ws.send(json.dumps({"type": "input_audio_buffer.speech_started"}))
                elif command == "BARGE_IN":
                    # Start speaking, then have the caller cut in mid-flow.
                    await ws.send(json.dumps({"type": "response.created"}))
                    for _ in range(3):
                        await ws.send(json.dumps({
                            "type": "response.output_audio.delta",
                            "delta": base64.b64encode(PCM_100MS).decode(),
                        }))
                        await asyncio.sleep(0.02)
                    await ws.send(json.dumps({"type": "input_audio_buffer.speech_started"}))
                    await ws.send(json.dumps({"type": "response.done",
                                              "response": {"output": []}}))
                elif command == "IDLE_SPEECH":
                    # Speech detected with nothing playing. The caller is
                    # simply taking their turn, which is not an interruption.
                    await ws.send(json.dumps({"type": "input_audio_buffer.speech_started"}))
                    await ws.send(json.dumps({"type": "input_audio_buffer.speech_started"}))
                    await ws.send(json.dumps({"type": "response.done",
                                              "response": {"output": []}}))
                elif command == "FRAGMENTS":
                    # One sentence split across items by a pause, each item
                    # reported cumulatively and then finalised.
                    for item_id, parts in (
                        ("frag_a", ["Change the", "Change the delivery"]),
                        ("frag_b", ["instructions to", "instructions to the doorman."]),
                    ):
                        for part in parts:
                            await ws.send(json.dumps({
                                "type": "conversation.item.input_audio_transcription.updated",
                                "item_id": item_id, "transcript": part,
                            }))
                            await asyncio.sleep(0.01)
                        await ws.send(json.dumps({
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": item_id, "transcript": parts[-1],
                        }))
                    # A stale partial must not shorten the finished line.
                    await ws.send(json.dumps({
                        "type": "conversation.item.input_audio_transcription.updated",
                        "item_id": "frag_b", "transcript": "instructions to",
                    }))
                    await ws.send(json.dumps({"type": "response.done",
                                              "response": {"output": []}}))
                else:
                    await ws.send(json.dumps({"type": "response.created"}))
                    await ws.send(json.dumps({
                        "type": "response.function_call_arguments.done",
                        "name": "check_order_status",
                        "call_id": "call_1",
                        "arguments": json.dumps({"order_number": "ORD-1042"}),
                    }))
                    await asyncio.sleep(0.05)
                    await ws.send(json.dumps({
                        "type": "response.output_audio.delta",
                        "delta": base64.b64encode(PCM_100MS).decode(),
                    }))
                    await ws.send(json.dumps({
                        "type": "response.output_audio_transcript.done",
                        "transcript": "Order ORD-1042 is out for delivery.",
                    }))
                    await ws.send(json.dumps({
                        "type": "response.done",
                        "response": {"output": [{"type": "function_call"}]},
                    }))
                    pending_followup = "default_tool"
                command = None
    except websockets.ConnectionClosed:
        pass
    finally:
        sessions_closed += 1


def run_checks():
    from live_call import LiveCall

    failures = []

    def check(label, condition, detail=""):
        print(f"  {'PASS' if condition else 'FAIL'}  {label}"
              + ("" if condition else f" -> {detail}"))
        if not condition:
            failures.append(label)

    def wait_for(predicate, timeout=6.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    played = bytearray()
    barge_ins = []

    call = LiveCall(
        session_config={"voice": "eve"},
        on_output_pcm=played.extend,
        on_barge_in=lambda: barge_ins.append(1),
    )
    call.start(timeout=10)

    print("Tool turn:")
    call.send_text("What's the status of order ORD-1042?")
    wait_for(lambda: any(t["role"] == "agent" for t in call.snapshot()["transcript"]))
    snap = call.snapshot()
    check("connected and configured", snap["session_model"] == "grok-voice-think-fast-2.0",
          snap["session_model"])
    check("conversation id captured", snap["conversation_id"] == "conv_fake_123",
          snap["conversation_id"])
    check("tool ran and returned the order",
          bool(snap["tool_calls"])
          and snap["tool_calls"][0]["result"].get("status") == "out_for_delivery",
          snap["tool_calls"])
    check("agent transcript captured",
          any("out for delivery" in t["text"] for t in snap["transcript"]), snap["transcript"])
    check("audio reached the playback track", len(played) == 4800, f"{len(played)} bytes")
    check("audio metered", snap["audio_seconds"] > 0, snap["audio_seconds"])
    check("events collapsed, not one row per delta", len(snap["events"]) < 20,
          f"{len(snap['events'])} rows")
    check("no error", snap["error"] is None, snap["error"])

    print("\nSpeech with nothing playing:")
    before = len([t for t in call.snapshot()["transcript"] if t["role"] == "system"])
    call.send_text("IDLE_SPEECH")
    time.sleep(0.6)
    after = [t for t in call.snapshot()["transcript"] if t["role"] == "system"]
    check("taking a turn is not logged as an interruption", len(after) == before,
          f"{len(after) - before} spurious line(s)")
    check("nothing was flushed", not barge_ins, barge_ins)

    print("\nGenuine barge-in:")
    call.send_text("BARGE_IN")
    check("playback was flushed", wait_for(lambda: len(barge_ins) >= 1), barge_ins)
    system_lines = [t for t in call.snapshot()["transcript"] if t["role"] == "system"]
    check("logged exactly once despite repeats", len(system_lines) == 1,
          f"{len(system_lines)} lines")
    check("the line says it was an interruption",
          system_lines and "interrupted" in system_lines[0]["text"].lower(),
          system_lines)

    print("\nBarge-in after response.done, while audio is still buffered:")
    # response.done only means the server finished generating; the browser's
    # PcmAudioSource can still have real audio queued. A speech_started that
    # lands in that gap must still flush playback (regression: _speaking used
    # to flip False on response.done, which made a barge-in landing here a
    # silent no-op while stale agent audio kept playing over the caller).
    before_flushes = len(barge_ins)
    call.send_text("BARGE_IN_AFTER_DONE")
    check("playback was still flushed",
          wait_for(lambda: len(barge_ins) > before_flushes), barge_ins)

    print("\nTool continuation waits for queued playback to drain:")
    # Regression: the client used to fire the follow-up response.create the
    # instant response.done arrived, without waiting for the ~300 ms of
    # audio it had just queued to actually finish playing - so the follow-up
    # response's audio could start overlapping the tool-call response's tail.
    event_times.clear()
    call.send_text("TOOL_ONCE")
    check("the follow-up request was sent",
          wait_for(lambda: "followup_request_at" in event_times), event_times)
    gap = event_times.get("followup_request_at", 0) - event_times.get("tool_done_at", 0)
    check("the client waited for the queued audio to finish before continuing",
          gap >= 0.25, f"{gap:.3f}s gap, expected >= ~0.3s")

    print("\nA sentence chopped by voice activity detection:")
    call.send_text("FRAGMENTS")
    wait_for(lambda: any("doorman" in t["text"] for t in call.snapshot()["transcript"]))
    time.sleep(0.3)
    spoken = [t["text"] for t in call.snapshot()["transcript"]
              if t["role"] == "caller" and "Change the" in t["text"]]
    check("the pieces join into one line", len(spoken) == 1, f"{len(spoken)} lines: {spoken}")
    check("the line reads as the whole sentence",
          spoken and spoken[0] == "Change the delivery instructions to the doorman.",
          spoken[0] if spoken else "none")
    check("a stale partial does not shorten it",
          spoken and spoken[0].endswith("doorman."), spoken)

    print("\nShutdown:")
    call.stop(timeout=5)
    time.sleep(0.5)
    check("worker thread stopped", not call.running, "still alive")
    check("exactly one session opened", sessions_opened == 1, sessions_opened)
    check("session closed cleanly", sessions_closed == 1, sessions_closed)

    return failures


async def main():
    async with websockets.serve(fake_server, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        config.REALTIME_URL_BASE = f"ws://127.0.0.1:{port}"
        config.REALTIME_URL = f"{config.REALTIME_URL_BASE}?model=grok-voice-think-fast-2.0"

        import voice_client
        voice_client.REALTIME_URL = config.REALTIME_URL
        voice_client.REALTIME_URL_BASE = config.REALTIME_URL_BASE
        voice_client.XAI_API_KEY = "test-key"

        failures = await asyncio.get_running_loop().run_in_executor(None, run_checks)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        raise SystemExit(1)
    print("All live-call checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
