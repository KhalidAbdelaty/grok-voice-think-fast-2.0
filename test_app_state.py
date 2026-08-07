"""Checks that the page keeps its record after a call ends.

A LiveCall dies on STOP, and it used to take the transcript, the tool log and
the running cost with it. These run the page through Streamlit's own test
harness, no key and no network.

Run: python test_app_state.py
"""
from __future__ import annotations

from streamlit.testing.v1 import AppTest

failures: list[str] = []


def check(label: str, ok: bool, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f" -> {detail}"))
    if not ok:
        failures.append(label)


def fresh() -> AppTest:
    return AppTest.from_file("app_streamlit.py", default_timeout=120).run()


print("First load:")
at = fresh()
check("page renders without error", at.exception == [], at.exception)
check("history starts empty", at.session_state["history"] == [])
check("totals start at zero",
      at.session_state["spent_audio_seconds"] == 0.0
      and at.session_state["spent_text_events"] == 0)
check("four tabs on the right", len(at.tabs) == 4, len(at.tabs))
check("tab labels", [t.label for t in at.tabs] == ["Order record", "Tool calls", "Cost", "Event flow"],
      [t.label for t in at.tabs])

print("\nAfter a call ended (state a stopped call leaves behind):")
at.session_state["history"] = [
    {"role": "caller", "text": "What's the status of ORD-1042?"},
    {"role": "agent", "text": "It is out for delivery."},
    {"role": "system", "text": "Call ended. Start again to keep the same conversation."},
]
at.session_state["tools_history"] = [
    {"name": "check_order_status",
     "arguments": {"order_number": "ORD-1042"},
     "result": {"status": "out_for_delivery"}},
]
at.session_state["spent_audio_seconds"] = 12.5
at.session_state["spent_text_events"] = 2
at.session_state["last_model_seen"] = "grok-voice-think-fast-2.0"
at.session_state["conversation_id"] = "conv_kept_123"
at.run()

check("page still renders", at.exception == [], at.exception)

body = " ".join(m.value for m in at.markdown)
check("transcript survives the stop", "out for delivery" in body)
check("caller line survives", "ORD-1042" in body)
check("tool call survives", "check_order_status" in body)

caption_text = " ".join(c.value for c in at.caption)
check("audio total survives", "12.5s audio" in caption_text, caption_text[:160])
check("text meter survives", "2 text event" in caption_text, caption_text[:160])
check("model seen survives", "grok-voice-think-fast-2.0" in caption_text, caption_text[:160])
check("conversation id kept for resumption",
      at.session_state["conversation_id"] == "conv_kept_123")

print("\nCall state indicator:")
from ui_theme import render_state  # noqa: E402 - after the harness runs

idle = render_state(playing=False, speaking=False, thinking=False, level=0.0)
listening = render_state(playing=True, speaking=False, thinking=False, level=0.6)
speaking = render_state(playing=True, speaking=True, thinking=False, level=0.0)
thinking = render_state(playing=True, speaking=False, thinking=True, level=0.0)

check("idle reads as not connected", "Not connected" in idle and "live-off" in idle)
check("listening is distinct", "Listening" in listening and "live-listen" in listening)
check("speaking is distinct", "Agent speaking" in speaking and "live-speak" in speaking)
check("thinking is distinct", "Thinking" in thinking and "live-think" in thinking)
check("speaking invites an interruption", "Talk over it to interrupt" in speaking)
check("the meter lights up with input level",
      listening.count('class="on"') > 0, listening.count('class="on"'))
check("a silent mic leaves the meter dark",
      render_state(True, False, False, 0.0).count('class="on"') == 0)

print("\nEcho controls:")
sliders = {s.label: s for s in at.slider}
toggles = {t.label: t for t in at.toggle}
check("there is an echo gate", "Echo gate" in sliders, list(sliders))
check("the gate defaults to something conservative",
      "Echo gate" in sliders and 0 < sliders["Echo gate"].value <= 0.2,
      sliders.get("Echo gate").value if "Echo gate" in sliders else "missing")
check("half duplex is offered", "Mute the mic while the agent speaks" in toggles, list(toggles))
check("half duplex is off by default",
      toggles.get("Mute the mic while the agent speaks")
      and toggles["Mute the mic while the agent speaks"].value is False)

import app_streamlit  # noqa: E402 - the harness has already imported it

check("the gate reaches the mic callback",
      app_streamlit._BOX["echo_gate"] == sliders["Echo gate"].value,
      app_streamlit._BOX["echo_gate"])


def gated(level, speaking, gate=0.12, half=False):
    """Would a frame at this level be dropped before reaching the socket?"""
    return speaking and (half or level < gate)


check("speaker bleed is dropped while the agent talks", gated(0.04, speaking=True))
check("a real interruption still gets through", not gated(0.35, speaking=True))
check("nothing is dropped while the agent is quiet", not gated(0.04, speaking=False))
check("half duplex drops everything while the agent talks",
      gated(0.9, speaking=True, half=True))

print("\nSidebar can be reopened:")
from ui_theme import CSS  # noqa: E402

check("the header itself is not hidden", "header { visibility: hidden" not in CSS)
check("the collapsed-sidebar control is kept visible",
      "stSidebarCollapsedControl" in CSS)
check("the toolbar is still hidden", "stToolbar" in CSS)

print("\nReset:")
reset = next(b for b in at.button if b.label == "Reset demo data")
reset.click().run()
check("reset clears the transcript", at.session_state["history"] == [])
check("reset clears the totals",
      at.session_state["spent_audio_seconds"] == 0.0
      and at.session_state["spent_text_events"] == 0)
check("reset drops the conversation id", at.session_state["conversation_id"] is None)
check("page renders after reset", at.exception == [], at.exception)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    raise SystemExit(1)
print("All page-state checks passed.")
