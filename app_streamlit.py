"""Order Support Voice Agent - a live call in the browser.

This is the demo the terminal scripts cannot be. The browser microphone
streams into the same WebSocket the article builds, the agent's voice streams
back, and you can talk over it mid-sentence to see barge-in actually work.

Three panels sit beside the transcript for the things a terminal hides: which
tool fired with which arguments, what the order record looks like afterwards,
and what the call is costing on both billing meters.

Needs XAI_API_KEY in .env, and a microphone on localhost (browsers only allow
mic access over localhost or HTTPS).

Run with:  streamlit run app_streamlit.py
"""
from __future__ import annotations

import sys
import threading
import time

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_pcm_audio_source_track,
    webrtc_streamer,
)

from assistant import SYSTEM_PROMPT, build_session_config
from config import MODEL, SAMPLE_RATE, XAI_API_KEY
from live_call import LiveCall
from tools import get_orders, get_tickets, reset_store
from ui_theme import (
    apply_theme,
    hero,
    render_cost_bar,
    render_events,
    render_orders,
    render_state,
    render_transcript,
    sidebar_logo,
)

if sys.platform == "win32":
    # aiortc's teardown logs a harmless ConnectionResetError (WinError 10054)
    # when a browser tab closes. Swallow only that one so the console stays
    # readable; nothing else changes.
    from asyncio.proactor_events import _ProactorBasePipeTransport
    from functools import wraps as _wraps

    def _silence_conn_reset(func):
        @_wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except ConnectionResetError:
                return None
        return wrapper

    _ProactorBasePipeTransport._call_connection_lost = _silence_conn_reset(
        _ProactorBasePipeTransport._call_connection_lost
    )

MODELS = {
    MODEL: "Versioned string. Pin this in anything you deploy.",
    "grok-voice-latest": "Alias. Moves to a new model on SpaceXAI's schedule, not yours.",
    "grok-voice-think-fast-1.0": "Previous generation, $0.05 a minute instead of $0.08.",
}
VOICES = {
    "eve": "Female, energetic",
    "ara": "Female, warm",
    "rex": "Male, confident",
    "sal": "Neutral, balanced",
    "leo": "Male, authoritative",
}
LANGUAGES = {
    "English": "en",
    "Auto-detect": None,
    "Arabic (Egypt)": "ar-EG",
    "Spanish (Mexico)": "es-MX",
    "Portuguese (Brazil)": "pt-BR",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
}
EXAMPLES = {
    "Order lookup": "What's the status of order ORD-1042?",
    "No order number": "I don't have my order number. My email is dana@example.com.",
    "Delivery change": (
        "Change the delivery instructions for ORD-1042 to leave it with the "
        "doorman. Yes, I confirm, please do it now."
    ),
    "Cancel too late": "I want to cancel order ORD-1042.",
    "Open a ticket": (
        "My order ORD-2210 never arrived and tracking hasn't moved in a week. "
        "Please open a ticket."
    ),
    "Ask for a human": "This is about a double charge on my card. Get me a person.",
}

# The mic callback runs on aiortc's loop, where st.session_state is not
# readable. A module-level box guarded by a lock is the documented way to
# hand values across that boundary.
_BOX_LOCK = threading.Lock()
_BOX: dict = {
    "call": None,
    "resampler": None,
    "level": 0.0,
    "echo_gate": 0.0,   # ignore mic audio below this while the agent speaks
    "half_duplex": False,  # or ignore it entirely, trading away barge-in
    "gated": 0,         # frames suppressed, so the page can show it working
}

st.set_page_config(page_title="Order Support Voice Agent", page_icon="🎧", layout="wide")
apply_theme()


def _on_mic_frame(frame: av.AudioFrame):
    """Browser audio in. Resample to the session format and hand it off.

    Runs on aiortc's event loop, so it does no real work: converting the
    frame, deciding whether to keep it, and queueing it. Anything slower
    would backpressure the media stream.

    On laptop speakers the agent's own voice reaches the microphone and the
    server hears it as the caller, so it interrupts itself and chops the
    caller's next sentence into pieces. Browser echo cancellation removes
    most of that; the gate below drops whatever survives, on the assumption
    that someone talking into the microphone is louder than a speaker
    bleeding into it from across the desk.
    """
    with _BOX_LOCK:
        call = _BOX["call"]
        resampler = _BOX["resampler"]
        gate = _BOX["echo_gate"]
        half_duplex = _BOX["half_duplex"]
    if call is None or resampler is None:
        return

    agent_speaking = call.is_speaking()
    try:
        for chunk in resampler.resample(frame):
            samples = chunk.to_ndarray()
            if samples.size == 0:
                continue
            level = float(np.abs(samples).max()) / 32768.0

            if agent_speaking and (half_duplex or level < gate):
                with _BOX_LOCK:
                    _BOX["level"] = level
                    _BOX["gated"] += 1
                continue

            with _BOX_LOCK:
                _BOX["level"] = level
            call.send_audio(samples.astype(np.int16).tobytes())
    except Exception:  # noqa: BLE001 - never kill the media loop over one frame
        pass


def _init_state():
    defaults = {
        "conversation_id": None,
        "changed_fields": set(),
        "orders_before": None,
        "pending_text": None,
        "last_error": None,
        # A LiveCall dies when you press STOP, and everything it was holding
        # would go with it. These keep the record of the calls that ended, so
        # the page still has a transcript and a running total afterwards.
        "history": [],
        "tools_history": [],
        "spent_audio_seconds": 0.0,
        "spent_text_events": 0,
        "last_model_seen": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _archive(call: LiveCall):
    """Fold a finished call into the session totals before dropping it."""
    snap = call.snapshot()
    st.session_state.conversation_id = snap["conversation_id"]
    st.session_state.history.extend(snap["transcript"])
    st.session_state.tools_history.extend(snap["tool_calls"])
    st.session_state.spent_audio_seconds += snap["audio_seconds"]
    st.session_state.spent_text_events += snap["text_events"]
    if snap["session_model"]:
        st.session_state.last_model_seen = snap["session_model"]
    if st.session_state.history:
        st.session_state.history.append(
            {"role": "system", "text": "Call ended. Start again to keep the same conversation."}
        )


def _track_changes():
    """Diff the order store so the page can highlight what the agent changed."""
    now = get_orders()
    before = st.session_state.orders_before
    if before is None:
        st.session_state.orders_before = now
        return
    for number, order in now.items():
        for key, value in order.items():
            if before.get(number, {}).get(key) != value:
                st.session_state.changed_fields.add((number, key))
    st.session_state.orders_before = now


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    sidebar_logo()

    if XAI_API_KEY:
        st.success("API key loaded.")
    else:
        st.error("Set XAI_API_KEY in code/.env, then reload.")

    model = st.selectbox("Model", list(MODELS))
    st.caption(MODELS[model])
    voice = st.selectbox("Voice", list(VOICES), format_func=lambda v: f"{v} ({VOICES[v]})")

    with st.expander("Turn taking"):
        silence_ms = st.slider(
            "Silence before the turn ends (ms)", 200, 1500, 600, 50,
            help="Server VAD. Too short cuts callers off mid-thought, too long feels sluggish.",
        )
        threshold = st.slider(
            "Speech threshold", 0.1, 0.9, 0.85, 0.05,
            help="How loud audio has to be to count as speech. Lower it in a quiet room.",
        )

    with st.expander("Echo control", expanded=not st.session_state.history):
        st.caption("On laptop speakers the agent hears itself and interrupts "
                   "its own sentence. Headphones fix it outright.")
        half_duplex = st.toggle(
            "Mute the mic while the agent speaks", value=False,
            help="Ends the echo completely, and gives up barge-in while it is on.",
        )
        echo_gate = st.slider(
            "Echo gate", 0.0, 0.5, 0.12, 0.01, disabled=half_duplex,
            help="While the agent speaks, ignore mic audio quieter than this. "
                 "Raise it if the agent keeps cutting itself off, lower it if "
                 "your interruptions are being missed. 0 turns the gate off.",
        )

    with st.expander("Voice and language"):
        language_label = st.selectbox("Language hint", list(LANGUAGES))
        speed = st.slider("Output speed", 0.7, 1.5, 1.0, 0.05)

    with st.expander("System prompt"):
        prompt = st.text_area("System prompt", SYSTEM_PROMPT, height=200,
                              label_visibility="collapsed")

    st.divider()
    if st.session_state.conversation_id:
        st.caption(f"Conversation `{st.session_state.conversation_id[:18]}...` kept for "
                   "resumption, expires after 30 minutes idle.")

    if st.button("Reset demo data", width="stretch",
                 help="Restore the order store and clear the transcript and totals."):
        reset_store()
        st.session_state.changed_fields = set()
        st.session_state.orders_before = get_orders()
        st.session_state.history = []
        st.session_state.tools_history = []
        st.session_state.spent_audio_seconds = 0.0
        st.session_state.spent_text_events = 0
        st.session_state.conversation_id = None
        st.toast("Order store, transcript and totals cleared.")

# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

hero()

# The mic callback cannot read session_state, so push the current settings
# into the shared box on every rerun.
with _BOX_LOCK:
    _BOX["echo_gate"] = echo_gate
    _BOX["half_duplex"] = half_duplex

session_config = build_session_config(
    voice=voice,
    turn_detection={
        "type": "server_vad",
        "threshold": threshold,
        "silence_duration_ms": silence_ms,
    },
    instructions=prompt,
    language_hint=LANGUAGES[language_label],
    speed=speed,
)

# Cached by key, so both tracks survive Streamlit's reruns. These need a live
# Streamlit runtime; under a test harness they raise, and the page should say
# so rather than dying on an import-time traceback.
media_error = ""
try:
    agent_voice = create_pcm_audio_source_track(key="agent-voice", sample_rate=SAMPLE_RATE)
    mic_sink = create_audio_sink_track(_on_mic_frame, key="caller-mic")
    media_ready = True
except Exception as exc:  # noqa: BLE001 - reported in the page below
    agent_voice = mic_sink = None
    media_ready = False
    media_error = type(exc).__name__

left, right = st.columns([3, 2], gap="large")

with left:
    ctx = None
    playing = False
    if media_ready:
        try:
            ctx = webrtc_streamer(
                key="call",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={
                    "video": False,
                    # Left to the browser's defaults this leaks badly on open
                    # speakers. Automatic gain is off on purpose: it lifts
                    # quiet speaker bleed up over the server's VAD threshold.
                    "audio": {
                        "echoCancellation": True,
                        "noiseSuppression": True,
                        "autoGainControl": False,
                    },
                },
                source_audio_track=agent_voice.track,
                sink_audio_track=mic_sink,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                audio_html_attrs={"autoPlay": True, "controls": False, "muted": False},
            )
            playing = ctx.state.playing
        except Exception as exc:  # noqa: BLE001 - reported just below
            media_ready = False
            media_error = type(exc).__name__

    if not media_ready:
        st.warning(
            f"Audio streaming is unavailable in this context ({media_error}). "
            "Run the app with `streamlit run app_streamlit.py` and open it on "
            "localhost."
        )
    call: LiveCall | None = st.session_state.get("_call")

    # Open the WebSocket when the call starts, close it when the call stops.
    if playing and call is None:
        if not XAI_API_KEY:
            st.error("No API key, so the call cannot connect.")
        else:
            try:
                call = LiveCall(
                    session_config=session_config,
                    model=model,
                    conversation_id=st.session_state.conversation_id,
                    on_output_pcm=agent_voice.push,
                    on_barge_in=agent_voice.clear,
                )
                with st.spinner("Connecting..."):
                    call.start()
                st.session_state._call = call
                with _BOX_LOCK:
                    _BOX["call"] = call
                    _BOX["resampler"] = av.AudioResampler(
                        format="s16", layout="mono", rate=SAMPLE_RATE
                    )
                st.session_state.last_error = None
            except Exception as exc:  # noqa: BLE001 - shown in the page
                st.session_state.last_error = f"{type(exc).__name__}: {exc}"
                call = None

    if not playing and call is not None:
        _archive(call)
        call.stop()
        agent_voice.clear()
        with _BOX_LOCK:
            _BOX["call"] = None
            _BOX["resampler"] = None
        st.session_state._call = None
        call = None

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    status_slot = st.empty()
    handoff_slot = st.empty()
    transcript_slot = st.empty()

    with st.expander("No microphone? Type a turn instead"):
        cols = st.columns(2)
        for i, (label, line) in enumerate(EXAMPLES.items()):
            if cols[i % 2].button(label, width="stretch", disabled=not playing):
                st.session_state.pending_text = line

    typed = st.chat_input("Type what the caller says", disabled=not playing)
    if typed:
        st.session_state.pending_text = typed

    if st.session_state.pending_text and call is not None:
        call.send_text(st.session_state.pending_text)
        st.session_state.pending_text = None

with right:
    record_tab, tools_tab, cost_tab, events_tab = st.tabs(
        ["Order record", "Tool calls", "Cost", "Event flow"]
    )
    with record_tab:
        st.caption("The agent can say anything. This is what actually changed.")
        record_slot = st.empty()
        tickets_slot = st.empty()
    with tools_tab:
        st.caption("Every function the model called, with what it passed and got back.")
        tools_slot = st.empty()
    with cost_tab:
        st.caption("Both meters, for the whole session rather than the current call.")
        cost_slot = st.empty()
        meta_slot = st.empty()
    with events_tab:
        st.caption("Repeats are collapsed, or audio deltas would bury everything else.")
        events_slot = st.empty()


def paint(snap: dict | None):
    """Draw the page from the calls that ended plus the one still running."""
    _track_changes()
    speaking = bool(snap and snap["speaking"])
    thinking = bool(snap and snap.get("thinking"))
    with _BOX_LOCK:
        level = _BOX["level"] if playing else 0.0
    status_slot.markdown(
        render_state(playing, speaking, thinking, level), unsafe_allow_html=True
    )

    handoff = (snap or {}).get("handoff")
    handoff_slot.markdown(
        f'<div class="banner banner-transfer">Transferring to a human: {handoff}</div>'
        if handoff else "",
        unsafe_allow_html=True,
    )

    turns = st.session_state.history + (snap["transcript"] if snap else [])
    transcript_slot.markdown(render_transcript(turns), unsafe_allow_html=True)
    record_slot.markdown(
        render_orders(get_orders(), st.session_state.changed_fields), unsafe_allow_html=True
    )

    tickets = get_tickets()
    tickets_slot.markdown(
        "".join(
            f'<div class="rec"><div class="rec-id">{t["ticket_id"]}</div>'
            f'<div class="rec-row"><span class="rec-key">order</span>'
            f'<span class="rec-val">{t["order_number"]}</span></div>'
            f'<div class="rec-row"><span class="rec-key">reason</span>'
            f'<span class="rec-val">{t["reason"]}</span></div></div>'
            for t in tickets
        ),
        unsafe_allow_html=True,
    )

    calls = st.session_state.tools_history + (snap["tool_calls"] if snap else [])
    if calls:
        tools_slot.markdown(
            "".join(
                f'<div class="rec"><span class="pill">{c["name"]}</span>'
                f'<div class="rec-row"><span class="rec-key">arguments</span>'
                f'<span class="rec-val">{c["arguments"]}</span></div>'
                f'<div class="rec-row"><span class="rec-key">result</span>'
                f'<span class="rec-val">{c["result"]}</span></div></div>'
                for c in calls
            ),
            unsafe_allow_html=True,
        )
    else:
        tools_slot.markdown(
            '<div class="hint">No tools called yet.</div>', unsafe_allow_html=True
        )

    # Totals span the session. A new call starts a fresh meter, so the spend
    # from calls already ended has to be carried rather than read off the
    # object that is gone.
    audio_seconds = st.session_state.spent_audio_seconds + (snap["audio_seconds"] if snap else 0.0)
    text_events = st.session_state.spent_text_events + (snap["text_events"] if snap else 0)
    audio_usd = audio_seconds / 60 * 0.08
    text_usd = text_events * 0.004
    cost_slot.markdown(render_cost_bar(audio_usd, text_usd), unsafe_allow_html=True)

    model_seen = (snap and snap["session_model"]) or st.session_state.last_model_seen
    bits = [f"{audio_seconds:.1f}s audio", f"{text_events} text event(s)"]
    if snap and snap["reply_latency"]:
        bits.append(f"last reply {snap['reply_latency']:.1f}s")
    if model_seen:
        bits.append(f"server model `{model_seen}`")
    meta_slot.caption(" · ".join(bits))
    events_slot.markdown(render_events(snap["events"] if snap else []), unsafe_allow_html=True)


# While the call is up, keep pulling state out of the worker thread. This is
# the documented way to read values a media callback produced: the script
# would otherwise finish and the page would freeze on the first frame.
if call is not None and ctx is not None and ctx.state.playing:
    while ctx.state.playing:
        snap = call.snapshot()
        paint(snap)
        if snap["error"]:
            st.error(snap["error"])
            break
        if not call.running:
            break
        time.sleep(0.4)
else:
    paint(None)
