# Order Support Voice Agent (Grok Voice Think Fast 2.0)

A real-time customer-support agent built on the SpaceXAI Speech to Speech API.
It configures a voice session over a WebSocket, streams audio, calls order
tools, speaks the reply, survives a dropped connection through session
resumption, and tracks cost on both billing meters.

The same workflow ships three ways: scripted scenarios in a terminal, a
FastAPI token endpoint, and a live browser call. It runs against the live API,
so you need an API key with a funded billing arrangement.

## Files

- `config.py` - loads `XAI_API_KEY`, pins the model string, and holds the
  endpoint URLs and sample rate.
- `tools.py` - the seven order functions and their JSON schemas, over an
  in-memory store standing in for a database: `check_order_status`,
  `find_orders` (by email or phone, for a caller with no order number),
  `update_delivery_instructions`, `cancel_order`, `create_support_ticket`,
  `check_ticket_status`, and `transfer_to_human`. The rules live here rather
  than in the prompt: `cancel_order` refuses once an order has shipped and
  returns the reason and the alternative, so the agent explains the policy
  instead of inventing one. `execute()` never raises, so a tool failure comes
  back as a result the model can speak instead of a dropped connection.
- `voice_client.py` - the WebSocket client: session config, audio in and out,
  tool results, and a `CostTracker` for the audio and text meters.
- `assistant.py` - the system prompt, session config, and the event loop that
  runs tools, waits for playback to drain before continuing, and clears the
  local queue on a barge-in.
- `audio_utils.py` - WAV to PCM16 and back, used by the tests and any script
  that needs to save or load a clip.
- `live_call.py` - one WebSocket held open for a whole call, driven from a
  background thread: microphone audio in, agent audio out, tools in between,
  and a playback flush the moment the caller interrupts.
- `run_demo.py` - four scripted scenarios in the terminal: lookup, unknown
  order, confirmed delivery change, and a ticket.
- `test_resumption.py` - drops the connection mid-conversation and reconnects
  with the saved `conversation_id`.
- `test_live_call.py` - runs the live-call engine against a fake Speech to
  Speech server: tool loop, barge-in, cost metering, one session per call, and
  a clean close. No API key, no spend.
- `test_audio.py` - checks both audio paths convert at the session rate, and
  that flushing the playback track really does silence it.
- `test_app_state.py` - drives the page through Streamlit's test harness to
  check the transcript, tool log and running cost survive pressing STOP, and
  that the echo controls behave.
- `test_tools.py` - the tool rules: a shipped order cannot be cancelled, a
  delivered one cannot be redirected, and contact details never reach the
  model.
- `token_server.py` - FastAPI `POST /session` that mints an ephemeral token so
  a browser never sees your API key.
- `app_streamlit.py` - the live browser call over WebRTC: real barge-in, the
  tool-call log, the order record with the changed field highlighted, the
  collapsed event flow, and cost split across both meters.
- `ui_theme.py` - the CSS and the render helpers for that page.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # then paste your key into .env
```

## Run

```bash
# Four scripted scenarios in the terminal
python run_demo.py

# Drop and resume a conversation
python test_resumption.py

# Ephemeral token endpoint, then open http://localhost:8000/docs
uvicorn token_server:app --reload

# Live browser call
streamlit run app_streamlit.py

# Offline tests, no key and no spend
python test_live_call.py
python test_audio.py
python test_app_state.py
python test_tools.py
```

Open the Streamlit URL as `http://localhost:8501`. Browsers only hand out the
microphone over localhost or HTTPS, so a plain `http://` address on another
machine will connect and then sit there recording silence. Serving it remotely
means terminating TLS and configuring a STUN or TURN server.

Wear headphones. On laptop speakers the agent's own voice reaches the
microphone, the server hears it as the caller, and the agent interrupts itself
and chops your next sentence into fragments. The browser's echo cancellation
is requested explicitly and there is an "Echo gate" slider that drops quiet
audio while the agent talks, but neither is as good as not putting the speaker
next to the microphone. The sidebar also has a "Mute the mic while the agent
speaks" toggle, which ends the problem completely and gives up barge-in while
it is on.

## Notes

- Everything reads `XAI_API_KEY` from `.env` and connects to
  `wss://api.x.ai/v1/realtime`. Keep the key server-side and hand browser or
  mobile clients an ephemeral token from `/session` instead.
- Audio bills at $0.08 per minute on `grok-voice-think-fast-2.0`, sent and
  received. Each non-audio `conversation.item.create` is $0.004, and
  `function_call_output` results are free.
- The browser demo holds one WebSocket for the whole call and uses server VAD,
  so the model decides when a turn ended. Stopping and restarting the call
  reuses the saved `conversation_id`, which is the resumption path. The
  terminal scripts take the other route, one connection per turn.
- Both paths close the socket on the loop that opened it. That matters more
  than it looks: the team gets 10 concurrent sessions, and a socket dropped
  without a close handshake keeps its session alive server-side.
- A confirmed delivery change edits the in-memory order for the life of the
  process. Use "Start over" in the sidebar to restore the demo data.

## License

MIT
