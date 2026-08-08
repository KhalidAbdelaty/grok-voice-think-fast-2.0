"""Shared configuration for the Grok Voice customer-support agent."""
import os

from dotenv import load_dotenv

load_dotenv()

# Read rather than require at import time, so the Streamlit demo can show a
# readable message instead of a KeyError traceback on a missing .env.
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Pin the versioned model. Do not use grok-voice-latest in production: the
# alias moved from Think Fast 1.0 to Think Fast 2.0 on August 5, 2026, and
# will keep moving to whatever ships next after that.
MODEL = "grok-voice-think-fast-2.0"
REALTIME_URL_BASE = "wss://api.x.ai/v1/realtime"
REALTIME_URL = f"{REALTIME_URL_BASE}?model={MODEL}"
CLIENT_SECRETS_URL = "https://api.x.ai/v1/realtime/client_secrets"

SAMPLE_RATE = 24000  # Hz, the documented default for audio/pcm
VOICE = "eve"

# Audio bills per minute, and the rate depends on which model actually ran
# the session. Keyed by the model string the server reports back in
# session.created, which is what should be billed against - not necessarily
# the alias or version string that was requested.
AUDIO_RATE_PER_MINUTE = {
    "grok-voice-think-fast-2.0": 0.08,
    "grok-voice-think-fast-1.0": 0.05,
}
DEFAULT_AUDIO_RATE_PER_MINUTE = AUDIO_RATE_PER_MINUTE[MODEL]


def audio_rate_per_minute(model: str | None) -> float:
    """USD per minute of audio for a given model string, falling back to the
    pinned default for an alias or an unrecognized/not-yet-known model."""
    return AUDIO_RATE_PER_MINUTE.get(model, DEFAULT_AUDIO_RATE_PER_MINUTE)
