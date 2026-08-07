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
