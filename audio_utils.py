"""WAV helpers for the Streamlit demo.

The realtime API speaks raw 16-bit mono PCM, but browsers record and play
WAV. These two functions are the whole bridge: one wraps PCM coming out of
the model so `st.audio` can play it, the other unwraps a browser recording
into the PCM the API expects.
"""
import io
import wave

import numpy as np

from config import SAMPLE_RATE


def pcm_to_wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 mono in a WAV container for playback in the browser."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def wav_to_pcm16_mono(wav_bytes: bytes, target_rate: int = SAMPLE_RATE) -> bytes:
    """Convert a browser WAV recording to the PCM16 mono the API expects.

    Browsers pick their own sample rate and channel count, so downmix to mono
    and resample to the session rate. Linear interpolation is plenty for
    speech going into an ASR front end.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if width != 2:
        raise ValueError(f"expected 16-bit audio, got {width * 8}-bit")

    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if rate != target_rate:
        duration = len(samples) / rate
        target_len = int(duration * target_rate)
        samples = np.interp(
            np.linspace(0, len(samples), target_len, endpoint=False),
            np.arange(len(samples)),
            samples,
        )

    return samples.astype(np.int16).tobytes()


def pcm_seconds(pcm: bytes, rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / 2 / rate


def peak_level(pcm: bytes) -> float:
    """Peak amplitude as a fraction of full scale. Near zero means the mic
    recorded silence, which is worth catching before spending an API call."""
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16)
    return float(np.abs(samples).max()) / 32768.0
