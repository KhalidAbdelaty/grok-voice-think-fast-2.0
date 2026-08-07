"""Offline checks for the two audio paths, no API key and no network needed.

Capture:  browser mic frames -> s16 mono at the session rate -> the socket.
Playback: model PCM -> the WebRTC source track -> the browser.

The interesting failure these catch is a silent one: a rate mismatch does not
raise, it just makes speech sound sped up and transcribe badly.

Run: python test_audio.py
"""
from __future__ import annotations

import fractions

import av
import numpy as np
from streamlit_webrtc import PcmAudioSource

from audio_utils import pcm_to_wav_bytes, pcm_seconds, peak_level, wav_to_pcm16_mono
from config import SAMPLE_RATE

failures: list[str] = []


def check(label: str, ok: bool, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + ("" if ok else f" -> {detail}"))
    if not ok:
        failures.append(label)


def tone(samples: int, rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, samples / rate, samples, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)


print("Playback path (model PCM -> browser):")
pcm = tone(SAMPLE_RATE).tobytes()
check("1 s of 24k PCM measures 1.0 s", abs(pcm_seconds(pcm) - 1.0) < 1e-9, pcm_seconds(pcm))

source = PcmAudioSource(sample_rate=SAMPLE_RATE)
source.push(pcm)
frame = source._source_callback(0, fractions.Fraction(1, SAMPLE_RATE))
check(
    "source track emits s16 mono at the session rate",
    frame.sample_rate == SAMPLE_RATE and frame.format.name == "s16" and frame.layout.name == "mono",
    f"{frame.sample_rate} {frame.format.name} {frame.layout.name}",
)
check("track carries the pushed audio", int(np.abs(frame.to_ndarray()).max()) > 0)

source.clear()
after = source._source_callback(1, fractions.Fraction(1, SAMPLE_RATE))
check("clear() silences the track, which is barge-in",
      int(np.abs(after.to_ndarray()).max()) == 0, np.abs(after.to_ndarray()).max())

print("\nCapture path (browser mic -> model):")
resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
produced = 0
for i in range(50):  # 50 frames of 20 ms = 1 second of 48 kHz stereo
    mono = tone(960, 48000)
    stereo = np.repeat(mono[None, :], 2, axis=0).reshape(1, -1)
    mic_frame = av.AudioFrame.from_ndarray(stereo, format="s16", layout="stereo")
    mic_frame.sample_rate = 48000
    mic_frame.pts = i * 960
    for out in resampler.resample(mic_frame):
        arr = out.to_ndarray()
        produced += arr.shape[1]
        if out.sample_rate != SAMPLE_RATE or out.layout.name != "mono":
            check("resampled frames are mono at the session rate", False,
                  f"{out.sample_rate} {out.layout.name}")
            break

check("1 s of 48k stereo becomes ~1 s of 24k mono",
      abs(produced - SAMPLE_RATE) < 250, f"{produced} samples")
check("no channel doubling", produced < SAMPLE_RATE * 1.5, f"{produced} samples")

print("\nWAV round trip:")
wav = pcm_to_wav_bytes(pcm)
back = wav_to_pcm16_mono(wav)
check("24k in, 24k out, byte-identical", back == pcm, f"{len(back)} vs {len(pcm)} bytes")
check("peak level preserved", abs(peak_level(back) - peak_level(pcm)) < 1e-6)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    raise SystemExit(1)
print("All audio checks passed.")
