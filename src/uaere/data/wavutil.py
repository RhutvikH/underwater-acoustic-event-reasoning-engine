"""Stdlib WAV I/O. Fixtures stay int16; pipeline works in float64 ∈ [-1, 1]."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from uaere.types import FloatArray


def write_wav(path: str | Path, x: FloatArray, fs: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.asarray(x), -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[FloatArray, int]:
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    x = pcm.astype(np.float64) / 32768.0
    return x, fs


def resample_linear(x: FloatArray, fs_in: int, fs_out: int) -> FloatArray:
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.linspace(0.0, 1.0, len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float64)
