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


def _wav_format_tag(path: Path) -> int | None:
    """PCM is 1. IEEE float is 3. Extensible is 0xFFFE. None if this is not RIFF."""
    try:
        with path.open("rb") as f:
            hdr = f.read(12)
            if len(hdr) < 12 or hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
                return None
            while True:
                chunk = f.read(8)
                if len(chunk) < 8:
                    return None
                cid, size = chunk[:4], int.from_bytes(chunk[4:8], "little")
                payload = f.read(size)
                if cid == b"fmt " and len(payload) >= 2:
                    return int.from_bytes(payload[:2], "little")
                if size & 1:
                    f.read(1)
    except OSError:
        return None
    return None


def read_wav_centre(
    path: str | Path,
    seconds: float = 1.0,
    pad_s: float = 0.0,
) -> tuple[FloatArray, int]:
    """Centre ``seconds`` of audio, plus ``pad_s`` on each side.

    16-bit and 32-bit PCM seek to that window. A long Watkins cut is not
    decoded in full. Other encodings fall back to :func:`read_wav`.
    """
    path = Path(path)
    fmt = _wav_format_tag(path)
    if fmt == 1:
        try:
            with wave.open(str(path), "rb") as handle:
                fs = int(handle.getframerate())
                n = int(handle.getnframes())
                ch = int(handle.getnchannels())
                sw = int(handle.getsampwidth())
                if fs <= 0 or n <= 0 or sw not in (2, 4) or ch < 1:
                    raise wave.Error("unsupported PCM layout")
                need = int(round(fs * (float(seconds) + 2.0 * float(pad_s))))
                if need <= 0 or n <= need:
                    start = 0
                    count = n
                else:
                    start = (n - need) // 2
                    count = need
                handle.setpos(int(start))
                raw = handle.readframes(int(count))
            if sw == 2:
                pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
            else:
                pcm = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
            if ch > 1:
                pcm = pcm.reshape(-1, ch).mean(axis=1)
            return np.asarray(pcm, dtype=np.float64), fs
        except (wave.Error, EOFError, ValueError):
            pass
    x, fs = read_wav(path)
    need = int(round(fs * (float(seconds) + 2.0 * float(pad_s))))
    if need > 0 and len(x) > need:
        start = (len(x) - need) // 2
        x = np.asarray(x[start : start + need], dtype=np.float64)
    return x, fs


def read_wav(path: str | Path) -> tuple[FloatArray, int]:
    """Read a mono waveform in float64.

    16-bit PCM stays on the stdlib path (divide by 32768). 32-bit integer
    PCM is full-scale normalised. IEEE-float, including pressure-corrected
    ToadFishFinder clips stored in µPa, is kept as stored so a later energy
    baseline still sees relative level. The playback copy is peak-limited
    separately in ``wav_bytes``.
    """
    path = Path(path)
    fmt = _wav_format_tag(path)
    # Only uncompressed PCM may be interpreted as integers. Float WAVs that
    # a future stdlib accepts must not be divided as if they were int32.
    if fmt in (None, 1):
        try:
            with wave.open(str(path), "rb") as w:
                fs = w.getframerate()
                n = w.getnframes()
                ch = w.getnchannels()
                sw = w.getsampwidth()
                raw = w.readframes(n)
            if sw == 2:
                pcm = np.frombuffer(raw, dtype="<i2")
                if ch > 1:
                    pcm = pcm.reshape(-1, ch).mean(axis=1)
                return pcm.astype(np.float64) / 32768.0, fs
            if sw == 4 and fmt == 1 and len(raw) == n * ch * 4:
                pcm = np.frombuffer(raw, dtype="<i4")
                if ch > 1:
                    pcm = pcm.reshape(-1, ch).mean(axis=1)
                return pcm.astype(np.float64) / 2147483648.0, fs
        except (wave.Error, EOFError):
            pass
    from scipy.io import wavfile

    fs, data = wavfile.read(path)
    x = np.asarray(data)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float64), int(fs)
    if x.dtype == np.int16:
        return x.astype(np.float64) / 32768.0, int(fs)
    if x.dtype == np.int32:
        return x.astype(np.float64) / 2147483648.0, int(fs)
    if x.dtype == np.uint8:
        return (x.astype(np.float64) - 128.0) / 128.0, int(fs)
    scale = float(np.max(np.abs(x.astype(np.float64))) + 1e-12)
    return x.astype(np.float64) / scale, int(fs)


def wav_bytes(x: FloatArray, fs: int) -> bytes:
    """In-memory 16-bit WAV so the demo can hand a window to a browser."""
    import io

    y = np.clip(np.asarray(x), -1.0, 1.0)
    # µPa-scale floats are not in [-1, 1]. Peak-limit only the playback copy.
    peak = float(np.max(np.abs(y))) if len(y) else 1.0
    if peak > 1.0:
        y = y / peak
    pcm = (y * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def resample_linear(x: FloatArray, fs_in: int, fs_out: int) -> FloatArray:
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.linspace(0.0, 1.0, len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float64)
