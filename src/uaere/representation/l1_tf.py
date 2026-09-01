"""Log-Mel spectrogram. CQT is an ablation (see `log_cqt`)."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from uaere.types import HOP_LENGTH, N_FFT, N_MELS, FloatArray


def mel_filterbank(fs: int, n_fft: int = N_FFT, n_mels: int = N_MELS) -> FloatArray:
    def hz_to_mel(hz: FloatArray | float) -> FloatArray:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: FloatArray) -> FloatArray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_max = fs / 2.0
    m = np.linspace(hz_to_mel(0.0), hz_to_mel(f_max), n_mels + 2)
    h = mel_to_hz(m)
    bins = np.floor((n_fft + 1) * h / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        if centre == left:
            centre += 1
        if right == centre:
            right += 1
        right = min(right, fb.shape[1] - 1)
        centre = min(centre, right - 1)
        for j in range(left, centre):
            fb[i, j] = (j - left) / max(centre - left, 1)
        for j in range(centre, right):
            fb[i, j] = (right - j) / max(right - centre, 1)
    return fb


_FB_CACHE: dict[tuple[int, int, int], FloatArray] = {}


def log_mel(
    x: FloatArray,
    fs: int,
    n_fft: int = N_FFT,
    hop: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> FloatArray:
    key = (fs, n_fft, n_mels)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = mel_filterbank(fs, n_fft, n_mels)
    fb = _FB_CACHE[key]
    _, _, z = stft(
        np.asarray(x, dtype=np.float64),
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        window="hann",
        boundary=None,
        padded=False,
    )
    mag = np.abs(z)
    mel = fb @ mag
    return np.log(mel + 1e-6)


def log_cqt_proxy(x: FloatArray, fs: int, n_bins: int = N_MELS) -> FloatArray:
    """Geometric-spaced filterbank standing in for CQT (ablation only)."""
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fmin, fmax = 20.0, fs / 2.0
    edges = np.geomspace(fmin, fmax, n_bins + 1)
    out = np.zeros((n_bins, 1), dtype=np.float64)
    for i in range(n_bins):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i, 0] = np.log(np.mean(spec[m] ** 2) + 1e-6) if np.any(m) else -13.8
    return out
