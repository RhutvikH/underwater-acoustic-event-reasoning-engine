"""Depthwise-separable tiny CNN in numpy. INT8 budget < 250 KB.

We own the IR so hardware MAC counts do not depend on TFLite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uaere.types import FloatArray


def _relu(x: FloatArray) -> FloatArray:
    return np.maximum(x, 0.0)


def _dw_conv(x: FloatArray, k: FloatArray) -> FloatArray:
    """x: (C, H, W), k: (C, kh, kw) depthwise, stride 1, valid-ish same pad."""
    c, h, w = x.shape
    kh, kw = k.shape[1], k.shape[2]
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (ph, ph), (pw, pw)))
    out = np.zeros_like(x)
    for ci in range(c):
        for i in range(h):
            for j in range(w):
                out[ci, i, j] = np.sum(xp[ci, i : i + kh, j : j + kw] * k[ci])
    return out


def _pw_conv(x: FloatArray, w: FloatArray, b: FloatArray) -> FloatArray:
    """w: (C_out, C_in)."""
    c_out, c_in = w.shape
    y = np.einsum("oi,ihw->ohw", w, x) + b[:, None, None]
    return y


@dataclass
class TinyCNN:
    c1: int = 8
    c2: int = 16
    n_classes: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        s = 0.2
        self.dw1 = rng.normal(0, s, size=(1, 3, 3)).repeat(self.c1, axis=0)
        # first pw expands from 1 → c1
        self.pw0 = rng.normal(0, s, size=(self.c1, 1))
        self.b0 = np.zeros(self.c1)
        self.dw1 = rng.normal(0, s, size=(self.c1, 3, 3))
        self.pw1 = rng.normal(0, s, size=(self.c1, self.c1))
        self.b1 = np.zeros(self.c1)
        self.dw2 = rng.normal(0, s, size=(self.c1, 3, 3))
        self.pw2 = rng.normal(0, s, size=(self.c2, self.c1))
        self.b2 = np.zeros(self.c2)
        self.fc_w = rng.normal(0, s, size=(self.n_classes, self.c2))
        self.fc_b = np.zeros(self.n_classes)
        self.ev_w = rng.normal(0, s, size=(self.n_classes, self.c2))
        self.ev_b = np.zeros(self.n_classes)

    def embed(self, mel: FloatArray) -> FloatArray:
        x = np.asarray(mel, dtype=np.float64)
        if x.ndim == 2:
            x = x[None, ...]
        x = _relu(_pw_conv(x, self.pw0, self.b0))
        x = _relu(_pw_conv(_dw_conv(x, self.dw1), self.pw1, self.b1))
        x = x[:, :, ::2]  # time pool
        x = _relu(_pw_conv(_dw_conv(x, self.dw2), self.pw2, self.b2))
        x = x[:, :, ::2]
        return x.mean(axis=(1, 2))  # (C2,)

    def logits(self, mel: FloatArray) -> FloatArray:
        h = self.embed(mel)
        return self.fc_w @ h + self.fc_b

    def evidence(self, mel: FloatArray) -> FloatArray:
        h = self.embed(mel)
        return self.ev_w @ h + self.ev_b

    def n_params(self) -> int:
        arrs = [
            self.pw0, self.b0, self.dw1, self.pw1, self.b1,
            self.dw2, self.pw2, self.b2, self.fc_w, self.fc_b, self.ev_w, self.ev_b,
        ]
        return int(sum(a.size for a in arrs))

    def int8_bytes(self) -> int:
        return self.n_params()  # 1 byte / param + we ignore 256-entry scales

    def parameters(self) -> list[np.ndarray]:
        return [
            self.pw0, self.b0, self.dw1, self.pw1, self.b1,
            self.dw2, self.pw2, self.b2, self.fc_w, self.fc_b, self.ev_w, self.ev_b,
        ]


def cnn_macs(mel_shape: tuple[int, int], c1: int = 8, c2: int = 16, n_classes: int = 5) -> int:
    h, w = mel_shape
    # pw0: c1 * 1 * h * w
    macs = c1 * h * w
    # dw1 3x3 + pw1
    macs += c1 * 9 * h * w + c1 * c1 * h * w
    w2 = (w + 1) // 2
    macs += c1 * 9 * h * w2 + c2 * c1 * h * w2
    w3 = (w2 + 1) // 2
    # fc + evidential
    macs += 2 * n_classes * c2
    _ = w3
    return int(macs)
