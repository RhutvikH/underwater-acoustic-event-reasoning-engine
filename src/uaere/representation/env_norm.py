"""Environment-conditioned instance-norm + running noise-PSD EMA.

n̂_t = (1-α) n̂_{t-1} + α PSD(x_t)   on non-event frames
φ^{adapt} = IN(φ; γ(s), β(s))
"""

from __future__ import annotations

import numpy as np

from uaere.types import EnvironmentState, FloatArray


def film_from_state(state: EnvironmentState, n_ch: int) -> tuple[FloatArray, FloatArray]:
    """Tiny affine map γ(s), β(s). Closed form, no extra trained MLP required at L1."""
    v = state.as_vector()
    v = (v - np.array([12.0, 34.0, 40.0, 2.0, 10.0, 0.1, 1490.0])) / np.array(
        [8.0, 4.0, 40.0, 3.0, 8.0, 0.3, 40.0]
    )
    # 7 → n_ch via a fixed (seed-0 orthogonal-ish) projection
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.3, size=(n_ch, 7))
    gamma = 1.0 + 0.15 * np.tanh(w @ v)
    beta = 0.15 * np.tanh(w[::-1] @ v) if n_ch > 1 else 0.15 * np.tanh(w @ v)
    return gamma.astype(np.float64), beta.astype(np.float64)


class AdaptiveNormalizer:
    def __init__(self, n_bins: int, alpha: float = 0.05) -> None:
        self.alpha = float(alpha)
        self.noise_psd = np.zeros(n_bins, dtype=np.float64)
        self.initialized = False

    def update_noise(self, psd: FloatArray, is_event: bool) -> None:
        p = np.asarray(psd, dtype=np.float64)
        if not self.initialized:
            self.noise_psd = p.copy()
            self.initialized = True
            return
        if not is_event:
            a = self.alpha
            self.noise_psd = (1.0 - a) * self.noise_psd + a * p

    def normalize_mel(self, mel: FloatArray, state: EnvironmentState) -> FloatArray:
        x = np.asarray(mel, dtype=np.float64)
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-6
        x_n = (x - mu) / sd
        gamma, beta = film_from_state(state, x.shape[0])
        return gamma[:, None] * x_n + beta[:, None]

    def state_dict(self) -> dict:
        return {"noise_psd": self.noise_psd.tolist(), "initialized": self.initialized, "alpha": self.alpha}

    def load_state_dict(self, d: dict) -> None:
        self.noise_psd = np.asarray(d["noise_psd"], dtype=np.float64)
        self.initialized = bool(d["initialized"])
        self.alpha = float(d["alpha"])
