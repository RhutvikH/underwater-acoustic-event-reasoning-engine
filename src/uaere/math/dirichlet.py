"""Evidential deep learning identities (Sensoy, Kaplan, Kandemir, NeurIPS 2018).

α = f_θ(·) + 1 > 0
p̂ = α / Σα
u_epistemic = K / Σα
u_aleatoric = 1 - max_k p̂_k
"""

from __future__ import annotations

import numpy as np

from uaere.types import FloatArray


def strengths_from_evidence(evidence: FloatArray) -> FloatArray:
    """α = softplus(evidence) + 1, guaranteeing α_k > 1."""
    ev = np.asarray(evidence, dtype=np.float64)
    softplus = np.log1p(np.exp(-np.abs(ev))) + np.maximum(ev, 0.0)
    return softplus + 1.0


def dirichlet_mean(alpha: FloatArray) -> FloatArray:
    a = np.asarray(alpha, dtype=np.float64)
    s = np.sum(a, axis=-1, keepdims=True)
    return a / s


def epistemic_uncertainty(alpha: FloatArray) -> FloatArray:
    a = np.asarray(alpha, dtype=np.float64)
    k = a.shape[-1]
    s = np.sum(a, axis=-1)
    return k / s


def aleatoric_uncertainty(p: FloatArray) -> FloatArray:
    p = np.asarray(p, dtype=np.float64)
    return 1.0 - np.max(p, axis=-1)


def evidential_sse_loss(alpha: FloatArray, y_onehot: FloatArray) -> float:
    """Type-II SSE + KL-to-uniform regulariser (Sensoy et al.)."""
    a = np.asarray(alpha, dtype=np.float64)
    y = np.asarray(y_onehot, dtype=np.float64)
    s = np.sum(a, axis=-1, keepdims=True)
    p = a / s
    sse = np.sum((y - p) ** 2, axis=-1)
    var = p * (1.0 - p) / (s + 1.0)
    var_term = np.sum(var, axis=-1)
    # KL(Dir(α̃) || Dir(1)) with α̃ = y + (1-y) ⊙ α
    tilde = y + (1.0 - y) * a
    k = a.shape[-1]
    kl = _dirichlet_kl_uniform(tilde, k)
    return float(np.mean(sse + var_term + 0.1 * kl))


def _dirichlet_kl_uniform(alpha: FloatArray, k: int) -> FloatArray:
    a = np.asarray(alpha, dtype=np.float64)
    s = np.sum(a, axis=-1)
    # log Γ(s) - log Γ(K) - Σ log Γ(α) + Σ (α-1)(ψ(α)-ψ(s))
    from scipy.special import digamma, gammaln

    kl = gammaln(s) - gammaln(k) - np.sum(gammaln(a), axis=-1)
    kl += np.sum((a - 1.0) * (digamma(a) - digamma(s)[:, None]), axis=-1)
    return np.maximum(kl, 0.0)
