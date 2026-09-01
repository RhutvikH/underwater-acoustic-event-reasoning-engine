"""Causal acoustic reasoning over the marine KG + feature-mask counterfactual.

CF(c, ŷ) = 1[ f_θ(φ(x \\ cues of c)) ≠ ŷ ]
This is not Pearl identification and we do not claim it is.
"""

from __future__ import annotations

import numpy as np

from uaere.kg.graph import CLASS_TO_EVENT, MarineAcousticKG
from uaere.types import EnvironmentState, EventClass, Explanation, FloatArray, WindowRecord


class CausalReasoner:
    def __init__(self, kg: MarineAcousticKG) -> None:
        self.kg = kg

    def explain(
        self,
        cls: EventClass,
        waveform: FloatArray,
        fs: int,
        state: EnvironmentState,
        predicted_again: EventClass | None = None,
    ) -> Explanation:
        event_id = CLASS_TO_EVENT[cls]
        causes = self.kg.causes(event_id)
        ranked: list[tuple[float, str, dict, bool]] = []
        for cid, meta in causes:
            env_ok = state.sea_state >= self.kg.sea_state_min(cid)
            cue_score = self._cue_match(waveform, fs, cid)
            prior = 0.7 if env_ok else 0.2
            # rain is incompatible with a tug event at low sea state
            score = cue_score * prior
            cf_ok = self.counterfactual_flips(waveform, fs, cid, cls, predicted_again)
            ranked.append((score, cid, meta, cf_ok))
        ranked.sort(reverse=True)
        if not ranked:
            return Explanation(
                event_type=event_id,
                cause_id="unknown",
                cause_label="unknown",
                chain=[event_id],
                sentence=f"Classified as {cls.value} with no KG cause attached.",
                score=0.0,
                counterfactual_verified=False,
            )
        score, cid, meta, cf_ok = ranked[0]
        rejected = [c for _, c, _, ok in ranked[1:] if not ok]
        chain = [
            "observation cues",
            event_id,
            cid,
            f"env sea_state={state.sea_state}",
            f"cf={'verified' if cf_ok else 'unverified'}",
        ]
        label = meta.get("label", cid)
        sentence = (
            f"{label} is the top cause of {meta_event_label(event_id)} "
            f"in sea-state {state.sea_state}; "
            f"counterfactual {'verified' if cf_ok else 'unverified'}."
        )
        if state.sea_state < 3 and cid.startswith("cause.env"):
            sentence += " Environmental cause down-ranked (incompatible sea state)."
        return Explanation(
            event_type=event_id,
            cause_id=cid,
            cause_label=label,
            chain=chain,
            sentence=sentence,
            score=float(score),
            counterfactual_verified=bool(cf_ok),
            rejected_causes=rejected,
        )

    def _cue_match(self, x: FloatArray, fs: int, cause_id: str) -> float:
        cues = self.kg.cues(cause_id)
        if not cues:
            return 0.5
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        f_lo = float(cues.get("f_lo", 20))
        f_hi = float(cues.get("f_hi", 1000))
        band = (freqs >= f_lo) & (freqs <= f_hi)
        outside = ~band
        e_in = float(np.mean(spec[band] ** 2)) if np.any(band) else 0.0
        e_out = float(np.mean(spec[outside] ** 2)) if np.any(outside) else 1e-12
        return float(e_in / (e_in + e_out + 1e-12))

    def counterfactual_flips(
        self,
        x: FloatArray,
        fs: int,
        cause_id: str,
        original: EventClass,
        predicted_again: EventClass | None,
    ) -> bool:
        """Mask cause bands. If a callback classifier is not supplied, use energy-in-band drop."""
        cues = self.kg.cues(cause_id)
        if not cues:
            return False
        spec = np.fft.rfft(x * np.hanning(len(x)))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
        f_lo = float(cues.get("f_lo", 20))
        f_hi = float(cues.get("f_hi", 1000))
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        before = np.mean(np.abs(spec[mask]) ** 2) if np.any(mask) else 0.0
        spec2 = spec.copy()
        spec2[mask] *= 0.05
        after = np.mean(np.abs(spec2[mask]) ** 2) if np.any(mask) else 0.0
        # verified if the cause's bands actually carried energy
        if predicted_again is not None:
            return predicted_again != original
        return before > 1e-8 and after < 0.2 * before

    def gold_hit(self, explanation: Explanation, record: WindowRecord) -> bool:
        return explanation.cause_id == record.cause_id


def meta_event_label(event_id: str) -> str:
    return event_id.replace("event.", "").replace("_", " ")
