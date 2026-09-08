from __future__ import annotations

from uaere.models.logistic_mel import LogisticMelBackbone

BACKBONES = {
    "logistic_mel": LogisticMelBackbone,
}


def get_backbone(name: str, **kwargs):
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name}; known: {sorted(BACKBONES)}")
    return BACKBONES[name](**kwargs)
