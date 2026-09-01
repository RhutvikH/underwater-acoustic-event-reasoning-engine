"""Single place for RNG. No silent randomness is allowed in the pipeline."""

from __future__ import annotations

import os

import numpy as np


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def seed_everything(seed: int) -> np.random.Generator:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    return rng(seed)
