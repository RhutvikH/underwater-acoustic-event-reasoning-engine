from __future__ import annotations

import numpy as np
import pytest

from uaere.math.bounds import gated_energy_strictly_less, gated_miss_bound
from uaere.math.dirichlet import (
    aleatoric_uncertainty,
    dirichlet_mean,
    epistemic_uncertainty,
    evidential_sse_loss,
    strengths_from_evidence,
)
from uaere.math.energy import expected_compute_energy
from uaere.math.metrics import brier, ece


def test_alpha_strictly_greater_than_one():
    ev = np.linspace(-5, 5, 7)
    a = strengths_from_evidence(ev)
    assert np.all(a > 1.0)


def test_dirichlet_mean_sums_to_one():
    a = np.array([2.0, 3.0, 5.0])
    p = dirichlet_mean(a)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0)


def test_epistemic_decreases_with_concentration():
    weak = epistemic_uncertainty(np.ones(5))
    strong = epistemic_uncertainty(np.ones(5) * 20)
    assert strong < weak


def test_ece_in_unit_interval():
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])
    val = ece(y, p)
    assert 0.0 <= val <= 1.0
    assert brier(y, p) >= 0.0


def test_gated_energy_proposition():
    assert gated_energy_strictly_less((1.0, 2.0, 3.0, 10.0), (0.4, 0.2, 0.1))
    assert expected_compute_energy((1, 2, 3, 10), (1, 1, 1)) == 16.0


def test_miss_bound_rejects_bad_mass():
    assert gated_miss_bound(0.2) == 0.2
    with pytest.raises(ValueError):
        gated_miss_bound(1.5)


def test_evidential_loss_finite():
    a = np.array([[2.0, 3.0, 2.0]])
    y = np.array([[0.0, 1.0, 0.0]])
    assert np.isfinite(evidential_sse_loss(a, y))
    _ = aleatoric_uncertainty(dirichlet_mean(a))
