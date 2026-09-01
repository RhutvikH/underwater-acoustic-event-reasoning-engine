from __future__ import annotations

import numpy as np

from uaere.twin.environment import EnvironmentModel, mackenzie_ssp, thorp_absorption_db_km
from uaere.twin.render import TwinRenderer


def test_mackenzie_ssp_reasonable():
    c = mackenzie_ssp(10.0, 35.0, 100.0)
    assert 1450 < c < 1550


def test_thorp_increases_with_frequency():
    assert thorp_absorption_db_km(10.0) > thorp_absorption_db_km(1.0)


def test_twin_render_deterministic():
    a = TwinRenderer("busy_strait", seed=7).render_dataset(6)
    b = TwinRenderer("busy_strait", seed=7).render_dataset(6)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x.waveform, y.waveform)
        assert x.event_class == y.event_class


def test_snr_not_nan():
    recs = TwinRenderer("calm_coastal", seed=1).render_dataset(8)
    for r in recs:
        assert np.isfinite(r.waveform).all()
        assert r.waveform.shape[0] == 16_000


def test_environment_model_clips_sea_state():
    s = EnvironmentModel().state(sea_state=99, turbulence=4)
    assert s.sea_state == 6
    assert 0.0 <= s.turbulence <= 1.0
