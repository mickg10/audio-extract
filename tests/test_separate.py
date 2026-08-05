import numpy as np

from audio_extract.separate import _residual_primary


def test_residual_primary_prefers_vocals_no_truthiness_error():
    # Regression: `_stem("vocals") or _stem("instrumental")` raised
    # "truth value of an array is ambiguous" for multi-element stems.
    v = np.zeros((100, 2))
    i = np.ones((100, 2))
    assert _residual_primary({"vocals": v, "instrumental": i}) is v
    assert _residual_primary({"instrumental": i}) is i
    assert _residual_primary({}) is None
