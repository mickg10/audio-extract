"""Smoke test for the judge training pipeline: build examples from known-truth cases
across several synthetic 'works', assemble, and train the LOWO baseline. Proves the
feature+label+train wiring end-to-end; real training runs on the GPU with separators."""
import numpy as np
import pytest

from audio_extract import fixtures as fx

SR = 44100
jt = pytest.importorskip("audio_extract.judge_train")   # skip cleanly if sklearn absent


def _case(seed, quality):
    """A synthetic (M,A,V,Y) case. quality in [0,1]: 1=perfect instrumental, 0=gouged+leaky."""
    rng = np.random.default_rng(seed)
    A = fx.synth_orchestra(SR, 2.5)
    V, _ = fx.synth_vocal(SR, 2.5)
    n = min(len(A), len(V)); A, V = A[:n], V[:n]
    M = A + V
    # candidate: perfect A at quality=1; increasingly holed + voice-leaky as quality->0
    hole = fx.spectral_hole(A, SR, 2000, 4000, (1 - quality) * 14.0)
    Y = hole + (1 - quality) * 0.4 * V
    return M, A, V, Y


def test_build_example_shapes_and_targets():
    M, A, V, Y = _case(0, 1.0)
    x, tgt = jt.build_example(M, A, V, Y, SR)
    assert x.ndim == 1 and np.all(np.isfinite(x))
    assert tgt["_available_tiles"] > 0
    assert set(jt.TARGETS) <= set(tgt)


def test_targets_track_quality():
    # a gouged/leaky candidate must have higher hole + retained-voice targets than a clean one
    _, tgt_good = (lambda c: jt.build_example(*c, SR))(_case(1, 1.0)[:1] + _case(1, 1.0)[1:])
    xg, tg = jt.build_example(*_case(1, 1.0), SR)
    xb, tb = jt.build_example(*_case(1, 0.0), SR)
    assert tb["event_hole_db_p90"] >= tg["event_hole_db_p90"]
    assert tb["retained_voice_db_p90"] >= tg["retained_voice_db_p90"] - 1e-6


def test_assemble_and_train_lowo():
    # 6 works x 2 quality levels -> LOWO baseline trains and reports per-head MAE
    examples = []
    for w in range(6):
        for q in (0.2, 0.9):
            M, A, V, Y = _case(w * 10 + int(q * 10), q)
            x, tgt = jt.build_example(M, A, V, Y, SR)
            examples.append((x, tgt, f"work{w}"))
    out = jt.assemble_and_train(examples)
    rep = out["report"]
    assert rep["n_examples"] == 12 and rep["n_groups"] == 6
    assert "event_hole_db_p90" in rep["heads"]
    # a fitted model exists for at least the hole head
    assert "event_hole_db_p90" in out["models"]
    assert rep["heads"]["event_hole_db_p90"]["lowo_mae"] is not None
