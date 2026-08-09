import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "package_baseline_delivery.py"
    spec = importlib.util.spec_from_file_location("package_baseline_delivery", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(name, voice_db, voice_coef, hole, hole_max, artifact=0.1, stereo=1.0):
    metrics = {metric: 0.1 for metric in _module().METRICS}
    metrics.update({
        "retained_voice_db_p90": voice_db,
        "retained_voice_coef_p90": voice_coef,
        "event_hole_db_p90": hole,
        "event_hole_db_max": hole_max,
        "artifact_ratio_p90": artifact,
        "stereo_width_dev_db/v2": stereo,
        "scale_dependent_sdr_db": 20.0,
        "_available_tiles": 10,
        "_total_tiles": 10,
    })
    return {"candidate": name, "metrics": metrics}


def test_rank_uses_near_critical_frontier_then_worst_event_tie_break():
    module = _module()
    rows = [
        _row("median", -20.00, 0.2, 0.50, 4.0),
        _row("geomedian", -19.99, 0.2, 0.30, 3.0),
        _row("mdx", -17.0, 0.2, 0.10, 2.0),
        _row("mean", -18.0, 0.2, 2.0, 5.0),
    ]
    ranked, policy = module._rank(rows)
    assert policy["qualification"] == "clears_all_frozen_caps"
    assert ranked[0]["candidate"] == "geomedian"
    assert ranked[1]["candidate"] == "median"


def test_rank_marks_best_available_when_every_candidate_misses_voice_cap():
    module = _module()
    rows = [
        _row("a", -20.0, 0.4, 0.2, 1.0),
        _row("b", -18.0, 0.5, 0.1, 1.0),
    ]
    _, policy = module._rank(rows)
    assert policy["qualification"] == "best_available_no_candidate_clears_all_frozen_caps"
