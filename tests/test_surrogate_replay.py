import pytest

from audio_extract.surrogate_replay import ReplayEvidenceError, comparison_summary


def _rows(*, false_safe=False):
    rows = []
    for sample, work in ((0, "work-a"), (1, "work-b")):
        for step in (0, 25, 50, 100):
            direction = (step / 100.0) * (1.0 if sample else 0.5)
            risk_delta = -direction
            external_delta = risk_delta
            if false_safe and sample == 0 and step == 100:
                external_delta = 1.0
            rows.append(
                {
                    "sample_index": sample,
                    "step": step,
                    "groups": {"work_id": work, "session_group_id": "session-" + work},
                    "surrogate": {
                        "recall_cvar": 1.0 + risk_delta,
                        "theft_cvar": 2.0 + risk_delta,
                        "artifact_cvar": 3.0 + risk_delta,
                        "source_coordinate_measure_fraction": 1.0,
                        "source_coordinate_tiles": 3,
                        "direct_fallback_tiles": 0,
                        "total_tiles": 3,
                    },
                    "external_metrics": {
                        "retained_voice_db_p90": -10.0 + external_delta,
                        "event_hole_db_p90": 2.0 + external_delta,
                        "artifact_ratio_p90": 0.2 + external_delta,
                    },
                    "old_declared_total": 1.0 - direction,
                }
            )
    return rows


def test_comparison_summary_accepts_aligned_whole_work_deltas():
    pairs, summary = comparison_summary(_rows())
    assert len(pairs) == 6
    assert summary["gate_result"] == "SURROGATE_ALIGNED"
    assert all(summary["checks"].values())
    assert summary["axis_summary"]["voice"]["crop_pair_sign_concordance"] == 1.0
    assert summary["axis_summary"]["holes"]["held_out_whole_work_rank"]["spearman_rho"] > 0


def test_comparison_summary_rejects_catastrophic_false_safe():
    pairs, summary = comparison_summary(_rows(false_safe=True))
    assert summary["gate_result"] == "SURROGATE_REJECTED"
    assert summary["catastrophic_false_safe_count"] == 2
    bad = [pair for pair in pairs if pair["sample_index"] == 0 and pair["step"] == 100][0]
    assert bad["catastrophic_false_safe_axes"] == ["voice", "holes"]


def test_comparison_summary_refuses_incomplete_checkpoint_grid():
    rows = _rows()
    rows.pop()
    with pytest.raises(ReplayEvidenceError, match="complete"):
        comparison_summary(rows)
