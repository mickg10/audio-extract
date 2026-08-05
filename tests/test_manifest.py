import pytest

from audio_extract.manifest import Manifest


def test_candidate_round_trip(tmp_path):
    with Manifest(tmp_path / "m.sqlite") as man:
        rec = {
            "recipe_id": "sha256:abc",
            "operation": "mixture_minus_source",
            "parents": ["sha256:in", "sha256:voc"],
            "artifact_pcm_sha256": "sha256:pcm",
            "sample_rate_hz": 44100,
            "channels": ["FL", "FR"],
            "frames": 5644800,
            "sample_format": "float32",
            "status": "complete",
            "created_at": "2026-08-05T00:00:00Z",
        }
        man.upsert_candidate(rec)
        got = man.get_candidate("sha256:abc")
    assert got is not None
    assert got["parents"] == ["sha256:in", "sha256:voc"]
    assert got["channels"] == ["FL", "FR"]
    assert got["frames"] == 5644800


def test_upsert_is_idempotent(tmp_path):
    with Manifest(tmp_path / "m.sqlite") as man:
        rec = {"recipe_id": "sha256:x", "operation": "separate", "parents": [], "status": "complete"}
        man.upsert_candidate(rec)
        man.upsert_candidate(rec)
        assert man.list_candidates() == ["sha256:x"]


def test_state_validation(tmp_path):
    with Manifest(tmp_path / "m.sqlite") as man:
        man.set_state("trk", "INGESTED")
        assert man.get_state("trk")["state"] == "INGESTED"
        with pytest.raises(ValueError):
            man.set_state("trk", "NOT_A_STATE")


def test_passage_and_metric(tmp_path):
    with Manifest(tmp_path / "m.sqlite") as man:
        man.upsert_passages([
            {
                "passage_id": "p_0007",
                "start_sample": 8123400,
                "end_sample": 8652600,
                "tags": ["high_soprano", "hall_tail"],
                "features": {"median_f0_hz": 661.2},
                "detectors": {"pitch": "pesto@1"},
            }
        ])
        man.upsert_metric({
            "recipe_id": "sha256:abc",
            "passage_id": "p_0007",
            "metric": "pump_depth_multiband/v1",
            "value": 1.18,
            "unit": "dB",
            "details": {"worst_band_hz": [2000, 5000]},
        })
    # no exception == schema + inserts are sound
