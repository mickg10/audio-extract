import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.classical_surrogate_alignment_v2 import SurrogateConfigV2
from audio_extract.classical_surrogate_replay_v2 import (
    ROW_SCHEMA,
    ReplayDecisionConfigV2,
    ReplayRowV2,
    SurrogateReplayV2Error,
    evaluate_replay_v2,
    load_rows_v2,
    row_sha256,
)


def _write(path: Path, audio: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype("float32"), 44_100, subtype="FLOAT")


def _record(path: Path):
    import hashlib

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return {
        "path": str(path),
        "container_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            audio, sr, ["FL", "FR"], len(audio)
        ),
        "frames": len(audio),
        "sample_rate_hz": sr,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
    }


def _data(tmp_path: Path, examples=("dev", "hold")):
    frames = 4096
    time = np.arange(frames, dtype=np.float64) / 44_100
    rows = []
    for example in examples:
        split = "development" if example == "dev" else "heldout"
        a = np.column_stack((
            0.3 * np.sin(2 * np.pi * 220 * time),
            0.25 * np.sin(2 * np.pi * 330 * time),
        ))
        v = np.column_stack((
            0.12 * np.sin(2 * np.pi * 610 * time),
            0.10 * np.sin(2 * np.pi * 680 * time),
        ))
        a_path = tmp_path / example / "A.wav"
        v_path = tmp_path / example / "V.wav"
        _write(a_path, a)
        _write(v_path, v)
        for step, scale in ((0, 0.50), (25, 0.65), (50, 0.80), (100, 1.00)):
            estimate_path = tmp_path / example / f"Vhat-{step}.wav"
            _write(estimate_path, scale * v)
            # Better recall has a lower/more-negative retained-voice dB metric.
            external_voice = -5.0 - 5.0 * scale
            row = {
                "schema": ROW_SCHEMA,
                "example_id": example,
                "split": split,
                "work_id": f"work-{example}",
                "session_id": f"session-{example}",
                "singer_id": f"singer-{example}",
                "checkpoint_step": step,
                "vocal_estimate": _record(estimate_path),
                "accompaniment_target": _record(a_path),
                "vocal_target": _record(v_path),
                "old_declared_total": 1.0 - 0.001 * step,
                "external_metrics": {
                    "retained_voice_db_p90": external_voice,
                    "event_hole_db_p90": 2.0,
                    "artifact_ratio_p90": 0.1,
                },
            }
            row["row_sha256"] = row_sha256(row)
            rows.append(row)
    return rows


def _manifest(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _surrogate_config():
    return SurrogateConfigV2(
        sample_rate_hz=44_100,
        tile_seconds=0.02,
        hop_seconds=0.01,
        tail_fraction=0.10,
    )


def _decision_config():
    return ReplayDecisionConfigV2(
        material_external_change_db=0.1,
        minimum_sign_concordance=0.75,
        minimum_macro_rank_correlation=-1e-9,
    )


def test_row_hash_excludes_only_itself_and_detects_mutation(tmp_path):
    row = _data(tmp_path, ("hold",))[0]
    parsed = ReplayRowV2.from_mapping(row)
    assert parsed.row_sha256 == row["row_sha256"]
    mutated = json.loads(json.dumps(row))
    mutated["checkpoint_step"] = 25
    with pytest.raises(ValueError, match="semantic hash mismatch"):
        ReplayRowV2.from_mapping(mutated)


def test_load_rows_rejects_duplicate_example_step(tmp_path):
    rows = _data(tmp_path, ("hold",))
    path = _manifest(tmp_path, rows + [rows[0]])
    with pytest.raises(SurrogateReplayV2Error, match="duplicate"):
        load_rows_v2(path)


def test_aligned_heldout_replay_passes(tmp_path):
    rows = load_rows_v2(_manifest(tmp_path, _data(tmp_path)))
    result = evaluate_replay_v2(
        rows,
        surrogate_config=_surrogate_config(),
        decision_config=_decision_config(),
    )
    assert result["decision"] == "SURROGATE_ALIGNED"
    assert result["summary"]["catastrophic_false_safe_pairs"] == []
    assert result["summary"]["coverage_failures"] == []
    assert result["summary"]["sign_concordance"]["recall"]["fraction"] == 1.0


def test_false_safe_step100_rejects(tmp_path):
    rows = _data(tmp_path)
    for row in rows:
        if row["split"] == "heldout" and row["checkpoint_step"] == 100:
            row["external_metrics"]["retained_voice_db_p90"] = -4.0
            row["row_sha256"] = row_sha256(row)
    result = evaluate_replay_v2(
        load_rows_v2(_manifest(tmp_path, rows)),
        surrogate_config=_surrogate_config(),
        decision_config=_decision_config(),
    )
    assert result["decision"] == "SURROGATE_REJECTED"
    assert any(
        "step-100" in failure for failure in result["summary"]["failures"]
    )


def test_no_heldout_group_is_invalid_evidence(tmp_path):
    rows = load_rows_v2(_manifest(tmp_path, _data(tmp_path, ("dev",))))
    result = evaluate_replay_v2(
        rows,
        surrogate_config=_surrogate_config(),
        decision_config=_decision_config(),
    )
    assert result["decision"] == "INVALID_EVIDENCE"


def test_changed_audio_container_is_refused(tmp_path):
    raw = _data(tmp_path, ("hold",))
    rows = load_rows_v2(_manifest(tmp_path, raw))
    path = Path(rows[0].vocal_estimate.path)
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio[10, 0] += 0.1
    sf.write(path, audio, sr, subtype="FLOAT")
    with pytest.raises(SurrogateReplayV2Error, match="container changed"):
        evaluate_replay_v2(
            rows,
            surrogate_config=_surrogate_config(),
            decision_config=_decision_config(),
        )


def test_target_identity_must_be_constant_across_checkpoints(tmp_path):
    raw = _data(tmp_path, ("hold",))
    wrong = tmp_path / "wrong-v.wav"
    audio, _ = sf.read(raw[-1]["vocal_target"]["path"], always_2d=True)
    _write(wrong, 0.9 * audio)
    raw[-1]["vocal_target"] = _record(wrong)
    raw[-1]["row_sha256"] = row_sha256(raw[-1])
    with pytest.raises(SurrogateReplayV2Error, match="targets changed"):
        evaluate_replay_v2(
            load_rows_v2(_manifest(tmp_path, raw)),
            surrogate_config=_surrogate_config(),
            decision_config=_decision_config(),
        )
