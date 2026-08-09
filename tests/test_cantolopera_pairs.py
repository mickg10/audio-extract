import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_extract.cantolopera_pairs import (
    INVENTORY_SCHEMA,
    audit_inventory,
    audit_pair,
    load_completed_pairs,
    take_group_facts,
    write_audit,
)


def _sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog(opera="Madama Butterfly", title="Un bel dì vedremo"):
    return {
        "composer": "Giacomo Puccini",
        "opera": opera,
        "title": title,
        "slug": "un-bel-di-vedremo",
        "voices": ["Soprano"],
    }


def _row(root, pair_id, version, catalog, audio, *, state="complete"):
    path = root / f"{pair_id}_{version}.wav"
    sf.write(path, audio, 48_000, subtype="FLOAT")
    return {
        "schema": INVENTORY_SCHEMA,
        "storage_state": state,
        "pair_id": pair_id,
        "version": version,
        "catalog": catalog,
        "file": {
            "relative_path": f"{root.name}/{path.name}",
            "sha256": _sha(path),
        },
        "audio": {
            "frames": len(audio),
            "sample_rate_hz": 48_000,
            "channels": 2,
            "subtype": "FLOAT",
        },
    }


def _inventory(tmp_path, *, mismatch=False):
    root = tmp_path / "full_48khz_f32"
    root.mkdir()
    rng = np.random.default_rng(7)
    orchestra = rng.normal(0, 0.03, (4096, 2)).astype("float32")
    vocal = np.zeros_like(orchestra)
    vocal[1400:2600] = rng.normal(0, 0.02, (1200, 2))
    mixture = orchestra + vocal
    if mismatch:
        orchestra = orchestra[:-3]
    rows = [
        _row(root, "butterfly", "voice", _catalog(), mixture),
        _row(root, "butterfly", "orchestra", _catalog(), orchestra),
    ]
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return inventory, root


def test_senza_role_variants_share_take_group():
    ordinary = take_group_facts(_catalog())
    without_role = take_group_facts(
        _catalog(
            opera="Madama Butterfly completa ( senza Cio Cio San )",
            title="Un bel dì vedremo (senza soprano)",
        )
    )
    assert ordinary == without_role


def test_load_and_audit_exact_grid_pair(tmp_path):
    inventory, root = _inventory(tmp_path)
    pairs = load_completed_pairs(inventory, root)
    assert len(pairs) == 1
    result = audit_pair(
        pairs[0],
        probe_seconds=0.02,
        probe_count=5,
        max_shift_samples=16,
        null_block_seconds=0.005,
    )
    assert result["status"] == "audited"
    assert result["grid"]["valid"] is True
    assert result["grid"]["frame_count_match"] is True
    assert result["alignment"][
        "zero_lag_positive_polarity_no_swap_supported"
    ] is True
    assert result["suggested_grade"] == "strong_same_take_evidence"
    assert result["null_evidence"]["null_db_p10"] < -40


def test_inventory_root_name_does_not_bind_staging_directory_name(tmp_path):
    inventory, root = _inventory(tmp_path)
    staged = tmp_path / "staged-audio"
    root.rename(staged)
    pairs = load_completed_pairs(inventory, staged)
    assert Path(pairs[0]["files"]["voice"]["path"]).parent == staged


def test_grid_mismatch_is_explicitly_invalid(tmp_path):
    inventory, root = _inventory(tmp_path, mismatch=True)
    pair = load_completed_pairs(inventory, root)[0]
    result = audit_pair(pair, probe_seconds=0.01)
    assert result["status"] == "invalid_pair_evidence"
    assert result["grid"]["frame_count_match"] is False
    assert "pair sample grids differ" in result["errors"]


def test_container_hash_mismatch_is_not_silently_accepted(tmp_path):
    inventory, root = _inventory(tmp_path)
    pair = load_completed_pairs(inventory, root)[0]
    pair["files"]["voice"]["container_sha256"] = "sha256:" + "0" * 64
    result = audit_pair(pair, probe_seconds=0.01)
    assert result["status"] == "invalid_pair_evidence"
    assert "voice container SHA-256 mismatch" in result["errors"]


def test_nonzero_pair_delay_requires_alignment_review(tmp_path):
    inventory, root = _inventory(tmp_path)
    rows = [json.loads(line) for line in inventory.read_text().splitlines()]
    voice_path = root / "butterfly_voice.wav"
    mixture, rate = sf.read(voice_path, dtype="float32", always_2d=True)
    delayed = np.vstack([np.zeros((7, 2), dtype="float32"), mixture[:-7]])
    sf.write(voice_path, delayed, rate, subtype="FLOAT")
    rows[0]["file"]["sha256"] = _sha(voice_path)
    inventory.write_text("".join(json.dumps(row) + "\n" for row in rows))

    pair = load_completed_pairs(inventory, root)[0]
    result = audit_pair(
        pair,
        probe_seconds=0.02,
        probe_count=5,
        max_shift_samples=16,
        null_block_seconds=0.005,
    )
    assert result["status"] == "audited"
    assert result["suggested_grade"] == "needs_alignment_review"
    assert abs(result["alignment"]["median_delay_samples"]) >= 6


def test_inventory_summary_and_immutable_output(tmp_path):
    inventory, root = _inventory(tmp_path)
    rows, summary = audit_inventory(
        inventory,
        root,
        probe_seconds=0.02,
        probe_count=3,
        max_shift_samples=16,
        null_block_seconds=0.005,
    )
    assert summary["pairs"] == 1
    assert summary["take_groups"] == 1
    assert summary["grades"] == {"strong_same_take_evidence": 1}
    output = tmp_path / "audit.jsonl"
    summary_path = tmp_path / "summary.json"
    write_audit(rows, summary, output, summary_path)
    write_audit(rows, summary, output, summary_path)
    changed = dict(summary, pairs=2)
    try:
        write_audit(rows, changed, output, summary_path)
    except RuntimeError as exc:
        assert "refusing to rewrite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("differing audit output was overwritten")
