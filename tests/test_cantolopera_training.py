import json

from audio_extract.cantolopera_pairs import PAIR_SCHEMA
from audio_extract.cantolopera_training import (
    TASK,
    assess_pair,
    build_selection,
    write_outputs,
)


def _row(index=0, *, opera=None, title=None, delay=7, swap=False, null=-25.0):
    pair_id = f"pair-{index}"
    return {
        "schema": PAIR_SCHEMA,
        "pair_id": pair_id,
        "take_group_id": f"cantolopera:take:{index:024x}",
        "catalog": {
            "composer": f"Composer {index}",
            "opera": opera or f"Opera {index}",
            "title": title or f"Title {index}",
            "slug": pair_id,
        },
        "status": "audited",
        "suggested_grade": "needs_alignment_review",
        "errors": [],
        "files": {
            "voice": {
                "path": f"/source/{pair_id}_voice.wav",
                "container_sha256": "sha256:" + f"{index + 1:064x}",
                "inventory_audio": {"frames": 480_000},
            },
            "orchestra": {
                "path": f"/source/{pair_id}_orchestra.wav",
                "container_sha256": "sha256:" + f"{index + 1000:064x}",
                "inventory_audio": {"frames": 480_000},
            },
        },
        "audio": {
            "voice": {"frames": 480_000},
            "orchestra": {"frames": 480_000},
        },
        "alignment": {
            "probes": [
                {
                    "delay_samples": delay + (probe == 4),
                    "fractional_samples": 0.125,
                    "polarity": 1,
                    "channel_swap": swap,
                    "confidence": 0.9,
                }
                for probe in range(5)
            ]
        },
        "null_evidence": {
            "null_db_p10": null,
            "null_db_p50": -8.0,
            "best_quintile_gain_median": 0.75,
            "best_quintile_gain_p10": 0.72,
            "best_quintile_gain_p90": 0.78,
        },
    }


def test_tier_a_selection_freezes_alignment_and_all_voice_task():
    result = assess_pair(_row())
    assert result["cohort"] == "tier_a"
    assert result["eligible"] is True
    assert result["reasons"] == []
    assert result["transform"]["mixture_to_orchestra"] == {
        "delay_samples": 7,
        "fractional_samples": "0.125",
        "polarity": 1,
        "channel_swap": False,
    }
    assert result["transform"]["orchestra_gain"] == "0.75"


def test_weak_null_is_tier_b_but_not_initially_eligible():
    result = assess_pair(_row(null=-12.0))
    assert result["cohort"] == "tier_b"
    assert result["eligible"] is False
    assert result["reasons"] == ["null_evidence_below_tier_a"]


def test_channel_swap_or_identical_pair_is_excluded():
    swapped = assess_pair(_row(swap=True))
    assert swapped["cohort"] == "excluded"
    assert "channel_order_not_consistently_straight" in swapped["reasons"]

    row = _row()
    row["files"]["orchestra"]["container_sha256"] = row["files"]["voice"][
        "container_sha256"
    ]
    identical = assess_pair(row)
    assert identical["cohort"] == "excluded"
    assert "identical_voice_and_orchestra_control" in identical["reasons"]


def test_connected_variants_never_cross_component_split():
    rows = [_row(index) for index in range(10)]
    # These look like separate catalogue rows but are linked by composer/opera.
    rows[1]["catalog"].update(rows[0]["catalog"] | {"title": "Other number"})
    decisions, manifest, summary, splits = build_selection(
        rows, audit_sha256="sha256:" + "a" * 64,
        selector_code_commit="1" * 40,
    )
    assert len(decisions) == len(manifest) == 10
    by_work = {row["work_id"]: row for row in manifest}
    assert by_work["pair-0"]["group_id"] == by_work["pair-1"]["group_id"]
    assert by_work["pair-0"]["split"] == by_work["pair-1"]["split"]
    assert set(summary["split_components"]) == {"train", "val"}
    assert sum(summary["split_components"].values()) == 9
    assert summary["task"] == TASK
    assert splits["group_split"][by_work["pair-0"]["group_id"]] == by_work[
        "pair-0"
    ]["split"]


def test_selection_outputs_are_immutable(tmp_path):
    outputs = build_selection(
        [_row(index) for index in range(5)],
        audit_sha256="sha256:" + "b" * 64,
        selector_code_commit="2" * 40,
    )
    paths = {
        "selection_path": tmp_path / "selection.jsonl",
        "manifest_path": tmp_path / "manifest.jsonl",
        "summary_path": tmp_path / "summary.json",
        "splits_path": tmp_path / "splits.json",
    }
    write_outputs(*outputs, **paths)
    write_outputs(*outputs, **paths)
    assert len(paths["selection_path"].read_text().splitlines()) == 5
    changed = list(outputs)
    changed[2] = dict(changed[2], rows=6)
    try:
        write_outputs(*changed, **paths)
    except RuntimeError as exc:
        assert "refusing to rewrite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("differing selection summary was overwritten")
