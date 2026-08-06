import numpy as np
import pytest
import soundfile as sf

from audio_extract import identity
from audio_extract.manifest_v2 import ImmutableFactError, ManifestV2
from audio_extract.storage import ImmutableWriteError, TrackLayout


def _node(mid="sha256:n1", **kw):
    base = dict(node_id=mid, node_type="candidate", role="accompaniment",
                cohort="instrumental-production-v1", operation="separate",
                parents=["sha256:src"], recipe={"schema": "x"}, status="complete",
                created_at="2026-08-06T00:00:00Z")
    base.update(kw)
    return base


def test_immutable_fact_semantics(tmp_path):
    with ManifestV2(tmp_path / "m.sqlite") as m:
        m.add_artifact_node(**_node())
        m.add_artifact_node(**_node())                      # identical re-insert: no-op
        with pytest.raises(ImmutableFactError):
            m.add_artifact_node(**_node(status="failed"))    # differing fact: refused


def test_role_and_cohort_filtering(tmp_path):
    with ManifestV2(tmp_path / "m.sqlite") as m:
        m.add_artifact_node(**_node("sha256:a"))
        m.add_artifact_node(**_node("sha256:v", role="vocal", cohort="vocal-evaluation-v1"))
        prod = m.nodes_in_cohort("instrumental-production-v1", role="accompaniment")
        assert [n["node_id"] for n in prod] == ["sha256:a"]
        with pytest.raises(ValueError):
            m.add_artifact_node(**_node("sha256:x", role="nonsense"))


def test_observation_availability(tmp_path):
    with ManifestV2(tmp_path / "m.sqlite") as m:
        m.record_observation(node_id="sha256:a", scope_id="p_0001", metric="leakage/v2",
                             value=None, available=False)
        m.record_observation(node_id="sha256:a", scope_id="p_0001", metric="holes/v2",
                             value=1.5, available=True, uncertainty=0.2)
        assert len(m.observations("sha256:a")) == 2
        avail = m.observations("sha256:a", only_available=True)
        assert len(avail) == 1 and avail[0]["metric"] == "holes/v2"


def test_run_state_preserves_session_and_budgets(tmp_path):
    with ManifestV2(tmp_path / "m.sqlite") as m:
        m.set_run_state("trk", "SCREENING", pi_session="sess-1", budgets={"rounds": 1})
        m.set_run_state("trk", "MEASURING")                 # no session passed
        st = m.run_state("trk")
        assert st["state"] == "MEASURING"
        assert st["pi_session"] == "sess-1"                 # preserved, not erased
        assert st["budgets"] == {"rounds": 1}               # preserved


def test_challenge_and_decision_records(tmp_path):
    with ManifestV2(tmp_path / "m.sqlite") as m:
        m.add_challenge_case(challenge_id="ch1", challenge_type="track_remix",
                             mixture_node_id="sha256:mix", target_node_id="sha256:tgt",
                             recipe={"g_db": -6}, klass={"pitch": "high"})
        m.add_challenge_result(challenge_id="ch1", candidate_recipe_id="sha256:a",
                               result={"target_error_db": 3.2})
        m.add_selection_decision(decision_id="d1", status="final", selector_version="v1",
                                 report={"risk": 0.4}, created_at="t",
                                 candidate_recipe_id="sha256:a")
        m.record_action(action_id="act1", round_no=0, proposed={"type": "change_overlap"},
                        status="executed")
        assert m.challenge_cases("track_remix")[0]["challenge_id"] == "ch1"
        assert m.action_counts() == {"executed": 1}


def test_atomic_candidate_write(tmp_path):
    layout = TrackLayout(tmp_path, "trk").ensure()
    arr = (np.random.default_rng(0).standard_normal((1000, 2)) * 0.1).astype(np.float32)
    wav = tmp_path / "staged.wav"
    sf.write(str(wav), arr, 44100, subtype="FLOAT")
    pcm = identity.artifact_pcm_sha256(arr, 44100, ["FL", "FR"], 1000)

    cdir = layout.write_candidate("sha256:c1", {"r": 1}, {"e": 1}, wav, pcm)
    assert (cdir / "COMPLETE").exists() and (cdir / "output.f32.wav").exists()
    assert not any(p.name.startswith(".tmp-") for p in layout.candidates_dir.iterdir())

    # append-only: second completed write refused
    sf.write(str(wav), arr, 44100, subtype="FLOAT")
    with pytest.raises(ImmutableWriteError):
        layout.write_candidate("sha256:c1", {"r": 1}, {"e": 1}, wav, pcm)


def test_atomic_write_rejects_hash_mismatch(tmp_path):
    layout = TrackLayout(tmp_path, "trk").ensure()
    arr = (np.random.default_rng(1).standard_normal((500, 2)) * 0.1).astype(np.float32)
    wav = tmp_path / "staged.wav"
    sf.write(str(wav), arr, 44100, subtype="FLOAT")
    with pytest.raises(ImmutableWriteError):
        layout.write_candidate("sha256:c2", {}, {}, wav, "sha256:" + "0" * 64)
    # failed write leaves NO final dir and no temp litter
    assert not layout.candidate_dir("sha256:c2").exists()
    assert not any(p.name.startswith(".tmp-") for p in layout.candidates_dir.iterdir())
