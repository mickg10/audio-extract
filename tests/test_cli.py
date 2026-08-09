import pytest

from audio_extract.cli import _candidate_role, _code_commit


def test_candidate_role_output_stem_semantics():
    # native_primary emits the target stem
    assert _candidate_role("native_primary", "instrumental") == "accompaniment"
    assert _candidate_role("native_primary", "vocals") == "vocal"
    # residual: mixture minus the target = the complement
    assert _candidate_role("mixture_minus_primary", "vocals") == "accompaniment"  # mix - vocals = instrumental
    assert _candidate_role("mixture_minus_primary", "instrumental") == "vocal"
    # native_secondary emits the other stem
    assert _candidate_role("native_secondary", "vocals") == "accompaniment"


def test_code_commit_accepts_exact_archive_override(monkeypatch):
    commit = "a" * 40
    monkeypatch.setenv("AUDIO_EXTRACT_CODE_COMMIT", commit)
    assert _code_commit() == commit


@pytest.mark.parametrize("value", ["abc", "A" * 40, "g" * 40, "a" * 39, "a" * 41])
def test_code_commit_rejects_invalid_archive_override(monkeypatch, value):
    monkeypatch.setenv("AUDIO_EXTRACT_CODE_COMMIT", value)
    with pytest.raises(RuntimeError, match="exact 40-character lowercase git SHA"):
        _code_commit()
