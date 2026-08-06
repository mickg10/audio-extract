from audio_extract.cli import _candidate_role


def test_candidate_role_output_stem_semantics():
    # native_primary emits the target stem
    assert _candidate_role("native_primary", "instrumental") == "accompaniment"
    assert _candidate_role("native_primary", "vocals") == "vocal"
    # residual: mixture minus the target = the complement
    assert _candidate_role("mixture_minus_primary", "vocals") == "accompaniment"  # mix - vocals = instrumental
    assert _candidate_role("mixture_minus_primary", "instrumental") == "vocal"
    # native_secondary emits the other stem
    assert _candidate_role("native_secondary", "vocals") == "accompaniment"
