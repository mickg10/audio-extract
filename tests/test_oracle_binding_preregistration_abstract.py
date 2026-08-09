from audio_extract.oracle_binding_preregistration_abstract import (
    Decision,
    State,
    decide,
    exhaustive_states,
)


def test_all_131072_preregistration_states_satisfy_promotion_invariants():
    count = 0
    decisions = set()
    for state in exhaustive_states():
        count += 1
        decision = decide(state)
        decisions.add(decision)

        if decision is Decision.ACTIONABLE:
            assert state.preregistration_valid
            assert state.all_method_resolution_evidence_valid
            assert state.selected_primary_passes

        if decision is Decision.RESOLUTION_SENSITIVE:
            assert state.preregistration_valid
            assert state.all_method_resolution_evidence_valid
            assert not state.selected_primary_passes
            assert state.selected_sensitivity_passes

        if decision is Decision.NO_ACTIONABLE_GAP:
            assert state.preregistration_valid
            assert state.all_method_resolution_evidence_valid
            assert not state.selected_primary_passes
            assert not state.selected_sensitivity_passes

        if not state.complete:
            assert decision is Decision.INVALID_EVIDENCE

    assert count == 131_072
    assert decisions == set(Decision)


def _valid(**changes):
    values = {
        "policy_pinned": True,
        "exact_resolutions": True,
        "voiced_nonempty": True,
        "no_vocal_nonempty": True,
        "manifests_physically_distinct": True,
        "witness_preexists": True,
        "report_binds_witness": True,
        "run_input_preexists": True,
        "claim_preexists": True,
        "external_anchor_valid": True,
        "digest_bindings_valid": True,
        "output_absent_through_preflight": True,
        "preflight_valid": True,
        "report_binds_run_input_claim": True,
        "all_method_resolution_evidence_valid": True,
        "selected_primary_passes": True,
        "selected_sensitivity_passes": False,
    }
    values.update(changes)
    return State(**values)


def test_rounded_resolution_cannot_promote():
    assert decide(_valid(exact_resolutions=False)) is Decision.INVALID_EVIDENCE


def test_hardlink_alias_cannot_promote():
    assert (
        decide(_valid(manifests_physically_distinct=False)) is Decision.INVALID_EVIDENCE
    )


def test_posthoc_or_unbound_witness_cannot_promote():
    assert decide(_valid(witness_preexists=False)) is Decision.INVALID_EVIDENCE
    assert decide(_valid(report_binds_witness=False)) is Decision.INVALID_EVIDENCE


def test_missing_v3_input_or_external_claim_cannot_promote():
    assert decide(_valid(run_input_preexists=False)) is Decision.INVALID_EVIDENCE
    assert decide(_valid(claim_preexists=False)) is Decision.INVALID_EVIDENCE
    assert decide(_valid(external_anchor_valid=False)) is Decision.INVALID_EVIDENCE


def test_invalid_sidecars_dependencies_or_chronology_cannot_promote():
    assert decide(_valid(digest_bindings_valid=False)) is Decision.INVALID_EVIDENCE
    assert (
        decide(_valid(output_absent_through_preflight=False))
        is Decision.INVALID_EVIDENCE
    )


def test_missing_preflight_or_report_v3_binding_cannot_promote():
    assert decide(_valid(preflight_valid=False)) is Decision.INVALID_EVIDENCE
    assert (
        decide(_valid(report_binds_run_input_claim=False)) is Decision.INVALID_EVIDENCE
    )


def test_sensitivity_only_success_is_nonpromoting():
    state = _valid(
        selected_primary_passes=False,
        selected_sensitivity_passes=True,
    )
    assert decide(state) is Decision.RESOLUTION_SENSITIVE
