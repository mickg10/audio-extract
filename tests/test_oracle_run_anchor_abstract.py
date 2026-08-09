from audio_extract.oracle_run_anchor_abstract import (
    AnchorDecision,
    AnchorEvidence,
    decide_anchor,
    exhaustive_anchor_evidence,
)


def test_exhaustive_anchor_state_space_is_fail_closed():
    visited = 0
    verified = 0
    for evidence in exhaustive_anchor_evidence():
        visited += 1
        decision = decide_anchor(evidence)
        identities = (
            evidence.prepared_digest,
            evidence.report_digest,
            evidence.reopened_digest,
        )
        same = (
            all(value is not None for value in identities)
            and len(set(identities)) == 1
        )
        facts = (
            evidence.commit_matches,
            evidence.clean_code_state,
            evidence.works_match,
            evidence.run_config_matches,
            evidence.policy_matches,
            evidence.source_groups_valid,
            evidence.global_basis_matches,
            evidence.truth_manifest_matches,
            evidence.audits_match,
        )
        if decision is AnchorDecision.VERIFIED:
            verified += 1
            assert same
            assert all(facts)
        else:
            assert not same or not all(facts)

    assert visited == 3 ** 3 * 2 ** 9
    # Two possible concrete digests (A/B) with every semantic fact true.
    assert verified == 2


def _valid() -> AnchorEvidence:
    return AnchorEvidence(
        prepared_digest="A",
        report_digest="A",
        reopened_digest="A",
        commit_matches=True,
        clean_code_state=True,
        works_match=True,
        run_config_matches=True,
        policy_matches=True,
        source_groups_valid=True,
        global_basis_matches=True,
        truth_manifest_matches=True,
        audits_match=True,
    )


def test_fully_bound_clean_evidence_verifies():
    assert decide_anchor(_valid()) is AnchorDecision.VERIFIED


def test_posthoc_or_mutated_input_is_rejected():
    value = _valid()
    mutated = AnchorEvidence(
        **{
            **value.__dict__,
            "reopened_digest": "B",
        }
    )
    assert decide_anchor(mutated) is AnchorDecision.REJECTED


def test_report_cannot_name_a_different_input_record():
    value = _valid()
    changed = AnchorEvidence(
        **{
            **value.__dict__,
            "report_digest": "B",
        }
    )
    assert decide_anchor(changed) is AnchorDecision.REJECTED


def test_missing_preparation_is_rejected_even_when_other_facts_pass():
    value = _valid()
    changed = AnchorEvidence(
        **{
            **value.__dict__,
            "prepared_digest": None,
        }
    )
    assert decide_anchor(changed) is AnchorDecision.REJECTED


def test_each_semantic_fact_is_independently_required():
    value = _valid()
    for name in (
        "commit_matches",
        "clean_code_state",
        "works_match",
        "run_config_matches",
        "policy_matches",
        "source_groups_valid",
        "global_basis_matches",
        "truth_manifest_matches",
        "audits_match",
    ):
        changed = AnchorEvidence(
            **{
                **value.__dict__,
                name: False,
            }
        )
        assert decide_anchor(changed) is AnchorDecision.REJECTED
