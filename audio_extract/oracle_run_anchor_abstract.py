"""Finite mirror of the pre-execution routing run-input anchor.

This abstraction is intentionally smaller than the filesystem implementation.
It models the semantic facts that the concrete producer/verifier must establish:

* one input digest was prepared before the run;
* the report embeds that exact digest;
* the verifier reopens bytes with the same digest;
* commit, work order, run config, policy, source groups, global basis, truth
  manifest, and audit identities all match;
* a dirty or falsely claimed code state cannot verify.

It provides an ordinary-CI mirror for the accompanying TLA+ lifecycle model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterable


class AnchorDecision(str, Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class AnchorEvidence:
    prepared_digest: str | None
    report_digest: str | None
    reopened_digest: str | None
    commit_matches: bool
    clean_code_state: bool
    works_match: bool
    run_config_matches: bool
    policy_matches: bool
    source_groups_valid: bool
    global_basis_matches: bool
    truth_manifest_matches: bool
    audits_match: bool

    def validate(self) -> None:
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
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in (
            "prepared_digest",
            "report_digest",
            "reopened_digest",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value
            ):
                raise TypeError(f"{name} must be a non-empty string or None")


def decide_anchor(evidence: AnchorEvidence) -> AnchorDecision:
    """Apply the fail-closed semantic run-input rule."""

    evidence.validate()
    identities = (
        evidence.prepared_digest,
        evidence.report_digest,
        evidence.reopened_digest,
    )
    same_nonempty_digest = (
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
    return (
        AnchorDecision.VERIFIED
        if same_nonempty_digest and all(facts)
        else AnchorDecision.REJECTED
    )


def exhaustive_anchor_evidence() -> Iterable[AnchorEvidence]:
    """Yield the complete 3^3 x 2^9 finite abstraction state space."""

    digests = (None, "A", "B")
    for identity_values in product(digests, repeat=3):
        for facts in product((False, True), repeat=9):
            yield AnchorEvidence(
                prepared_digest=identity_values[0],
                report_digest=identity_values[1],
                reopened_digest=identity_values[2],
                commit_matches=facts[0],
                clean_code_state=facts[1],
                works_match=facts[2],
                run_config_matches=facts[3],
                policy_matches=facts[4],
                source_groups_valid=facts[5],
                global_basis_matches=facts[6],
                truth_manifest_matches=facts[7],
                audits_match=facts[8],
            )
