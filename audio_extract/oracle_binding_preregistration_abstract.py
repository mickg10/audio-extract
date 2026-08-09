"""Finite model of preregistered oracle-routing promotion safety.

The concrete runner/verifier contains hashes, files, metrics, and solvers. This
module abstracts only the safety-relevant booleans and exhaustively proves that
no result becomes actionable when the policy, resolutions, source manifests,
legacy witness, v3 prewritten input, externally anchored claim, preflight, or
report binding is invalid.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import product


class Decision(str, Enum):
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    ACTIONABLE = "ACTIONABLE"
    RESOLUTION_SENSITIVE = "RESOLUTION_SENSITIVE"
    NO_ACTIONABLE_GAP = "NO_ACTIONABLE_GAP"


@dataclass(frozen=True)
class State:
    policy_pinned: bool
    exact_resolutions: bool
    voiced_nonempty: bool
    no_vocal_nonempty: bool
    manifests_physically_distinct: bool
    witness_preexists: bool
    report_binds_witness: bool
    run_input_preexists: bool
    claim_preexists: bool
    external_anchor_valid: bool
    digest_bindings_valid: bool
    output_absent_through_preflight: bool
    preflight_valid: bool
    report_binds_run_input_claim: bool
    hall_reference_complete: bool
    hall_metric_evidence_valid: bool
    all_method_resolution_evidence_valid: bool
    selected_primary_passes: bool
    selected_sensitivity_passes: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in self.__dict__.values()):
            raise TypeError("formal abstraction fields must be bool")

    @property
    def preregistration_valid(self) -> bool:
        return (
            self.policy_pinned
            and self.exact_resolutions
            and self.voiced_nonempty
            and self.no_vocal_nonempty
            and self.manifests_physically_distinct
            and self.witness_preexists
            and self.report_binds_witness
            and self.run_contract_valid
        )

    @property
    def run_contract_valid(self) -> bool:
        """Mirror the mandatory v3 publication/claim/preflight boundary."""

        return (
            self.run_input_preexists
            and self.claim_preexists
            and self.external_anchor_valid
            and self.digest_bindings_valid
            and self.output_absent_through_preflight
            and self.preflight_valid
            and self.report_binds_run_input_claim
            and self.hall_reference_complete
            and self.hall_metric_evidence_valid
        )

    @property
    def complete(self) -> bool:
        return self.preregistration_valid and self.all_method_resolution_evidence_valid


def decide(state: State) -> Decision:
    if not state.complete:
        return Decision.INVALID_EVIDENCE
    if state.selected_primary_passes:
        return Decision.ACTIONABLE
    if state.selected_sensitivity_passes:
        return Decision.RESOLUTION_SENSITIVE
    return Decision.NO_ACTIONABLE_GAP


def exhaustive_states() -> Iterable[State]:
    for values in product((False, True), repeat=19):
        yield State(*values)
