"""Immutable paired serialization artifact for grouped D0/R0 reports v4.

The in-memory v3 report contains ``ArmRouteSubmissionV1`` objects whose labels
are NumPy arrays.  A caller retaining an array reference can mutate it after the
report is built, changing the submission SHA and destabilizing the object graph.
V4 is the public artifact boundary: it snapshots the fully validated v3 report
into canonical UTF-8 bytes and groups D0/R0 evidence by held-out unit.

The frozen bytes remain stable even if a caller later mutates an input array.
Validation against the original source object then fails, as it should, while
the already-issued artifact retains its original identity.

This artifact is descriptive and contains no promotion decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json
import re

from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
)
from .counterfactual_risk_grouped_comparison_v3 import GroupedComparisonReportV3

REPORT_SCHEMA = "audio-extract/d0-r0-grouped-comparison-report/v4"
PAIRED_UNIT_SCHEMA = "audio-extract/d0-r0-paired-work-evaluation/v4"
STATUS = "COMPLETE_NO_PROMOTION_DECISION"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class GroupedReportArtifactV4Error(ValueError):
    """The immutable grouped-report artifact is malformed or source-drifted."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedReportArtifactV4Error(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise GroupedReportArtifactV4Error(
            f"{name} must be 40 lowercase hexadecimal characters"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GroupedReportArtifactV4Error(
            f"report value is not canonical JSON: {exc}"
        ) from exc


def _payload(
    report: GroupedComparisonReportV3,
    prereg: GroupedComparisonPreregistrationV1,
    verifier_commit: str,
) -> dict[str, Any]:
    report.validate(prereg)
    _git(verifier_commit, "v4 artifact verifier_commit")
    by_key = {
        (row.unit.sha256, row.arm_id): row
        for row in report.work_evaluations
    }
    pairs = []
    for unit in prereg.expected_units:
        d0 = by_key[(unit.sha256, "D0")]
        r0 = by_key[(unit.sha256, "R0")]
        if (d0.unit, r0.unit) != (unit, unit):
            raise GroupedReportArtifactV4Error(
                "paired report rows do not share the preregistered unit"
            )
        if (d0.arm_id, r0.arm_id) != ("D0", "R0"):
            raise GroupedReportArtifactV4Error(
                "paired report rows are not ordered D0/R0"
            )
        d0_unavailable = d0.evaluation.status == "ORACLE_UNAVAILABLE"
        r0_unavailable = r0.evaluation.status == "ORACLE_UNAVAILABLE"
        if d0_unavailable != r0_unavailable:
            raise GroupedReportArtifactV4Error(
                "paired report has one-sided oracle availability"
            )
        pairs.append(
            {
                "schema": PAIRED_UNIT_SCHEMA,
                "unit": unit.identity_dict(),
                "unit_sha256": unit.sha256,
                "d0": d0.identity_dict(prereg),
                "r0": r0.identity_dict(prereg),
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": STATUS,
        "promotion_decision": None,
        "preregistration_sha256": prereg.sha256,
        "source_v3_report_sha256": report.sha256(prereg),
        "verifier_commit": verifier_commit,
        "paired_units": pairs,
        "d0_summary": report.d0_summary.identity_dict(),
        "r0_summary": report.r0_summary.identity_dict(),
        "pairwise_summary": report.pairwise_summary.identity_dict(),
    }


@dataclass(frozen=True)
class GroupedComparisonReportArtifactV4:
    preregistration_sha256: str
    source_v3_report_sha256: str
    verifier_commit: str
    payload_utf8: bytes
    payload_sha256: str
    status: str = STATUS

    @classmethod
    def build(
        cls,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
        *,
        verifier_commit: str,
    ) -> "GroupedComparisonReportArtifactV4":
        value = _payload(report, prereg, verifier_commit)
        payload = _canonical(value)
        result = cls(
            preregistration_sha256=prereg.sha256,
            source_v3_report_sha256=report.sha256(prereg),
            verifier_commit=verifier_commit,
            payload_utf8=payload,
            payload_sha256=(
                "sha256:" + hashlib.sha256(payload).hexdigest()
            ),
        )
        result.validate_frozen()
        return result

    def validate_frozen(self) -> None:
        if self.status != STATUS:
            raise GroupedReportArtifactV4Error(
                "v4 report artifact must remain non-promoting"
            )
        _sha(self.preregistration_sha256, "preregistration_sha256")
        _sha(self.source_v3_report_sha256, "source_v3_report_sha256")
        _sha(self.payload_sha256, "payload_sha256")
        _git(self.verifier_commit, "verifier_commit")
        if type(self.payload_utf8) is not bytes or not self.payload_utf8:
            raise GroupedReportArtifactV4Error(
                "v4 report payload must be non-empty immutable bytes"
            )
        actual = "sha256:" + hashlib.sha256(self.payload_utf8).hexdigest()
        if actual != self.payload_sha256:
            raise GroupedReportArtifactV4Error(
                "v4 report payload bytes differ from their identity"
            )
        try:
            value = json.loads(self.payload_utf8.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GroupedReportArtifactV4Error(
                f"v4 report payload is not canonical UTF-8 JSON: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise GroupedReportArtifactV4Error(
                "v4 report payload is not a JSON object"
            )
        if _canonical(value) != self.payload_utf8:
            raise GroupedReportArtifactV4Error(
                "v4 report payload is not in canonical form"
            )
        expected_header = {
            "schema": REPORT_SCHEMA,
            "status": STATUS,
            "promotion_decision": None,
            "preregistration_sha256": self.preregistration_sha256,
            "source_v3_report_sha256": self.source_v3_report_sha256,
            "verifier_commit": self.verifier_commit,
        }
        drift = [
            name
            for name, expected in expected_header.items()
            if value.get(name) != expected
        ]
        if drift:
            raise GroupedReportArtifactV4Error(
                f"v4 report payload header differs: {drift}"
            )
        if not isinstance(value.get("paired_units"), list) or not value[
            "paired_units"
        ]:
            raise GroupedReportArtifactV4Error(
                "v4 report payload lacks paired held-out units"
            )

    def validate_against_source(
        self,
        report: GroupedComparisonReportV3,
        prereg: GroupedComparisonPreregistrationV1,
    ) -> None:
        self.validate_frozen()
        expected = _canonical(_payload(report, prereg, self.verifier_commit))
        if expected != self.payload_utf8:
            raise GroupedReportArtifactV4Error(
                "v4 report artifact differs from the current source object graph"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate_frozen()
        return json.loads(self.payload_utf8.decode("utf-8"))

    @property
    def sha256(self) -> str:
        self.validate_frozen()
        return self.payload_sha256
