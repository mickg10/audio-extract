"""Grouped D0 training-manifest boundary over independent per-panel teachers.

A single per-panel structured teacher
(:mod:`counterfactual_risk_d0_structured_teacher_v1`) certifies exactly one frozen
group's cell targets and behavioral margins; the strongest status it may emit is
``READY_FOR_GROUPED_ASSEMBLY``.  It must never claim finite-sample conformal or
training coverage, because one panel is one exchangeable group.

This module is the *only* place a D0 training-ready status may be emitted.  It
aggregates the distinct group identities (source-family / work / session / room,
carried by each panel's ``group_family_sha256``) of grouped-assembly-ready
teachers and admits training only when the finite-group conformal inequality

    ceil((n_groups + 1) * (1 - alpha)) <= n_groups

is satisfiable.  ``alpha`` and ``n_groups`` are identity-bearing (hashed into the
manifest's canonical identity).  When the inequality is unsatisfiable the manifest
fails closed with an uncertifiable status and never claims training readiness.

This is an offline assembly contract.  It composes immutable per-panel teacher
artifacts and must never be used by the inference API.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .counterfactual_risk_d0_structured_teacher_v1 import (
    READY_FOR_GROUPED_ASSEMBLY,
    D0StructuredTeacherV1,
)

MANIFEST_SCHEMA = "audio-extract/counterfactual-risk-grouped-training-manifest/v1"
MEMBER_SCHEMA = "audio-extract/counterfactual-risk-grouped-assembly-member/v1"
# Only this boundary may emit a training-ready status, and only after validating
# the independent-group evidence via the finite-group conformal inequality.
READY_FOR_D0_TRAINING = "READY_FOR_D0_TRAINING"
UNCERTIFIABLE_INSUFFICIENT_INDEPENDENT_GROUPS = (
    "UNCERTIFIABLE_INSUFFICIENT_INDEPENDENT_GROUPS"
)
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class GroupedTrainingManifestError(ValueError):
    """A grouped assembly member or grouped training manifest is invalid."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise GroupedTrainingManifestError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class GroupedAssemblyMemberV1:
    """One immutable grouped-assembly-ready per-panel teacher artifact.

    Every field is derived from a real :class:`D0StructuredTeacherV1`, so the
    group identity and the content-addressed teacher artifact identity always
    describe the same panel; a caller cannot assert a group for an unrelated
    teacher.
    """

    group_family_sha256: str
    panel_sha256: str
    teacher_sha256: str
    status: str

    @classmethod
    def from_teacher(
        cls,
        teacher: D0StructuredTeacherV1,
        *,
        candidate_count: int,
    ) -> GroupedAssemblyMemberV1:
        teacher.validate(candidate_count=candidate_count)
        if teacher.status != READY_FOR_GROUPED_ASSEMBLY:
            raise GroupedTrainingManifestError(
                "only a grouped-assembly-ready teacher may join the manifest; "
                f"got status {teacher.status!r}"
            )
        return cls(
            group_family_sha256=teacher.group_family_sha256,
            panel_sha256=teacher.panel_sha256,
            teacher_sha256=teacher.sha256(candidate_count=candidate_count),
            status=teacher.status,
        )

    def validate(self) -> None:
        _sha(self.group_family_sha256, "member group_family_sha256")
        _sha(self.panel_sha256, "member panel_sha256")
        _sha(self.teacher_sha256, "member teacher_sha256")
        if self.status != READY_FOR_GROUPED_ASSEMBLY:
            raise GroupedTrainingManifestError(
                "grouped assembly member must be grouped-assembly-ready"
            )

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": MEMBER_SCHEMA,
            "group_family_sha256": self.group_family_sha256,
            "panel_sha256": self.panel_sha256,
            "teacher_sha256": self.teacher_sha256,
            "status": self.status,
        }


@dataclass(frozen=True)
class GroupedTrainingManifestV1:
    """Aggregate independent grouped-assembly panels into a D0 training gate."""

    members: tuple[GroupedAssemblyMemberV1, ...]
    alpha: float

    @classmethod
    def build(
        cls,
        members: Sequence[GroupedAssemblyMemberV1],
        *,
        alpha: float,
    ) -> GroupedTrainingManifestV1:
        ordered = tuple(
            sorted(members, key=lambda member: member.teacher_sha256)
        )
        result = cls(members=ordered, alpha=float(alpha))
        result.validate()
        return result

    @classmethod
    def from_teachers(
        cls,
        teachers: Sequence[tuple[D0StructuredTeacherV1, int]],
        *,
        alpha: float,
    ) -> GroupedTrainingManifestV1:
        """Build from ``(teacher, candidate_count)`` pairs."""

        members = tuple(
            GroupedAssemblyMemberV1.from_teacher(
                teacher, candidate_count=candidate_count
            )
            for teacher, candidate_count in teachers
        )
        return cls.build(members, alpha=alpha)

    def validate(self) -> None:
        alpha = float(self.alpha)
        # Mirror the finite-group conformal convention used elsewhere in the
        # study: the coverage target 1 - alpha must lie strictly in (0.5, 1).
        if not math.isfinite(alpha) or not 0.0 < alpha < 0.5:
            raise GroupedTrainingManifestError(
                "alpha must be finite and strictly within (0, 0.5)"
            )
        if not self.members:
            raise GroupedTrainingManifestError(
                "grouped training manifest has no members"
            )
        for member in self.members:
            member.validate()
        teacher_shas = [member.teacher_sha256 for member in self.members]
        if len(set(teacher_shas)) != len(teacher_shas):
            raise GroupedTrainingManifestError(
                "grouped training manifest repeats a per-panel teacher artifact"
            )
        if teacher_shas != sorted(teacher_shas):
            raise GroupedTrainingManifestError(
                "grouped training manifest members must be in canonical "
                "teacher-hash order"
            )

    @property
    def group_family_sha256s(self) -> tuple[str, ...]:
        return tuple(
            sorted({member.group_family_sha256 for member in self.members})
        )

    @property
    def n_groups(self) -> int:
        return len(self.group_family_sha256s)

    @property
    def target_coverage(self) -> float:
        return 1.0 - float(self.alpha)

    @property
    def conformal_rank(self) -> int:
        return math.ceil((self.n_groups + 1) * self.target_coverage)

    @property
    def certifiable(self) -> bool:
        return self.conformal_rank <= self.n_groups

    @property
    def status(self) -> str:
        if self.certifiable:
            return READY_FOR_D0_TRAINING
        return UNCERTIFIABLE_INSUFFICIENT_INDEPENDENT_GROUPS

    def training_ready(self) -> bool:
        return self.status == READY_FOR_D0_TRAINING

    def identity_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": MANIFEST_SCHEMA,
            "status": self.status,
            "alpha": float(self.alpha),
            "target_coverage": self.target_coverage,
            "n_groups": self.n_groups,
            "conformal_rank": self.conformal_rank,
            "group_family_sha256s": list(self.group_family_sha256s),
            "members": [member.identity_dict() for member in self.members],
        }

    @property
    def sha256(self) -> str:
        return _mapping_sha(self.identity_dict())
