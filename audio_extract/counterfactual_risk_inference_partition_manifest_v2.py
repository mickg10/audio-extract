"""Content-bearing exact-source and inference-partition manifests.

The v3 grouped comparison intentionally stored only the SHA of each inference
input manifest.  A hash alone does not prove which spectral grid, resolution,
range set, or cell partition the arm actually consumed.  This module freezes the
pre-inference input record itself and links the two distinct exact-reference
provenance nodes used elsewhere:

* complete-route risk evidence for the held-out unit;
* the source/geometry report used to certify a cell partition.

Those nodes are not required to have identical hashes.  Their relationship is
established through one content-bearing source-lineage record with common truth
and decoded-PCM parents.

This is dormant CPU-only research infrastructure.  It does not fit a model,
render audio, or make a promotion decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import hashlib
import json
import re

from .counterfactual_risk_cell_partition_v1 import CellPartitionCertificate
from .counterfactual_risk_grouped_comparison_v1 import (
    GroupedComparisonPreregistrationV1,
    HeldOutUnitV1,
)
from .counterfactual_risk_grouped_comparison_v3 import FrozenArmInferenceRunV3

SOURCE_LINEAGE_SCHEMA = "audio-extract/d0-r0-exact-source-lineage/v2"
INPUT_MANIFEST_SCHEMA = "audio-extract/d0-r0-inference-partition-inputs/v2"
_SHA_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class InferencePartitionManifestError(ValueError):
    """Exact-source and inference-grid evidence are not one coherent record."""


def _sha(value: Any, name: str) -> str:
    result = str(value or "")
    if _SHA_RE.fullmatch(result) is None:
        raise InferencePartitionManifestError(
            f"{name} must be canonical sha256:<64 lowercase hex>"
        )
    return result


def _git(value: Any, name: str) -> str:
    result = str(value or "")
    if _GIT_RE.fullmatch(result) is None:
        raise InferencePartitionManifestError(
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
        raise InferencePartitionManifestError(
            f"inference input value is not canonical JSON: {exc}"
        ) from exc


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _expected_arm(prereg: GroupedComparisonPreregistrationV1, arm_id: str):
    if arm_id == "D0":
        return prereg.d0_arm
    if arm_id == "R0":
        return prereg.r0_arm
    raise InferencePartitionManifestError(f"unknown arm {arm_id!r}")


@dataclass(frozen=True)
class ExactSourceLineageV2:
    """Relate risk evidence and partition-source evidence through common truth."""

    unit_sha256: str
    group_family_sha256: str
    exact_evidence_sha256: str
    exact_source_report_sha256: str
    truth_manifest_sha256: str
    mixture_pcm_sha256: str
    accompaniment_pcm_sha256: str
    vocal_pcm_sha256: str
    exact_evidence_builder_commit: str
    partition_source_verifier_commit: str
    lineage_verifier_commit: str
    status: str = "verified"

    def validate(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> None:
        unit.validate()
        certificate.validate()
        if self.status != "verified":
            raise InferencePartitionManifestError(
                "exact source lineage status must be verified"
            )
        for name in (
            "unit_sha256",
            "group_family_sha256",
            "exact_evidence_sha256",
            "exact_source_report_sha256",
            "truth_manifest_sha256",
            "mixture_pcm_sha256",
            "accompaniment_pcm_sha256",
            "vocal_pcm_sha256",
        ):
            _sha(getattr(self, name), name)
        for name in (
            "exact_evidence_builder_commit",
            "partition_source_verifier_commit",
            "lineage_verifier_commit",
        ):
            _git(getattr(self, name), name)
        expected = {
            "unit_sha256": unit.sha256,
            "group_family_sha256": unit.group_family_sha256,
            "exact_evidence_sha256": unit.exact_evidence_sha256,
            "exact_source_report_sha256": (
                certificate.exact_source_report_sha256
            ),
            "partition_source_verifier_commit": certificate.verifier_commit,
        }
        drift = [
            name
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise InferencePartitionManifestError(
                f"exact source lineage differs from unit/partition: {drift}"
            )
        if certificate.group_family_sha256 != unit.group_family_sha256:
            raise InferencePartitionManifestError(
                "partition source report belongs to another group family"
            )

    def identity_dict(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> dict[str, Any]:
        self.validate(unit, certificate)
        return {"schema": SOURCE_LINEAGE_SCHEMA, **self.__dict__}

    def sha256(
        self,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> str:
        return _mapping_sha(self.identity_dict(unit, certificate))


@dataclass(frozen=True)
class InferenceInputPartitionManifestV2:
    """The immutable input/grid record written before one arm is executed."""

    preregistration_sha256: str
    unit_sha256: str
    group_family_sha256: str
    query_condition_sha256: str
    query_quality_stratum_sha256: str
    arm_id: str
    arm_run_sha256: str
    dataset_sha256: str
    split_sha256: str
    test_subset_sha256: str
    candidate_panel_sha256: str
    feature_contract_sha256: str
    metric_contract_sha256: str
    partition_certificate_sha256: str
    spectral_grid_sha256: str
    resolution_ms: int
    time_ranges_sha256: str
    frequency_ranges_sha256: str
    measure_contract_sha256: str
    source_lineage: ExactSourceLineageV2
    builder_commit: str
    status: str = "frozen"

    @property
    def partition_key(self) -> tuple[str, str, int]:
        return (
            self.group_family_sha256,
            self.spectral_grid_sha256,
            self.resolution_ms,
        )

    def _validate_without_run(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> None:
        prereg.validate()
        unit.validate()
        certificate.validate()
        if self.status != "frozen":
            raise InferencePartitionManifestError(
                "inference input manifest must be frozen before execution"
            )
        for name in (
            "preregistration_sha256",
            "unit_sha256",
            "group_family_sha256",
            "query_condition_sha256",
            "query_quality_stratum_sha256",
            "arm_run_sha256",
            "dataset_sha256",
            "split_sha256",
            "test_subset_sha256",
            "candidate_panel_sha256",
            "feature_contract_sha256",
            "metric_contract_sha256",
            "partition_certificate_sha256",
            "spectral_grid_sha256",
            "time_ranges_sha256",
            "frequency_ranges_sha256",
            "measure_contract_sha256",
        ):
            _sha(getattr(self, name), name)
        _git(self.builder_commit, "inference input builder_commit")
        if type(self.resolution_ms) is not int or self.resolution_ms < 1:
            raise InferencePartitionManifestError(
                "inference input resolution_ms must be a positive integer"
            )
        arm = _expected_arm(prereg, self.arm_id)
        expected = {
            "preregistration_sha256": prereg.sha256,
            "unit_sha256": unit.sha256,
            "group_family_sha256": unit.group_family_sha256,
            "query_condition_sha256": unit.query_condition_sha256,
            "query_quality_stratum_sha256": (
                unit.query_quality_stratum_sha256
            ),
            "arm_run_sha256": arm.sha256,
            "dataset_sha256": prereg.dataset_sha256,
            "split_sha256": prereg.split_sha256,
            "test_subset_sha256": prereg.test_subset_sha256,
            "candidate_panel_sha256": prereg.candidate_panel_sha256,
            "feature_contract_sha256": prereg.feature_contract_sha256,
            "metric_contract_sha256": prereg.metric_contract_sha256,
            "partition_certificate_sha256": certificate.sha256,
            "spectral_grid_sha256": certificate.spectral_grid_sha256,
            "resolution_ms": certificate.resolution_ms,
            "time_ranges_sha256": certificate.time_ranges_sha256,
            "frequency_ranges_sha256": certificate.frequency_ranges_sha256,
            "measure_contract_sha256": certificate.measure_contract_sha256,
        }
        drift = [
            name
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if drift:
            raise InferencePartitionManifestError(
                f"inference input manifest differs from frozen context: {drift}"
            )
        if self.partition_key != certificate.partition_key:
            raise InferencePartitionManifestError(
                "inference input partition key differs from certificate"
            )
        self.source_lineage.validate(unit, certificate)

    def identity_dict(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> dict[str, Any]:
        self._validate_without_run(prereg, unit, certificate)
        return {
            "schema": INPUT_MANIFEST_SCHEMA,
            "status": self.status,
            "preregistration_sha256": self.preregistration_sha256,
            "unit_sha256": self.unit_sha256,
            "group_family_sha256": self.group_family_sha256,
            "query_condition_sha256": self.query_condition_sha256,
            "query_quality_stratum_sha256": (
                self.query_quality_stratum_sha256
            ),
            "arm_id": self.arm_id,
            "arm_run_sha256": self.arm_run_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "test_subset_sha256": self.test_subset_sha256,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "partition_certificate_sha256": (
                self.partition_certificate_sha256
            ),
            "spectral_grid_sha256": self.spectral_grid_sha256,
            "resolution_ms": self.resolution_ms,
            "time_ranges_sha256": self.time_ranges_sha256,
            "frequency_ranges_sha256": self.frequency_ranges_sha256,
            "measure_contract_sha256": self.measure_contract_sha256,
            "source_lineage": self.source_lineage.identity_dict(
                unit, certificate
            ),
            "source_lineage_sha256": self.source_lineage.sha256(
                unit, certificate
            ),
            "builder_commit": self.builder_commit,
        }

    def sha256(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        certificate: CellPartitionCertificate,
    ) -> str:
        return _mapping_sha(self.identity_dict(prereg, unit, certificate))

    def validate(
        self,
        prereg: GroupedComparisonPreregistrationV1,
        unit: HeldOutUnitV1,
        run: FrozenArmInferenceRunV3,
        certificate: CellPartitionCertificate,
    ) -> None:
        self._validate_without_run(prereg, unit, certificate)
        run.validate(prereg, unit)
        expected_run = {
            "arm_id": self.arm_id,
            "arm_run_sha256": self.arm_run_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "test_subset_sha256": self.test_subset_sha256,
            "candidate_panel_sha256": self.candidate_panel_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
        }
        drift = [
            name
            for name, value in expected_run.items()
            if getattr(run, name) != value
        ]
        if drift:
            raise InferencePartitionManifestError(
                f"inference run differs from frozen input manifest: {drift}"
            )
        expected_sha = self.sha256(prereg, unit, certificate)
        if run.inference_input_manifest_sha256 != expected_sha:
            raise InferencePartitionManifestError(
                "inference run does not name this input/partition manifest"
            )
