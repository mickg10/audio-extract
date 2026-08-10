import dataclasses
import hashlib
import json

import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import (
    CellPartitionCertificate,
    CellPartitionEntry,
    RationalMeasure,
)
from audio_extract.counterfactual_risk_grouped_comparison_v1 import HeldOutUnitV1
from audio_extract.counterfactual_risk_source_lineage_v3 import (
    ExactSourceLineageV3,
    ExactSourceLineageV3Error,
    ExactSourceReportDocumentV3,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def context(name="opera"):
    seed = HeldOutUnitV1(
        work_id=name,
        recording_session_id=f"session-{name}",
        target_singer_id=f"singer-{name}",
        source_family_sha256=sha(f"source-{name}"),
        query_condition_sha256=sha(f"query-{name}"),
        query_quality_stratum_sha256=sha(f"quality-{name}"),
        exact_evidence_sha256=sha("placeholder"),
        exact_panel_sha256=sha(f"panel-{name}"),
        exact_preflight_sha256=sha(f"preflight-{name}"),
        exact_oracle_decision_sha256=sha(f"oracle-{name}"),
    )
    parents = {
        "truth_manifest_sha256": sha(f"truth-{name}"),
        "mixture_pcm_sha256": sha(f"mixture-{name}"),
        "accompaniment_pcm_sha256": sha(f"accompaniment-{name}"),
        "vocal_pcm_sha256": sha(f"vocal-{name}"),
    }
    risk = ExactSourceReportDocumentV3.build(
        role="risk_evidence",
        group_family_sha256=seed.group_family_sha256,
        source_artifact_sha256=sha("risk-artifact"),
        source_artifact_schema="audio-extract/exact-risk-evidence/v1",
        verifier_commit=git("risk-verifier"),
        **parents,
    )
    unit = dataclasses.replace(seed, exact_evidence_sha256=risk.sha256)
    partition = ExactSourceReportDocumentV3.build(
        role="partition_source",
        group_family_sha256=unit.group_family_sha256,
        source_artifact_sha256=sha("partition-artifact"),
        source_artifact_schema="audio-extract/cell-partition-source/v1",
        verifier_commit=git("partition-verifier"),
        **parents,
    )
    certificate = CellPartitionCertificate(
        group_family_sha256=unit.group_family_sha256,
        spectral_grid_sha256=sha("grid"),
        resolution_ms=500,
        time_cell_count=1,
        band_count=1,
        entries=(
            CellPartitionEntry(0, 0, sha("cell"), RationalMeasure(1)),
        ),
        time_ranges_sha256=sha("time-ranges"),
        frequency_ranges_sha256=sha("frequency-ranges"),
        measure_contract_sha256=sha("measure-contract"),
        exact_source_report_sha256=partition.sha256,
        verifier_commit=git("partition-verifier"),
    )
    lineage = ExactSourceLineageV3(
        risk_evidence_report=risk,
        partition_source_report=partition,
        lineage_verifier_commit=git("lineage-verifier"),
    )
    return unit, certificate, lineage, parents


def test_distinct_report_hashes_validate_when_derived_parents_match():
    unit, certificate, lineage, parents = context()
    assert lineage.risk_evidence_report.sha256 != (
        lineage.partition_source_report.sha256
    )
    lineage.validate(unit, certificate)
    assert lineage.parent_tuple == (
        parents["truth_manifest_sha256"],
        parents["mixture_pcm_sha256"],
        parents["accompaniment_pcm_sha256"],
        parents["vocal_pcm_sha256"],
    )


def test_changed_parent_changes_report_identity_and_is_rejected():
    unit, certificate, lineage, parents = context()
    changed = ExactSourceReportDocumentV3.build(
        role="partition_source",
        group_family_sha256=unit.group_family_sha256,
        source_artifact_sha256=sha("partition-artifact"),
        source_artifact_schema="audio-extract/cell-partition-source/v1",
        truth_manifest_sha256=sha("another-truth"),
        mixture_pcm_sha256=parents["mixture_pcm_sha256"],
        accompaniment_pcm_sha256=parents["accompaniment_pcm_sha256"],
        vocal_pcm_sha256=parents["vocal_pcm_sha256"],
        verifier_commit=certificate.verifier_commit,
    )
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="partition source report bytes differ",
    ):
        dataclasses.replace(
            lineage,
            partition_source_report=changed,
        ).validate(unit, certificate)


def test_report_role_cannot_be_substituted():
    unit, certificate, lineage, parents = context()
    wrong_role = ExactSourceReportDocumentV3.build(
        role="risk_evidence",
        group_family_sha256=unit.group_family_sha256,
        source_artifact_sha256=sha("partition-artifact"),
        source_artifact_schema="audio-extract/cell-partition-source/v1",
        verifier_commit=certificate.verifier_commit,
        **parents,
    )
    wrong_certificate = dataclasses.replace(
        certificate,
        exact_source_report_sha256=wrong_role.sha256,
    )
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="partition source document has the wrong role",
    ):
        dataclasses.replace(
            lineage,
            partition_source_report=wrong_role,
        ).validate(unit, wrong_certificate)


def test_noncanonical_report_bytes_are_refused():
    unit, _certificate, lineage, _parents = context()
    value = lineage.risk_evidence_report.report()
    pretty = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with pytest.raises(
        ExactSourceLineageV3Error,
        match="not canonical",
    ):
        ExactSourceReportDocumentV3(pretty).validate()
    assert unit.exact_evidence_sha256 != (
        "sha256:" + hashlib.sha256(pretty).hexdigest()
    )
