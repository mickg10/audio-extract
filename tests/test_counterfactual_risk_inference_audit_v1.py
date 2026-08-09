import hashlib

import numpy as np
import pytest

from audio_extract.counterfactual_risk_dataset_contract_v1 import (
    CellGeometry,
    GroupFamilyIdentity,
)
from audio_extract.counterfactual_risk_dataset_contract_v2 import (
    CandidateArtifactBinding,
    CandidatePanel,
    CandidateSlot,
    CounterfactualRiskRowV2,
    DatasetManifestV2,
)
from audio_extract.counterfactual_risk_feature_evidence_v1 import (
    FeatureEvidenceCertificate,
    FeatureEvidenceRegistry,
)
from audio_extract.counterfactual_risk_inference_audit_v1 import (
    validate_inference_manifest_against_exact_evidence,
)
from audio_extract.counterfactual_risk_inference_contract_v1 import (
    InferenceAuditBinding,
    InferenceContractError,
    InferenceManifestV1,
    InferenceRowV1,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


SOURCE_COMMIT = git("source")


def exact_dataset():
    panel = CandidatePanel(
        (
            CandidateSlot(
                "slot-a",
                sha("model-a"),
                sha("adapter-a"),
                sha("construction-a"),
                sha("query-contract"),
            ),
            CandidateSlot(
                "slot-b",
                sha("model-b"),
                sha("adapter-b"),
                sha("construction-b"),
                sha("query-contract"),
            ),
        )
    )
    source = sha("source-family")
    artifacts = tuple(
        CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=sha(f"recipe-{slot.slot_id}"),
            artifact_pcm_sha256=sha(f"pcm-{slot.slot_id}"),
            recipe_semantic_sha256=sha(f"semantic-{slot.slot_id}"),
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=source,
            verifier_commit=git("artifact-verifier"),
        )
        for slot in panel.slots
    )
    row = CounterfactualRiskRowV2(
        group_family=GroupFamilyIdentity(
            work_id="work",
            recording_session_id="session",
            target_singer_id="singer",
            source_family_sha256=source,
            query_condition_sha256=sha("query"),
        ),
        mixture_pcm_sha256=sha("mixture"),
        accompaniment_truth_pcm_sha256=sha("a-truth"),
        vocal_truth_pcm_sha256=sha("v-truth"),
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=0,
            end_frame=1000,
            band_low_hz=0,
            band_high_hz=500,
            spectral_grid_sha256=sha("grid"),
        ),
        candidate_panel=panel,
        candidate_artifacts=artifacts,
        metric_names=("voice", "hole"),
        metric_units=("ratio", "ratio"),
        metric_directions=("lower_is_better", "lower_is_better"),
        features=np.asarray([0.25, -0.5]),
        exact_risks=np.asarray([[0.1, 0.2], [0.2, 0.1]]),
        available=np.ones((2, 2), dtype=bool),
        feature_contract_sha256=sha("feature-contract"),
        metric_contract_sha256=sha("metric-contract"),
        route_policy_sha256=sha("route-policy"),
    )
    dataset = DatasetManifestV2.build(
        (row,), source_commit=SOURCE_COMMIT
    )
    certificate = FeatureEvidenceCertificate(
        row_id=row.row_id,
        source_family_sha256=source,
        feature_contract_sha256=row.feature_contract_sha256,
        feature_sha256=row.feature_sha256,
        available=(True, True),
        required=(True, True),
        missing_value_policy_sha256=sha("missing-policy"),
        extraction_report_sha256=sha("feature-report"),
        extractor_bundle_sha256=sha("extractor"),
        verifier_commit=git("feature-verifier"),
    )
    registry = FeatureEvidenceRegistry.build(
        (certificate,), source_commit=SOURCE_COMMIT
    )
    return dataset, registry


def test_independent_reconstruction_accepts_exact_manifest():
    dataset, registry = exact_dataset()
    manifest = InferenceManifestV1.build(dataset, registry)
    validate_inference_manifest_against_exact_evidence(
        manifest, dataset, registry
    )


def test_forged_offline_audit_binding_is_refused_even_when_manifest_self_validates():
    dataset, registry = exact_dataset()
    manifest = InferenceManifestV1.build(dataset, registry)
    original = manifest.audit_bindings[0]
    forged = InferenceAuditBinding(
        inference_row_sha256=original.inference_row_sha256,
        offline_row_id=sha("forged-offline-row"),
        feature_evidence_sha256=sha("forged-feature-evidence"),
    )
    damaged = InferenceManifestV1(
        **{**manifest.__dict__, "audit_bindings": (forged,)}
    )
    # The standalone manifest has only syntactic commitments for its audit
    # hashes; independent reconstruction is what proves their origin.
    damaged.validate()
    with pytest.raises(
        InferenceContractError,
        match="audit binding differs",
    ):
        validate_inference_manifest_against_exact_evidence(
            damaged, dataset, registry
        )


def test_forged_model_input_is_refused_by_independent_reconstruction():
    dataset, registry = exact_dataset()
    manifest = InferenceManifestV1.build(dataset, registry)
    row = manifest.rows[0]
    changed = InferenceRowV1(
        **{**row.__dict__, "features": (9.0, row.features[1])}
    )
    changed_sha = changed.sha256(candidate_panel=manifest.candidate_panel)
    audit = InferenceAuditBinding(
        inference_row_sha256=changed_sha,
        offline_row_id=manifest.audit_bindings[0].offline_row_id,
        feature_evidence_sha256=(
            manifest.audit_bindings[0].feature_evidence_sha256
        ),
    )
    damaged = InferenceManifestV1(
        **{
            **manifest.__dict__,
            "rows": (changed,),
            "audit_bindings": (audit,),
        }
    )
    damaged.validate()
    with pytest.raises(
        InferenceContractError,
        match="model input differs",
    ):
        validate_inference_manifest_against_exact_evidence(
            damaged, dataset, registry
        )
