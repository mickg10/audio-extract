"""Independent reconstruction audit for truth-free inference manifests."""

from __future__ import annotations

from .counterfactual_risk_dataset_contract_v2 import DatasetManifestV2
from .counterfactual_risk_feature_evidence_v1 import FeatureEvidenceRegistry
from .counterfactual_risk_inference_contract_v1 import (
    InferenceContractError,
    InferenceManifestV1,
)


def validate_inference_manifest_against_exact_evidence(
    manifest: InferenceManifestV1,
    dataset: DatasetManifestV2,
    feature_registry: FeatureEvidenceRegistry,
) -> None:
    """Rebuild the projection and require exact audit/model-input equality."""

    manifest.validate()
    expected = InferenceManifestV1.build(dataset, feature_registry)
    if manifest.model_input_document() != expected.model_input_document():
        raise InferenceContractError(
            "published model input differs from independent exact-evidence projection"
        )
    if manifest.identity_dict() != expected.identity_dict():
        raise InferenceContractError(
            "published inference audit binding differs from independent reconstruction"
        )
