"""Strict JSON I/O and independent semantic verification for dataset v2.

The writer emits canonical JSON.  The reader rejects duplicate object keys,
unknown/missing fields, non-finite arrays, noncanonical row order, stale hashes,
and any disagreement between the published top-level contract and the
reconstructed Python semantics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .counterfactual_risk_dataset_contract_v1 import (
    CellGeometry,
    GroupFamilyIdentity,
)
from .counterfactual_risk_dataset_contract_v2 import (
    ARTIFACT_BINDING_SCHEMA,
    DATASET_SCHEMA,
    PANEL_SCHEMA,
    ROW_SCHEMA,
    CandidateArtifactBinding,
    CandidatePanel,
    CandidateSlot,
    CounterfactualRiskDatasetV2Error,
    CounterfactualRiskRowV2,
    DatasetManifestV2,
)


class CounterfactualRiskDatasetIOV2Error(CounterfactualRiskDatasetV2Error):
    """A serialized dataset-v2 document is ambiguous or semantically invalid."""


def _duplicate_rejecting_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CounterfactualRiskDatasetIOV2Error(
                f"duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def loads_dataset_json_v2(text: str | bytes | bytearray) -> dict[str, Any]:
    """Parse one dataset document while refusing duplicate keys and constants."""

    if isinstance(text, (bytes, bytearray)):
        try:
            value = bytes(text).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CounterfactualRiskDatasetIOV2Error(
                "dataset document is not valid UTF-8"
            ) from exc
    elif isinstance(text, str):
        value = text
    else:
        raise CounterfactualRiskDatasetIOV2Error(
            "dataset JSON input must be str or UTF-8 bytes"
        )
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CounterfactualRiskDatasetIOV2Error(
                    f"non-finite JSON number {token!r} is forbidden"
                )
            ),
        )
    except CounterfactualRiskDatasetIOV2Error:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise CounterfactualRiskDatasetIOV2Error(
            f"dataset document is not valid strict JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise CounterfactualRiskDatasetIOV2Error(
            "dataset document root must be an object"
        )
    return parsed


def _expect_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CounterfactualRiskDatasetIOV2Error(f"{name} must be an object")
    return value


def _expect_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise CounterfactualRiskDatasetIOV2Error(f"{name} must be an array")
    return value


def _reject_booleans(value: Any, name: str) -> Any:
    """Reject JSON booleans anywhere in a numeric scalar or nested array.

    ``True``/``False`` coerce silently to ``1.0``/``0.0`` under NumPy, so the
    JSON scalar type must be checked before any float coercion.
    """

    if isinstance(value, bool):
        raise CounterfactualRiskDatasetIOV2Error(f"{name} contains a boolean")
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_booleans(item, name)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    keys = set(value)
    missing = expected - keys
    extra = keys - expected
    if missing or extra:
        raise CounterfactualRiskDatasetIOV2Error(
            f"{name} keys differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )


_TOP_KEYS = {
    "schema",
    "purpose",
    "dataset_name",
    "source_commit",
    "ordered_row_ids",
    "group_family_sha256s",
    "candidate_panel",
    "candidate_panel_sha256",
    "metric_names",
    "metric_units",
    "metric_directions",
    "feature_contract_sha256",
    "metric_contract_sha256",
    "route_policy_sha256",
    "feature_count",
    "dataset_sha256",
    "rows",
}
_PANEL_KEYS = {"schema", "ordered_slots"}
_SLOT_KEYS = {
    "slot_id",
    "model_bundle_sha256",
    "adapter_bundle_sha256",
    "construction_contract_sha256",
    "query_contract_sha256",
    "output_role",
}
_GROUP_KEYS = {
    "work_id",
    "recording_session_id",
    "target_singer_id",
    "source_family_sha256",
    "query_condition_sha256",
}
_CELL_KEYS = {
    "sample_rate_hz",
    "resolution_ms",
    "start_frame",
    "end_frame",
    "band_low_hz",
    "band_high_hz",
    "spectral_grid_sha256",
}
_ARTIFACT_KEYS = {
    "schema",
    "status",
    "slot_id",
    "slot_sha256",
    "recipe_id",
    "artifact_pcm_sha256",
    "recipe_semantic_sha256",
    "recipe_slot_projection_sha256",
    "source_family_sha256",
    "verifier_commit",
}
_ROW_KEYS = {
    "schema",
    "row_id",
    "group_family_sha256",
    "group_family",
    "mixture_pcm_sha256",
    "accompaniment_truth_pcm_sha256",
    "vocal_truth_pcm_sha256",
    "cell_sha256",
    "cell",
    "candidate_panel_sha256",
    "candidate_artifacts",
    "metric_names",
    "metric_units",
    "metric_directions",
    "feature_contract_sha256",
    "metric_contract_sha256",
    "route_policy_sha256",
    "feature_sha256",
    "exact_risks_sha256",
    "availability_sha256",
    "features",
    "exact_risks",
    "available",
}


def _parse_panel(value: Any) -> CandidatePanel:
    mapping = _expect_mapping(value, "candidate_panel")
    _exact_keys(mapping, _PANEL_KEYS, "candidate_panel")
    if mapping.get("schema") != PANEL_SCHEMA:
        raise CounterfactualRiskDatasetIOV2Error(
            "candidate_panel schema is not dataset-v2 panel schema"
        )
    slots = []
    for index, raw in enumerate(
        _expect_list(mapping.get("ordered_slots"), "candidate_panel.ordered_slots")
    ):
        item = _expect_mapping(raw, f"candidate_panel.ordered_slots[{index}]")
        _exact_keys(item, _SLOT_KEYS, f"candidate_panel.ordered_slots[{index}]")
        slots.append(
            CandidateSlot(
                slot_id=item["slot_id"],
                model_bundle_sha256=item["model_bundle_sha256"],
                adapter_bundle_sha256=item["adapter_bundle_sha256"],
                construction_contract_sha256=(
                    item["construction_contract_sha256"]
                ),
                query_contract_sha256=item["query_contract_sha256"],
                output_role=item["output_role"],
            )
        )
    panel = CandidatePanel(tuple(slots))
    panel.validate()
    return panel


def _parse_group(value: Any, *, row_index: int) -> GroupFamilyIdentity:
    mapping = _expect_mapping(value, f"rows[{row_index}].group_family")
    _exact_keys(mapping, _GROUP_KEYS, f"rows[{row_index}].group_family")
    result = GroupFamilyIdentity(
        work_id=mapping["work_id"],
        recording_session_id=mapping["recording_session_id"],
        target_singer_id=mapping["target_singer_id"],
        source_family_sha256=mapping["source_family_sha256"],
        query_condition_sha256=mapping["query_condition_sha256"],
    )
    result.validate()
    return result


def _parse_cell(value: Any, *, row_index: int) -> CellGeometry:
    mapping = _expect_mapping(value, f"rows[{row_index}].cell")
    _exact_keys(mapping, _CELL_KEYS, f"rows[{row_index}].cell")
    result = CellGeometry(
        sample_rate_hz=mapping["sample_rate_hz"],
        resolution_ms=mapping["resolution_ms"],
        start_frame=mapping["start_frame"],
        end_frame=mapping["end_frame"],
        band_low_hz=mapping["band_low_hz"],
        band_high_hz=mapping["band_high_hz"],
        spectral_grid_sha256=mapping["spectral_grid_sha256"],
    )
    result.validate()
    return result


def _parse_artifacts(value: Any, *, row_index: int) -> tuple[CandidateArtifactBinding, ...]:
    result = []
    for index, raw in enumerate(
        _expect_list(value, f"rows[{row_index}].candidate_artifacts")
    ):
        item = _expect_mapping(
            raw, f"rows[{row_index}].candidate_artifacts[{index}]"
        )
        _exact_keys(
            item,
            _ARTIFACT_KEYS,
            f"rows[{row_index}].candidate_artifacts[{index}]",
        )
        if item.get("schema") != ARTIFACT_BINDING_SCHEMA:
            raise CounterfactualRiskDatasetIOV2Error(
                "candidate artifact binding uses the wrong schema"
            )
        result.append(
            CandidateArtifactBinding(
                slot_id=item["slot_id"],
                slot_sha256=item["slot_sha256"],
                recipe_id=item["recipe_id"],
                artifact_pcm_sha256=item["artifact_pcm_sha256"],
                recipe_semantic_sha256=item["recipe_semantic_sha256"],
                recipe_slot_projection_sha256=(
                    item["recipe_slot_projection_sha256"]
                ),
                source_family_sha256=item["source_family_sha256"],
                verifier_commit=item["verifier_commit"],
                status=item["status"],
            )
        )
    return tuple(result)


def _parse_row(value: Any, *, panel: CandidatePanel, row_index: int) -> CounterfactualRiskRowV2:
    mapping = _expect_mapping(value, f"rows[{row_index}]")
    _exact_keys(mapping, _ROW_KEYS, f"rows[{row_index}]")
    if mapping.get("schema") != ROW_SCHEMA:
        raise CounterfactualRiskDatasetIOV2Error(
            f"rows[{row_index}] uses the wrong row schema"
        )
    row = CounterfactualRiskRowV2(
        group_family=_parse_group(mapping["group_family"], row_index=row_index),
        mixture_pcm_sha256=mapping["mixture_pcm_sha256"],
        accompaniment_truth_pcm_sha256=(
            mapping["accompaniment_truth_pcm_sha256"]
        ),
        vocal_truth_pcm_sha256=mapping["vocal_truth_pcm_sha256"],
        cell=_parse_cell(mapping["cell"], row_index=row_index),
        candidate_panel=panel,
        candidate_artifacts=_parse_artifacts(
            mapping["candidate_artifacts"], row_index=row_index
        ),
        metric_names=tuple(
            _expect_list(mapping["metric_names"], f"rows[{row_index}].metric_names")
        ),
        metric_units=tuple(
            _expect_list(mapping["metric_units"], f"rows[{row_index}].metric_units")
        ),
        metric_directions=tuple(
            _expect_list(
                mapping["metric_directions"],
                f"rows[{row_index}].metric_directions",
            )
        ),
        features=np.asarray(
            _reject_booleans(
                _expect_list(
                    mapping["features"], f"rows[{row_index}].features"
                ),
                "features",
            ),
            dtype=np.float64,
        ),
        exact_risks=np.asarray(
            _reject_booleans(
                _expect_list(
                    mapping["exact_risks"], f"rows[{row_index}].exact_risks"
                ),
                "exact_risks",
            ),
            dtype=np.float64,
        ),
        available=np.asarray(
            _expect_list(mapping["available"], f"rows[{row_index}].available")
        ),
        feature_contract_sha256=mapping["feature_contract_sha256"],
        metric_contract_sha256=mapping["metric_contract_sha256"],
        route_policy_sha256=mapping["route_policy_sha256"],
    )
    row.validate()
    expected = row.to_record()
    if dict(mapping) != expected:
        differing = sorted(
            key for key in _ROW_KEYS if mapping.get(key) != expected.get(key)
        )
        raise CounterfactualRiskDatasetIOV2Error(
            f"rows[{row_index}] does not match recomputed semantics; "
            f"differing fields={differing}"
        )
    return row


def parse_dataset_document_v2(document: Mapping[str, Any]) -> DatasetManifestV2:
    """Reconstruct and independently verify a dataset-v2 document."""

    mapping = _expect_mapping(document, "dataset document")
    _exact_keys(mapping, _TOP_KEYS, "dataset document")
    if mapping.get("schema") != DATASET_SCHEMA:
        raise CounterfactualRiskDatasetIOV2Error(
            "dataset document uses the wrong schema"
        )
    if isinstance(mapping.get("feature_count"), bool) or not isinstance(
        mapping.get("feature_count"), int
    ) or mapping["feature_count"] < 1:
        raise CounterfactualRiskDatasetIOV2Error(
            "feature_count must be a positive integer, not boolean"
        )
    panel = _parse_panel(mapping["candidate_panel"])
    rows = tuple(
        _parse_row(raw, panel=panel, row_index=index)
        for index, raw in enumerate(
            _expect_list(mapping["rows"], "dataset document rows")
        )
    )
    dataset = DatasetManifestV2(
        rows=rows,
        source_commit=mapping["source_commit"],
        dataset_name=mapping["dataset_name"],
        purpose=mapping["purpose"],
    )
    dataset.validate()
    expected = dataset.to_document()
    if dict(mapping) != expected:
        differing = sorted(
            key for key in _TOP_KEYS if mapping.get(key) != expected.get(key)
        )
        raise CounterfactualRiskDatasetIOV2Error(
            "dataset top-level semantics differ from recomputation; "
            f"differing fields={differing}"
        )
    return dataset


def dump_dataset_json_v2(dataset: DatasetManifestV2) -> bytes:
    """Return canonical UTF-8 JSON with a trailing newline."""

    dataset.validate()
    return (
        json.dumps(
            dataset.to_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def dataset_container_sha256_v2(payload: bytes | bytearray) -> str:
    """Hash exact published bytes separately from the semantic dataset hash."""

    return "sha256:" + hashlib.sha256(bytes(payload)).hexdigest()


def load_dataset_path_v2(path: str | Path) -> tuple[DatasetManifestV2, str]:
    """Read, parse, and return semantic plus exact-container identities."""

    payload = Path(path).read_bytes()
    dataset = parse_dataset_document_v2(loads_dataset_json_v2(payload))
    return dataset, dataset_container_sha256_v2(payload)
