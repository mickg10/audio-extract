import hashlib
from fractions import Fraction

import numpy as np
import pytest

from audio_extract.counterfactual_risk_cell_partition_v1 import (
    CellPartitionCertificate,
    CellPartitionEntry,
    CellPartitionRegistry,
    RationalMeasure,
)
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
from audio_extract.counterfactual_risk_inference_contract_v1 import (
    InferenceManifestV1,
)
from audio_extract.counterfactual_risk_partitioned_prediction_v1 import (
    PartitionedPredictionError,
    PartitionedUpperRiskPanelV1,
    assemble_partitioned_upper_risks,
)
from audio_extract.counterfactual_risk_student_inference_v1 import (
    PredictedUpperRiskManifestV1,
    predict_upper_fail_closed,
)


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def git(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


SOURCE_COMMIT = git("source")
FEATURE_CONTRACT = sha("feature-contract")
METRIC_CONTRACT = sha("metric-contract")
ROUTE_POLICY = sha("route-policy")
PREDICTION_POLICY = sha("prediction-policy")


def panel() -> CandidatePanel:
    return CandidatePanel(
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


def group_identity(group_index: int) -> GroupFamilyIdentity:
    return GroupFamilyIdentity(
        work_id=f"work-{group_index}",
        recording_session_id=f"session-{group_index}",
        target_singer_id=f"singer-{group_index}",
        source_family_sha256=sha(f"source-family-{group_index}"),
        query_condition_sha256=sha(f"query-{group_index}"),
    )


def exact_row(
    group_index: int,
    time_index: int,
    band_index: int,
    *,
    blocked: bool = False,
) -> CounterfactualRiskRowV2:
    candidate_panel = panel()
    group = group_identity(group_index)
    artifacts = tuple(
        CandidateArtifactBinding(
            slot_id=slot.slot_id,
            slot_sha256=slot.sha256,
            recipe_id=sha(f"recipe-{group_index}-{slot.slot_id}"),
            artifact_pcm_sha256=sha(f"pcm-{group_index}-{slot.slot_id}"),
            recipe_semantic_sha256=sha(
                f"semantic-{group_index}-{slot.slot_id}"
            ),
            recipe_slot_projection_sha256=slot.sha256,
            source_family_sha256=group.source_family_sha256,
            verifier_commit=git("artifact-verifier"),
        )
        for slot in candidate_panel.slots
    )
    encoded = float(group_index * 100 + time_index * 10 + band_index)
    features = np.asarray([0.0 if blocked else encoded, 1.0])
    return CounterfactualRiskRowV2(
        group_family=group,
        mixture_pcm_sha256=sha(f"mixture-{group_index}"),
        accompaniment_truth_pcm_sha256=sha(f"a-{group_index}"),
        vocal_truth_pcm_sha256=sha(f"v-{group_index}"),
        cell=CellGeometry(
            sample_rate_hz=44_100,
            resolution_ms=500,
            start_frame=time_index * 1000,
            end_frame=(time_index + 1) * 1000,
            band_low_hz=band_index * 500,
            band_high_hz=(band_index + 1) * 500,
            spectral_grid_sha256=sha("grid"),
        ),
        candidate_panel=candidate_panel,
        candidate_artifacts=artifacts,
        metric_names=("voice", "hole"),
        metric_units=("ratio", "ratio"),
        metric_directions=("lower_is_better", "lower_is_better"),
        features=features,
        exact_risks=np.asarray([[0.1, 0.2], [0.2, 0.1]]),
        available=np.ones((2, 2), dtype=bool),
        feature_contract_sha256=FEATURE_CONTRACT,
        metric_contract_sha256=METRIC_CONTRACT,
        route_policy_sha256=ROUTE_POLICY,
    )


def group_rows(group_index: int, *, blocked_cell=None):
    return tuple(
        exact_row(
            group_index,
            time_index,
            band_index,
            blocked=(blocked_cell == (time_index, band_index)),
        )
        for time_index in range(2)
        for band_index in range(2)
    )


def build_inputs(*, two_groups=True, blocked_cell=(0, 1), value_offset=0.0):
    rows = list(group_rows(0, blocked_cell=blocked_cell))
    if two_groups:
        rows.extend(group_rows(1))
    dataset = DatasetManifestV2.build(rows, source_commit=SOURCE_COMMIT)

    certificates = []
    for row in dataset.rows:
        blocked = (
            row.group_family.work_id == "work-0"
            and row.cell.start_frame == 0
            and row.cell.band_low_hz == 500
            and blocked_cell == (0, 1)
        )
        certificates.append(
            FeatureEvidenceCertificate(
                row_id=row.row_id,
                source_family_sha256=row.group_family.source_family_sha256,
                feature_contract_sha256=row.feature_contract_sha256,
                feature_sha256=row.feature_sha256,
                available=(not blocked, True),
                required=(True, True),
                missing_value_policy_sha256=sha("missing-policy"),
                extraction_report_sha256=sha(f"report-{row.row_id}"),
                extractor_bundle_sha256=sha("extractor"),
                verifier_commit=git("feature-verifier"),
            )
        )
    feature_registry = FeatureEvidenceRegistry.build(
        tuple(certificates), source_commit=SOURCE_COMMIT
    )
    inference = InferenceManifestV1.build(dataset, feature_registry)
    student = FakeStudent(inference, value_offset=value_offset)
    prediction = predict_upper_fail_closed(
        student,
        inference,
        prediction_policy_sha256=PREDICTION_POLICY,
    )

    partition_certificates = []
    for group_index in range(2 if two_groups else 1):
        by_cell = {
            row.cell.sha256: row
            for row in rows
            if row.group_family.work_id == f"work-{group_index}"
        }
        entries = []
        measures = (1, 2, 3, 4)
        ordered = group_rows(
            group_index,
            blocked_cell=(blocked_cell if group_index == 0 else None),
        )
        for index, row in enumerate(ordered):
            assert row.cell.sha256 in by_cell
            entries.append(
                CellPartitionEntry(
                    time_index=index // 2,
                    band_index=index % 2,
                    cell_sha256=row.cell.sha256,
                    measure=RationalMeasure(measures[index]),
                )
            )
        partition_certificates.append(
            CellPartitionCertificate(
                group_family_sha256=group_identity(group_index).sha256,
                spectral_grid_sha256=sha("grid"),
                resolution_ms=500,
                time_cell_count=2,
                band_count=2,
                entries=tuple(entries),
                time_ranges_sha256=sha(f"time-ranges-{group_index}"),
                frequency_ranges_sha256=sha(
                    f"frequency-ranges-{group_index}"
                ),
                measure_contract_sha256=sha("measure-contract"),
                exact_source_report_sha256=sha(
                    f"source-report-{group_index}"
                ),
                verifier_commit=git("partition-verifier"),
            )
        )
    partitions = CellPartitionRegistry.build(
        tuple(partition_certificates), source_commit=SOURCE_COMMIT
    )
    return dataset, inference, prediction, partitions, student


class FakeStudent:
    def __init__(self, inference, *, value_offset=0.0):
        self.candidate_count = 2
        self.metric_count = 2
        self.feature_count = 2
        self.candidate_panel_sha256 = inference.candidate_panel.sha256
        self.feature_contract_sha256 = (
            inference.rows[0].feature_contract_sha256
        )
        self.metric_contract_sha256 = inference.rows[0].metric_contract_sha256
        self._sha256 = sha("student")
        self.value_offset = value_offset

    @property
    def sha256(self):
        return self._sha256

    def validate(self):
        return None

    def predict_upper(self, features):
        values = np.asarray(features, dtype=np.float64)
        result = np.empty((len(values), 2, 2), dtype=np.float64)
        for index, feature in enumerate(values):
            base = float(feature.sum()) + self.value_offset
            result[index] = (
                (base + 0.1, base + 0.2),
                (base + 0.3, base + 0.4),
            )
        return result


def panel_by_work(registry, work_id):
    return next(panel for panel in registry.panels if panel.work_id == work_id)


def test_flat_predictions_are_reordered_to_certified_row_major_grids():
    _, inference, prediction, partitions, _ = build_inputs()
    result = assemble_partitioned_upper_risks(
        inference, prediction, partitions
    )
    assert len(result.panels) == 2
    first = panel_by_work(result, "work-0")
    assert first.upper.shape == (2, 2, 2, 2)
    assert first.available.shape == first.upper.shape
    assert first.cell_sha256s == tuple(
        row.cell.sha256 for row in group_rows(0, blocked_cell=(0, 1))
    )
    assert first.normalized_measure_matrix() == (
        (Fraction(1, 10), Fraction(1, 5)),
        (Fraction(3, 10), Fraction(2, 5)),
    )

    # The student base is feature sum.  Cell (1,0) has encoded feature 10 + 1.
    assert first.upper[1, 0, 0, 0] == pytest.approx(11.1)
    assert first.upper[1, 0, 1, 1] == pytest.approx(11.4)
    # Cell (0,1) is feature-blocked and must remain completely unavailable.
    assert first.row_prediction_allowed == (True, False, True, True)
    assert not first.available[0, 1].any()
    assert np.all(first.upper[0, 1] == 0.0)
    assert first.available[0, 0].all()
    assert first.available[1, 0].all()
    assert first.available[1, 1].all()


def test_repeated_cell_geometry_across_groups_remains_group_scoped():
    _, inference, prediction, partitions, _ = build_inputs()
    result = assemble_partitioned_upper_risks(
        inference, prediction, partitions
    )
    first = panel_by_work(result, "work-0")
    second = panel_by_work(result, "work-1")
    assert first.cell_sha256s == second.cell_sha256s
    assert first.group_family_sha256 != second.group_family_sha256
    assert first.partition_key != second.partition_key
    assert set(first.inference_row_sha256s).isdisjoint(
        second.inference_row_sha256s
    )


def test_outputs_are_read_only_and_registry_identity_is_truth_free():
    _, inference, prediction, partitions, _ = build_inputs()
    result = assemble_partitioned_upper_risks(
        inference, prediction, partitions
    )
    panel = result.panels[0]
    assert not panel.upper.flags.writeable
    assert not panel.available.flags.writeable
    with pytest.raises(ValueError):
        panel.upper[0, 0, 0, 0] = 99.0
    with pytest.raises(ValueError):
        panel.available[0, 0, 0, 0] = False
    document = result.identity_dict()
    serialized = repr(document).lower()
    for forbidden in (
        "accompaniment_truth",
        "vocal_truth",
        "exact_risk",
        "offline_row",
        "oracle_margin",
    ):
        assert forbidden not in serialized


def test_missing_or_unused_partition_certificates_are_refused():
    _, inference, prediction, partitions, _ = build_inputs()
    missing = CellPartitionRegistry.build(
        partitions.certificates[:1], source_commit=SOURCE_COMMIT
    )
    with pytest.raises(
        PartitionedPredictionError,
        match="missing_certificates",
    ):
        assemble_partitioned_upper_risks(inference, prediction, missing)

    extra_group = group_identity(9)
    base = partitions.certificates[0]
    unused = CellPartitionCertificate(
        **{
            **base.__dict__,
            "group_family_sha256": extra_group.sha256,
            "spectral_grid_sha256": sha("other-grid"),
            "exact_source_report_sha256": sha("unused-source-report"),
        }
    )
    expanded = CellPartitionRegistry.build(
        tuple(sorted((*partitions.certificates, unused), key=lambda item: item.partition_key)),
        source_commit=SOURCE_COMMIT,
    )
    with pytest.raises(
        PartitionedPredictionError,
        match="unused_certificates",
    ):
        assemble_partitioned_upper_risks(inference, prediction, expanded)


def test_partition_cell_set_must_exactly_match_inference_rows():
    _, inference, prediction, partitions, _ = build_inputs(two_groups=False)
    certificate = partitions.certificates[0]
    first = certificate.entries[0]
    forged = CellPartitionEntry(
        time_index=first.time_index,
        band_index=first.band_index,
        cell_sha256=sha("forged-cell"),
        measure=first.measure,
    )
    changed = CellPartitionCertificate(
        **{
            **certificate.__dict__,
            "entries": (forged, *certificate.entries[1:]),
        }
    )
    changed_registry = CellPartitionRegistry.build(
        (changed,), source_commit=SOURCE_COMMIT
    )
    with pytest.raises(
        PartitionedPredictionError,
        match="exactly cover the certified partition",
    ):
        assemble_partitioned_upper_risks(
            inference, prediction, changed_registry
        )


def test_prediction_and_inference_contract_substitutions_are_refused():
    _, inference, prediction, partitions, _ = build_inputs()
    mutations = (
        ("model_input_sha256", sha("other-input"), "model input"),
        ("source_commit", git("other-source"), "source commit"),
        ("candidate_panel_sha256", sha("other-panel"), "candidate panel"),
        ("feature_contract_sha256", sha("other-features"), "feature contract"),
        ("metric_contract_sha256", sha("other-metrics"), "metric contract"),
        ("route_policy_sha256", sha("other-route"), "route policy"),
        ("metric_names", ("voice", "other"), "metric names"),
    )
    for field, value, expected in mutations:
        damaged = PredictedUpperRiskManifestV1(
            **{**prediction.__dict__, field: value}
        )
        with pytest.raises(
            PartitionedPredictionError,
            match=expected,
        ):
            assemble_partitioned_upper_risks(
                inference, damaged, partitions
            )


def test_row_prediction_allowed_must_match_truth_free_feature_masks():
    _, inference, prediction, partitions, _ = build_inputs()
    changed = list(prediction.row_prediction_allowed)
    changed[0] = not changed[0]
    damaged = PredictedUpperRiskManifestV1(
        **{**prediction.__dict__, "row_prediction_allowed": tuple(changed)}
    )
    # Make the tensor internally consistent so the join check, not the prediction
    # manifest's own mask validation, catches the disagreement with inference.
    upper = np.asarray(damaged.upper).copy()
    available = np.asarray(damaged.available).copy()
    if changed[0]:
        upper[0] = 1.0
        available[0] = True
    else:
        upper[0] = 0.0
        available[0] = False
    damaged = PredictedUpperRiskManifestV1(
        **{**damaged.__dict__, "upper": upper, "available": available}
    )
    with pytest.raises(
        PartitionedPredictionError,
        match="prediction allowed",
    ):
        assemble_partitioned_upper_risks(
            inference, damaged, partitions
        )


def test_partition_measure_and_prediction_values_are_identity_bearing():
    _, inference, prediction, partitions, _ = build_inputs()
    first = assemble_partitioned_upper_risks(
        inference, prediction, partitions
    )

    changed_certificates = []
    for certificate in partitions.certificates:
        entries = tuple(
            CellPartitionEntry(
                time_index=entry.time_index,
                band_index=entry.band_index,
                cell_sha256=entry.cell_sha256,
                measure=RationalMeasure(entry.measure.numerator * 10),
            )
            for entry in certificate.entries
        )
        changed_certificates.append(
            CellPartitionCertificate(
                **{**certificate.__dict__, "entries": entries}
            )
        )
    changed_partitions = CellPartitionRegistry.build(
        tuple(changed_certificates), source_commit=SOURCE_COMMIT
    )
    changed_measure = assemble_partitioned_upper_risks(
        inference, prediction, changed_partitions
    )
    assert first.panels[0].normalized_measure_matrix() == (
        changed_measure.panels[0].normalized_measure_matrix()
    )
    assert first.sha256 != changed_measure.sha256

    _, other_inference, other_prediction, other_partitions, _ = build_inputs(
        value_offset=2.0
    )
    changed_values = assemble_partitioned_upper_risks(
        other_inference, other_prediction, other_partitions
    )
    assert first.sha256 != changed_values.sha256


def test_panel_validation_rejects_availability_contradictions_and_nonzero_unknowns():
    _, inference, prediction, partitions, _ = build_inputs(two_groups=False)
    result = assemble_partitioned_upper_risks(
        inference, prediction, partitions
    )
    panel = result.panels[0]
    blocked = panel.row_prediction_allowed.index(False)
    time_index, band_index = divmod(blocked, panel.band_count)

    exposed = np.asarray(panel.available).copy()
    exposed[time_index, band_index] = True
    upper = np.asarray(panel.upper).copy()
    upper[time_index, band_index] = 1.0
    with pytest.raises(
        PartitionedPredictionError,
        match="feature-blocked route cell",
    ):
        PartitionedUpperRiskPanelV1(
            **{**panel.__dict__, "available": exposed, "upper": upper}
        ).validate()

    nonzero = np.asarray(panel.upper).copy()
    nonzero[time_index, band_index, 0, 0] = 1.0
    with pytest.raises(
        PartitionedPredictionError,
        match="canonical zero",
    ):
        PartitionedUpperRiskPanelV1(
            **{**panel.__dict__, "upper": nonzero}
        ).validate()


def test_partition_registry_source_commit_must_match_inference():
    _, inference, prediction, partitions, _ = build_inputs()
    changed = CellPartitionRegistry(
        certificates=partitions.certificates,
        source_commit=git("other-source"),
    )
    with pytest.raises(
        PartitionedPredictionError,
        match="different source commits",
    ):
        assemble_partitioned_upper_risks(inference, prediction, changed)
