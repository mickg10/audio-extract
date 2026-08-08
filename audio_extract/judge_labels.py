"""Exact source-coordinate labels and group-atomic judge dataset rows.

The reference-free judge consumes ``(M, Y, D, task)`` at inference time, but its
supervision comes from cases where the retained accompaniment ``A`` and removed
voice ``V`` are known.  On each local tile this module fits

``Y = alpha * A + beta * V + R``

with weighted complex ridge regression.  The fit separates accompaniment
transfer, retained voice, and orthogonal artifacts.  Ill-conditioned tiles are
explicitly unavailable; they never become clean-looking zero labels.

The dataset layer stores immutable, content-addressed rows and enforces atomic
splits across every source-derived grouping key.  Lossy previews may be used as
weak training evidence, but are refused from threshold-calibration splits.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .challenges import EVALUATION_TASKS, PAIR_INTEGRITY

LABEL_SCHEMA = "audio-extract/source-coordinate-label/v1"
DATASET_ROW_SCHEMA = "audio-extract/judge-dataset-row/v1"
DATASET_SCHEMA = "audio-extract/judge-dataset/v1"

REFERENCE_GRADES = (
    "linear_exact",
    "lossless_reference",
    "lossless_same_take",
    "lossy_preview",
    "screening_only",
)
LOSSY_REFERENCE_GRADES = frozenset({"lossy_preview", "screening_only"})
DATASET_SPLITS = ("train", "validation", "test", "calibration")

_LABEL_DOMAIN = b"audio-extract-source-coordinate-label-v1\x00"
_ROW_DOMAIN = b"audio-extract-judge-dataset-row-v1\x00"
_EXAMPLE_DOMAIN = b"audio-extract-judge-example-v1\x00"
_EPS = np.finfo(np.float64).tiny


class LabelInputError(ValueError):
    """Exact-label inputs violate the sample-grid or numeric contract."""


class ImmutableRowError(RuntimeError):
    """The same logical training example was offered with different facts."""


class SplitLeakageError(ValueError):
    """Source-related examples were assigned to different dataset splits."""


class ThresholdEligibilityError(ValueError):
    """Evidence unsuitable for threshold calibration entered that split."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not finite JSON data: {exc}") from exc


def _hash(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + _json_bytes(value)).hexdigest()


def _finite_float(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _sha256_id(value: str, name: str) -> str:
    result = _nonempty(value, name)
    if not result.startswith("sha256:") or len(result) != 71:
        raise ValueError(f"{name} must be a sha256:<64 hex> identity")
    try:
        int(result[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256:<64 hex> identity") from exc
    return result.lower()


def _array(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.shape[0] == 0:
        raise LabelInputError(f"{name} must have a non-empty leading time axis")
    if not (np.issubdtype(arr.dtype, np.floating)
            or np.issubdtype(arr.dtype, np.complexfloating)):
        raise LabelInputError(f"{name} must contain real or complex samples")
    if not np.all(np.isfinite(arr)):
        raise LabelInputError(f"{name} contains non-finite samples")
    return arr.astype(np.complex128, copy=False)


def _weights(value: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == shape[0]:
        arr = arr.reshape((shape[0],) + (1,) * (len(shape) - 1))
    try:
        arr = np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise LabelInputError(
            f"weights shape {arr.shape} cannot broadcast to signal shape {shape}"
        ) from exc
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise LabelInputError("weights must be finite and non-negative")
    if not np.any(arr > 0):
        raise LabelInputError("weights must contain at least one positive value")
    return arr


@dataclass(frozen=True)
class SourceCoordinateLabel:
    """Serializable supervision for one local tile.

    Complex coefficients are represented by real/imaginary parts.  Supervision
    fields are ``None`` when ``available`` is false, while conditioning facts and
    the uncertainty remain visible for masking and diagnostics.
    """

    tile_index: int
    start_frame: int
    end_frame: int
    available: bool
    mask_reason: str | None
    alpha_real: float | None
    alpha_imag: float | None
    alpha_abs: float | None
    alpha_error_abs: float | None
    accompaniment_hole_db: float | None
    beta_real: float | None
    beta_imag: float | None
    retained_voice_coefficient: float | None
    retained_voice_energy_ratio: float | None
    retained_voice_energy_db: float | None
    orthogonal_artifact_ratio: float | None
    condition_number: float | None
    ridge_lambda: float
    uncertainty: float

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["schema"] = LABEL_SCHEMA
        payload["label_id"] = _hash(_LABEL_DOMAIN, payload)
        return payload


def fit_source_coordinates(
    candidate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    ridge_relative: float = 1e-8,
    max_condition: float = 1e6,
    min_source_energy: float = 1e-12,
    tile_index: int = 0,
    start_frame: int = 0,
) -> SourceCoordinateLabel:
    """Fit one weighted complex source-coordinate tile.

    Inputs must have exactly the same shape.  They may be waveforms shaped
    ``(time, channels)`` or complex representations shaped
    ``(time, frequency, channels)``; all non-time axes are jointly flattened.
    No alignment, truncation, resampling, or normalization is performed here.
    """

    y = _array(candidate, "candidate")
    a = _array(accompaniment, "accompaniment")
    v = _array(vocal, "vocal")
    if y.shape != a.shape or y.shape != v.shape:
        raise LabelInputError(
            "candidate, accompaniment, and vocal must have identical shapes; "
            f"got {y.shape}, {a.shape}, and {v.shape}"
        )
    ridge_relative = _finite_float(ridge_relative, "ridge_relative")
    max_condition = _finite_float(max_condition, "max_condition")
    min_source_energy = _finite_float(min_source_energy, "min_source_energy")
    if ridge_relative < 0 or max_condition <= 1 or min_source_energy < 0:
        raise ValueError(
            "ridge_relative and min_source_energy must be non-negative; "
            "max_condition must be greater than one"
        )

    w = _weights(weights, y.shape).reshape(-1)
    yf, af, vf = y.reshape(-1), a.reshape(-1), v.reshape(-1)
    x = np.column_stack((af, vf))
    gram = x.conj().T @ (w[:, None] * x)
    rhs = x.conj().T @ (w * yf)
    energies = np.maximum(np.real(np.diag(gram)), 0.0)
    ridge_scale = max(float(np.real(np.trace(gram))) / 2.0, _EPS)
    ridge_lambda = ridge_relative * ridge_scale

    condition: float | None
    try:
        raw_condition = float(np.linalg.cond(gram))
        condition = raw_condition if math.isfinite(raw_condition) else None
    except np.linalg.LinAlgError:
        condition = None

    mask_reason = None
    if np.any(energies <= min_source_energy):
        mask_reason = "source_energy_too_low"
    elif condition is None or condition > max_condition:
        mask_reason = "ill_conditioned_sources"

    uncertainty = 1.0
    if condition is not None:
        uncertainty = float(np.clip(
            math.log10(max(condition, 1.0)) / math.log10(max_condition), 0.0, 1.0
        ))

    alpha = beta = 0.0j
    residual_ratio = 0.0
    solve_failed = False
    try:
        alpha, beta = np.linalg.solve(
            gram + ridge_lambda * np.eye(2, dtype=np.complex128), rhs
        )
        residual = yf - alpha * af - beta * vf
        residual_energy = float(np.real(np.vdot(residual * np.sqrt(w), residual * np.sqrt(w))))
        residual_ratio = math.sqrt(max(residual_energy, 0.0) / max(energies[0], _EPS))
    except np.linalg.LinAlgError:
        solve_failed = True
        mask_reason = "ridge_solve_failed"

    available = mask_reason is None and not solve_failed
    end_frame = start_frame + y.shape[0]
    common = dict(
        tile_index=int(tile_index),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        available=available,
        mask_reason=mask_reason,
        condition_number=condition,
        ridge_lambda=float(ridge_lambda),
        uncertainty=uncertainty,
    )
    if not available:
        return SourceCoordinateLabel(
            **common,
            alpha_real=None,
            alpha_imag=None,
            alpha_abs=None,
            alpha_error_abs=None,
            accompaniment_hole_db=None,
            beta_real=None,
            beta_imag=None,
            retained_voice_coefficient=None,
            retained_voice_energy_ratio=None,
            retained_voice_energy_db=None,
            orthogonal_artifact_ratio=None,
        )

    alpha_abs = float(abs(alpha))
    beta_abs = float(abs(beta))
    voice_energy = beta_abs ** 2 * energies[1]
    retained_energy = alpha_abs ** 2 * energies[0]
    voice_ratio = float(voice_energy / max(retained_energy, _EPS))
    return SourceCoordinateLabel(
        **common,
        alpha_real=float(alpha.real),
        alpha_imag=float(alpha.imag),
        alpha_abs=alpha_abs,
        alpha_error_abs=float(abs(alpha - 1.0)),
        accompaniment_hole_db=float(max(-20.0 * math.log10(max(alpha_abs, _EPS)), 0.0)),
        beta_real=float(beta.real),
        beta_imag=float(beta.imag),
        retained_voice_coefficient=beta_abs,
        retained_voice_energy_ratio=voice_ratio,
        retained_voice_energy_db=float(10.0 * math.log10(max(voice_ratio, _EPS))),
        orthogonal_artifact_ratio=float(residual_ratio),
    )


def local_source_coordinate_labels(
    candidate: np.ndarray,
    accompaniment: np.ndarray,
    vocal: np.ndarray,
    *,
    tile_frames: int,
    hop_frames: int | None = None,
    min_tile_frames: int = 1,
    weights: np.ndarray | None = None,
    ridge_relative: float = 1e-8,
    max_condition: float = 1e6,
    min_source_energy: float = 1e-12,
) -> tuple[SourceCoordinateLabel, ...]:
    """Fit overlapping local labels along the leading time axis.

    A final partial tile is retained when it contains at least
    ``min_tile_frames``.  This keeps exact frame accounting explicit.
    """

    y = _array(candidate, "candidate")
    a = _array(accompaniment, "accompaniment")
    v = _array(vocal, "vocal")
    if y.shape != a.shape or y.shape != v.shape:
        raise LabelInputError(
            "candidate, accompaniment, and vocal must have identical shapes; "
            f"got {y.shape}, {a.shape}, and {v.shape}"
        )
    tile_frames = int(tile_frames)
    hop_frames = tile_frames if hop_frames is None else int(hop_frames)
    min_tile_frames = int(min_tile_frames)
    if tile_frames <= 0 or hop_frames <= 0 or min_tile_frames <= 0:
        raise ValueError("tile_frames, hop_frames, and min_tile_frames must be positive")

    full_weights = None if weights is None else _weights(weights, y.shape)
    labels = []
    tile_index = 0
    for start in range(0, y.shape[0], hop_frames):
        end = min(start + tile_frames, y.shape[0])
        if end - start < min_tile_frames:
            break
        labels.append(fit_source_coordinates(
            y[start:end],
            a[start:end],
            v[start:end],
            weights=None if full_weights is None else full_weights[start:end],
            ridge_relative=ridge_relative,
            max_condition=max_condition,
            min_source_energy=min_source_energy,
            tile_index=tile_index,
            start_frame=start,
        ))
        tile_index += 1
        if end == y.shape[0]:
            break
    return tuple(labels)


def label_uncertainty(labels: Iterable[SourceCoordinateLabel]) -> float:
    """Conservative row uncertainty: unavailable tiles force uncertainty 1."""

    values = tuple(labels)
    if not values or any(not label.available for label in values):
        return 1.0
    return float(max(label.uncertainty for label in values))


@dataclass(frozen=True)
class JudgeDatasetRow:
    """One immutable training example plus its leakage-control identity."""

    row_id: str
    example_key: str
    split: str
    split_group: str
    work_id: str
    corpus_id: str
    donor_ids: tuple[str, ...]
    source_lineage_ids: tuple[str, ...]
    additional_group_ids: tuple[str, ...]
    candidate_recipe_id: str
    challenge_id: str
    task: str
    reference_grade: str
    pair_integrity: str
    threshold_eligible: bool
    label_uncertainty: float
    labels_json: str
    metadata_json: str

    @classmethod
    def build(
        cls,
        *,
        split: str,
        split_group: str,
        work_id: str,
        corpus_id: str,
        donor_ids: Sequence[str],
        source_lineage_ids: Sequence[str],
        additional_group_ids: Sequence[str],
        candidate_recipe_id: str,
        challenge_id: str,
        task: str,
        reference_grade: str,
        pair_integrity: str,
        threshold_eligible: bool,
        label_uncertainty: float,
        labels: Sequence[SourceCoordinateLabel | Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> "JudgeDatasetRow":
        split = _nonempty(split, "split")
        if split not in DATASET_SPLITS:
            raise ValueError(f"unknown split {split!r}; allowed: {DATASET_SPLITS}")
        split_group = _nonempty(split_group, "split_group")
        work_id = _nonempty(work_id, "work_id")
        corpus_id = _nonempty(corpus_id, "corpus_id")
        donors = tuple(sorted({_nonempty(v, "donor_id") for v in donor_ids}))
        lineages = tuple(sorted({_nonempty(v, "source_lineage_id")
                                 for v in source_lineage_ids}))
        additional_groups = tuple(sorted({_nonempty(v, "additional_group_id")
                                          for v in additional_group_ids}))
        if not lineages:
            raise ValueError("source_lineage_ids must contain at least one source identity")
        candidate_recipe_id = _sha256_id(candidate_recipe_id, "candidate_recipe_id")
        challenge_id = _sha256_id(challenge_id, "challenge_id")
        task = _nonempty(task, "task")
        if task not in EVALUATION_TASKS:
            raise ValueError(f"unknown task {task!r}; allowed: {EVALUATION_TASKS}")
        reference_grade = _nonempty(reference_grade, "reference_grade")
        if reference_grade not in REFERENCE_GRADES:
            raise ValueError(
                f"unknown reference_grade {reference_grade!r}; allowed: {REFERENCE_GRADES}"
            )
        pair_integrity = _nonempty(pair_integrity, "pair_integrity")
        if pair_integrity not in PAIR_INTEGRITY:
            raise ValueError(
                f"unknown pair_integrity {pair_integrity!r}; allowed: {PAIR_INTEGRITY}"
            )
        uncertainty = _finite_float(label_uncertainty, "label_uncertainty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("label_uncertainty must be in [0, 1]")
        if threshold_eligible and reference_grade in LOSSY_REFERENCE_GRADES:
            raise ThresholdEligibilityError(
                f"{reference_grade} evidence cannot be threshold eligible"
            )
        if split == "calibration" and (
            not threshold_eligible or reference_grade in LOSSY_REFERENCE_GRADES
        ):
            raise ThresholdEligibilityError(
                "threshold-calibration rows require eligible non-lossy references"
            )

        label_dicts = [label.to_dict() if isinstance(label, SourceCoordinateLabel)
                       else dict(label) for label in labels]
        if not label_dicts:
            raise ValueError("labels must contain at least one exact-label observation")
        labels_json = _json_bytes(label_dicts).decode("utf-8")
        metadata_json = _json_bytes(dict(metadata or {})).decode("utf-8")
        key_payload = {
            "schema": DATASET_ROW_SCHEMA,
            "split_group": split_group,
            "work_id": work_id,
            "corpus_id": corpus_id,
            "donor_ids": donors,
            "source_lineage_ids": lineages,
            "additional_group_ids": additional_groups,
            "candidate_recipe_id": candidate_recipe_id,
            "challenge_id": challenge_id,
            "task": task,
            "reference_grade": reference_grade,
        }
        example_key = _hash(_EXAMPLE_DOMAIN, key_payload)
        row_payload = {
            **key_payload,
            "example_key": example_key,
            "split": split,
            "pair_integrity": pair_integrity,
            "threshold_eligible": bool(threshold_eligible),
            "label_uncertainty": uncertainty,
            "labels": label_dicts,
            "metadata": json.loads(metadata_json),
        }
        row_id = _hash(_ROW_DOMAIN, row_payload)
        return cls(
            row_id=row_id,
            example_key=example_key,
            split=split,
            split_group=split_group,
            work_id=work_id,
            corpus_id=corpus_id,
            donor_ids=donors,
            source_lineage_ids=lineages,
            additional_group_ids=additional_groups,
            candidate_recipe_id=candidate_recipe_id,
            challenge_id=challenge_id,
            task=task,
            reference_grade=reference_grade,
            pair_integrity=pair_integrity,
            threshold_eligible=bool(threshold_eligible),
            label_uncertainty=uncertainty,
            labels_json=labels_json,
            metadata_json=metadata_json,
        )

    def grouping_keys(self) -> tuple[str, ...]:
        keys = [
            f"split_group:{self.split_group}",
            f"work:{self.work_id}",
            f"corpus:{self.corpus_id}",
        ]
        keys.extend(f"donor:{value}" for value in self.donor_ids)
        keys.extend(f"source:{value}" for value in self.source_lineage_ids)
        keys.extend(f"additional:{value}" for value in self.additional_group_ids)
        return tuple(keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DATASET_ROW_SCHEMA,
            "row_id": self.row_id,
            "example_key": self.example_key,
            "split": self.split,
            "split_group": self.split_group,
            "work_id": self.work_id,
            "corpus_id": self.corpus_id,
            "donor_ids": list(self.donor_ids),
            "source_lineage_ids": list(self.source_lineage_ids),
            "additional_group_ids": list(self.additional_group_ids),
            "candidate_recipe_id": self.candidate_recipe_id,
            "challenge_id": self.challenge_id,
            "task": self.task,
            "reference_grade": self.reference_grade,
            "pair_integrity": self.pair_integrity,
            "threshold_eligible": self.threshold_eligible,
            "label_uncertainty": self.label_uncertainty,
            "labels": json.loads(self.labels_json),
            "metadata": json.loads(self.metadata_json),
        }


class JudgeDatasetBuilder:
    """Append-only rows with pre-commit split-leakage validation."""

    def __init__(self) -> None:
        self._rows_by_example: dict[str, JudgeDatasetRow] = {}
        self._group_splits: dict[str, str] = {}

    def add(self, row: JudgeDatasetRow) -> JudgeDatasetRow:
        existing = self._rows_by_example.get(row.example_key)
        if existing is not None:
            if existing != row:
                raise ImmutableRowError(
                    f"immutable judge row conflict for {row.example_key}: "
                    f"stored={existing.row_id}, offered={row.row_id}"
                )
            return existing

        conflicts = [
            (key, assigned)
            for key in row.grouping_keys()
            if (assigned := self._group_splits.get(key)) is not None
            and assigned != row.split
        ]
        if conflicts:
            key, assigned = conflicts[0]
            raise SplitLeakageError(
                f"group {key!r} is already assigned to {assigned!r}, "
                f"not {row.split!r}"
            )

        self._rows_by_example[row.example_key] = row
        for key in row.grouping_keys():
            self._group_splits[key] = row.split
        return row

    @property
    def rows(self) -> tuple[JudgeDatasetRow, ...]:
        return tuple(sorted(self._rows_by_example.values(), key=lambda row: row.row_id))

    def threshold_calibration_rows(self) -> tuple[JudgeDatasetRow, ...]:
        rows = tuple(row for row in self.rows if row.split == "calibration")
        for row in rows:
            if (not row.threshold_eligible
                    or row.reference_grade in LOSSY_REFERENCE_GRADES):
                raise ThresholdEligibilityError(
                    f"row {row.row_id} is not eligible for threshold calibration"
                )
        return rows

    def to_dict(self) -> dict[str, Any]:
        counts = {split: 0 for split in DATASET_SPLITS}
        for row in self.rows:
            counts[row.split] += 1
        return {
            "schema": DATASET_SCHEMA,
            "rows": [row.to_dict() for row in self.rows],
            "split_counts": counts,
        }

    def to_jsonl(self) -> str:
        return "".join(
            _json_bytes(row.to_dict()).decode("utf-8") + "\n" for row in self.rows
        )
