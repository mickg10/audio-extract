"""End-to-end certified O1/O2/O3 routing experiment runner.

The runner consumes verified/deduplicated accompaniment candidates plus exact
`M/A/V` truth, renders raw and common-STFT controls, solves globally certified
O2 and convex-certified O3, writes immutable FLOAT artifacts, computes the
complete-work metric bank, applies voiced routes to the Aalto no-vocal basis,
and emits the fail-closed architecture decision.

It never resamples, truncates, aligns, normalizes, or rewrites a candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import canon, identity
from .oracle_binding_preregistration import (
    PreregisteredRun,
    bind_report,
    binding_fields,
    canonical_resolution_sequence,
    verify_source_manifests,
)
from .oracle_binding_preregistration import (
    load as load_preregistration,
)
from .oracle_routing_basis_v2 import DeduplicatedCandidate
from .oracle_routing_binding_policy_v2 import evaluate_report_strict
from .oracle_routing_decision_v2 import (
    REPORT_SCHEMA,
    RoutingGateConfig,
)
from .oracle_routing_math_v2 import (
    best_whole_track,
    one_hot,
    route_objective,
    solve_o2_global,
    solve_o3_convex,
    unary_costs,
)
from .oracle_routing_metrics_v2 import (
    complete_work_metrics,
    no_vocal_false_positive_energy_ratio,
    route_boundary_metrics,
    worst_identifiable_event,
)
from .oracle_routing_run_contract_v3 import (
    REQUIRED_WORKS as V3_REQUIRED_WORKS,
)
from .oracle_routing_run_contract_v3 import (
    RunContractError,
    load_bound_json_artifact,
    preflight_run,
    require_canonical_resolutions,
)
from .oracle_routing_run_contract_v3 import (
    verify_report_binding as verify_v3_report_binding,
)
from .oracle_routing_spectral_v2 import (
    RoutingSpectralConfig,
    build_quadratic_grid,
    render_spectral_route,
    roundtrip_control,
    stft_stack,
    validate_exact_audio_basis,
)

RUN_SCHEMA = "audio-extract/oracle-routing-run/v2"
RECIPE_SCHEMA = "audio-extract/oracle-routing-artifact-recipe/v2"
TRUTH_SCHEMA = "audio-extract/oracle-routing-truth/v2"

DEFAULT_WORKS = (
    "bologna_verdi",
    "bologna_donizetti",
    "bologna_puccini",
    "aalto_mozart_dry",
)
DEFAULT_RESOLUTIONS = (2.0, 1.0, 0.5)
REQUIRED_ALIASES = (
    "median_mdx_mel_bs",
    "geomedian_mdx_mel_bs",
    "convex_fusion_uniform",
    "residual_mdx23c",
    "residual_melband",
    "residual_bs_roformer",
    "htdemucs_04573f0d",
    "htdemucs_955717e8",
)


class CertifiedRoutingRunError(RuntimeError):
    """The certified run cannot produce a binding, reproducible result."""


@dataclass(frozen=True)
class CertifiedRoutingRunConfig:
    spectral: RoutingSpectralConfig = field(default_factory=RoutingSpectralConfig)
    resolutions_seconds: tuple[float, ...] = DEFAULT_RESOLUTIONS
    base_resolution_seconds: float = 2.0
    temporal_switch_penalty_at_base: float = 0.05
    frequency_switch_penalty_at_base: float = 0.05
    temporal_smoothness_at_base: float = 0.05
    frequency_smoothness_at_base: float = 0.05
    o2_time_limit_seconds: float = 600.0
    o2_mip_relative_gap: float = 0.0
    o3_tolerance: float = 1e-7
    o3_max_iterations: int = 20_000
    truth_identity_tolerance: float = 2e-5

    def validate(self) -> None:
        self.spectral.validate()
        try:
            require_canonical_resolutions(self.resolutions_seconds)
        except RunContractError as exc:
            raise ValueError(
                f"certified routing resolutions violate run-contract v3: {exc}"
            ) from exc
        for name in (
            "base_resolution_seconds",
            "o2_time_limit_seconds",
            "o3_tolerance",
            "truth_identity_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "temporal_switch_penalty_at_base",
            "frequency_switch_penalty_at_base",
            "temporal_smoothness_at_base",
            "frequency_smoothness_at_base",
            "o2_mip_relative_gap",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.o3_max_iterations < 1:
            raise ValueError("o3_max_iterations must be positive")

    def penalties(self, resolution_seconds: float) -> dict[str, float]:
        """Scale graph penalties to hold one physical boundary approximately fixed.

        Data costs are averaged over cells, so one boundary contributes in
        proportion to the cell duration. Multiplying by base/resolution keeps a
        single physical route change comparable across the frozen sensitivity
        grid.
        """

        scale = self.base_resolution_seconds / float(resolution_seconds)
        return {
            "temporal_switch": self.temporal_switch_penalty_at_base * scale,
            "frequency_switch": self.frequency_switch_penalty_at_base * scale,
            "temporal_smoothness": self.temporal_smoothness_at_base * scale,
            "frequency_smoothness": self.frequency_smoothness_at_base * scale,
        }


@dataclass(frozen=True)
class ExactTruth:
    work_id: str
    mixture: np.ndarray
    accompaniment: np.ndarray
    vocal: np.ndarray
    sample_rate_hz: int
    frames: int
    pcm_identities: dict[str, str]


@dataclass(frozen=True)
class LoadedCandidate:
    row: DeduplicatedCandidate
    audio: np.ndarray


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _strict_audio(path: Path, *, frames: int | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    if info.samplerate != 44_100 or info.channels != 2 or info.subtype != "FLOAT":
        raise CertifiedRoutingRunError(
            f"expected 44.1-kHz stereo FLOAT: {path}: {info}"
        )
    if frames is not None and info.frames != frames:
        raise CertifiedRoutingRunError(
            f"frame mismatch for {path}: {info.frames} != {frames}"
        )
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape != (info.frames, 2) or not np.all(np.isfinite(audio)):
        raise CertifiedRoutingRunError(f"invalid decoded audio: {path}")
    return audio, int(sample_rate)


def _pcm(audio: np.ndarray, sample_rate_hz: int = 44_100) -> str:
    return identity.artifact_pcm_sha256(
        np.asarray(audio, dtype="float32"),
        sample_rate_hz,
        ["FL", "FR"],
        len(audio),
    )


def load_exact_truth(
    truth_root: Path,
    work_id: str,
    *,
    identity_tolerance: float = 2e-5,
) -> ExactTruth:
    root = truth_root / work_id
    paths = {
        "mixture": root / "mix_with_voice.wav",
        "accompaniment": root / "orchestra_only.wav",
        "vocal": root / "voice_ref.wav",
    }
    mixture, sample_rate = _strict_audio(paths["mixture"])
    accompaniment, _ = _strict_audio(paths["accompaniment"], frames=len(mixture))
    vocal, _ = _strict_audio(paths["vocal"], frames=len(mixture))
    residual = (
        mixture.astype(np.float64)
        - accompaniment.astype(np.float64)
        - vocal.astype(np.float64)
    )
    maximum = float(np.max(np.abs(residual)))
    if maximum > float(identity_tolerance):
        raise CertifiedRoutingRunError(
            f"{work_id}: exact truth M=A+V failed: {maximum} > {identity_tolerance}"
        )
    return ExactTruth(
        work_id=work_id,
        mixture=mixture,
        accompaniment=accompaniment,
        vocal=vocal,
        sample_rate_hz=sample_rate,
        frames=len(mixture),
        pcm_identities={
            name: _pcm(audio, sample_rate)
            for name, audio in (
                ("mixture", mixture),
                ("accompaniment", accompaniment),
                ("vocal", vocal),
            )
        },
    )


def rows_for_work(
    rows: Sequence[DeduplicatedCandidate], work_id: str
) -> tuple[DeduplicatedCandidate, ...]:
    result = tuple(row for row in rows if row.work_id == work_id)
    if not result:
        raise CertifiedRoutingRunError(f"no candidate basis for work {work_id}")
    aliases: dict[str, str] = {}
    for row in result:
        for alias in row.aliases:
            previous = aliases.setdefault(alias, row.artifact_pcm_sha256)
            if previous != row.artifact_pcm_sha256:
                raise CertifiedRoutingRunError(
                    f"ambiguous alias {work_id}/{alias}: {previous} != "
                    f"{row.artifact_pcm_sha256}"
                )
    missing = set(REQUIRED_ALIASES) - set(aliases)
    if missing:
        raise CertifiedRoutingRunError(
            f"work {work_id} lacks required basis aliases: {sorted(missing)}"
        )
    return result


def row_for_alias(
    rows: Sequence[DeduplicatedCandidate], alias: str
) -> DeduplicatedCandidate:
    matches = [row for row in rows if alias in row.aliases]
    if len(matches) != 1:
        raise CertifiedRoutingRunError(
            f"alias {alias!r} resolves to {len(matches)} decoded candidates"
        )
    return matches[0]


def load_candidates(
    rows: Sequence[DeduplicatedCandidate], *, frames: int
) -> tuple[LoadedCandidate, ...]:
    result = []
    for row in rows:
        path = Path(row.resolved_path)
        audio, sample_rate = _strict_audio(path, frames=frames)
        if sample_rate != row.sample_rate_hz:
            raise CertifiedRoutingRunError(
                f"sample-rate mismatch for {row.canonical_name}"
            )
        if _sha_file(path) != row.container_sha256:
            raise CertifiedRoutingRunError(
                f"container identity changed for {row.canonical_name}"
            )
        if _pcm(audio, sample_rate) != row.artifact_pcm_sha256:
            raise CertifiedRoutingRunError(
                f"decoded PCM identity changed for {row.canonical_name}"
            )
        result.append(LoadedCandidate(row, audio))
    return tuple(result)


def _raw_artifact(candidate: LoadedCandidate) -> dict[str, Any]:
    return {
        "path": candidate.row.resolved_path,
        "artifact_pcm_sha256": candidate.row.artifact_pcm_sha256,
        "container_sha256": candidate.row.container_sha256,
        "frames": candidate.row.frames,
        "sample_rate_hz": candidate.row.sample_rate_hz,
        "channels": list(candidate.row.channels),
        "subtype": candidate.row.subtype,
        "reopen_verified": True,
        "source_recipe_id": candidate.row.canonical_recipe_id,
        "aliases": list(candidate.row.aliases),
    }


def _canonical_recipe_value(value: Any) -> Any:
    """Materialize recipe decimals as exact strings accepted by v2 canon."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise CertifiedRoutingRunError(
                f"non-finite value in generated recipe: {value!r}"
            )
        return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_recipe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_recipe_value(item) for item in value]
    return value


def _verify_clean_source_tree(expected_commit: str) -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CertifiedRoutingRunError(
            "binding runner cannot verify its source-tree identity"
        ) from exc
    if commit.lower() != expected_commit.lower() or status:
        raise CertifiedRoutingRunError(
            "binding runner source tree is dirty or differs from the "
            "preregistered commit"
        )


def _plan_hash(plan: np.ndarray, header: Mapping[str, Any]) -> str:
    array = np.ascontiguousarray(plan)
    payload = (
        canon.canonicalize(
            _canonical_recipe_value(
                {
                    **dict(header),
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                }
            )
        )
        + array.tobytes()
    )
    return identity.blob_sha256(payload)


def _verify_generated(
    directory: Path,
    recipe: Mapping[str, Any],
    audio: np.ndarray,
    sample_rate_hz: int,
    plan: np.ndarray,
) -> dict[str, Any]:
    if json.loads((directory / "recipe.json").read_text()) != recipe:
        raise CertifiedRoutingRunError(f"generated recipe mismatch: {directory}")
    reopened, _ = _strict_audio(directory / "output.f32.wav", frames=len(audio))
    if not np.array_equal(reopened, np.asarray(audio, dtype="float32")):
        raise CertifiedRoutingRunError(f"generated FLOAT mismatch: {directory}")
    saved_plan = np.load(directory / "plan.npy", allow_pickle=False)
    if saved_plan.dtype != plan.dtype or not np.array_equal(saved_plan, plan):
        raise CertifiedRoutingRunError(f"generated routing plan mismatch: {directory}")
    artifact = _pcm(reopened, sample_rate_hz)
    container = _sha_file(directory / "output.f32.wav")
    if (directory / "output.pcm.sha256").read_text().strip() != artifact:
        raise CertifiedRoutingRunError(f"generated PCM sidecar mismatch: {directory}")
    if (directory / "container.sha256").read_text().strip() != container:
        raise CertifiedRoutingRunError(
            f"generated container sidecar mismatch: {directory}"
        )
    return {
        "path": str(directory / "output.f32.wav"),
        "artifact_pcm_sha256": artifact,
        "container_sha256": container,
        "frames": len(reopened),
        "sample_rate_hz": sample_rate_hz,
        "channels": ["FL", "FR"],
        "subtype": "FLOAT",
        "reopen_verified": True,
        "recipe_id": directory.name,
        "plan_sha256": recipe["routing_plan_sha256"],
    }


def write_generated_artifact(
    output_root: Path,
    *,
    work_id: str,
    resolution_seconds: float,
    method: str,
    audio: np.ndarray,
    sample_rate_hz: int,
    plan: np.ndarray,
    recipe_facts: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one immutable, content-addressed FLOAT result and exact plan."""

    audio = np.asarray(audio, dtype="float32")
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.all(np.isfinite(audio)):
        raise CertifiedRoutingRunError(f"invalid generated audio for {method}")
    plan = np.ascontiguousarray(plan)
    plan_sha = _plan_hash(
        plan,
        {
            "work_id": work_id,
            "resolution_seconds": float(resolution_seconds),
            "method": method,
        },
    )
    recipe = _canonical_recipe_value(
        {
            "schema": RECIPE_SCHEMA,
            "work_id": work_id,
            "resolution_seconds": float(resolution_seconds),
            "method": method,
            "routing_plan_sha256": plan_sha,
            **dict(recipe_facts),
        }
    )
    recipe_id = identity.blob_sha256(canon.canonicalize(recipe))
    final = (
        output_root / work_id / f"{float(resolution_seconds):.1f}s" / method / recipe_id
    )
    if final.is_dir():
        return _verify_generated(final, recipe, audio, sample_rate_hz, plan)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".route-", dir=final.parent))
    try:
        sf.write(
            temporary / "output.f32.wav",
            audio,
            sample_rate_hz,
            subtype="FLOAT",
        )
        reopened, _ = _strict_audio(temporary / "output.f32.wav", frames=len(audio))
        if not np.array_equal(reopened, audio):
            raise CertifiedRoutingRunError(
                f"FLOAT reopen mismatch before publish: {method}"
            )
        np.save(temporary / "plan.npy", plan, allow_pickle=False)
        (temporary / "recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "output.pcm.sha256").write_text(
            _pcm(reopened, sample_rate_hz) + "\n"
        )
        (temporary / "container.sha256").write_text(
            _sha_file(temporary / "output.f32.wav") + "\n"
        )
        (temporary / "COMPLETE").write_text("complete\n")
        try:
            os.replace(temporary, final)
        except OSError:
            if not final.is_dir():
                raise
        return _verify_generated(final, recipe, audio, sample_rate_hz, plan)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _evidence(
    candidate: np.ndarray,
    truth: ExactTruth,
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple]:
    metrics, labels = complete_work_metrics(
        candidate, truth.accompaniment, truth.vocal, truth.sample_rate_hz
    )
    return {
        "metrics": metrics,
        "artifact": dict(artifact),
        "worst_identifiable_event": worst_identifiable_event(labels),
    }, labels


def _o2_recompute(
    labels: np.ndarray, grid, temporal: float, frequency: float
) -> tuple[float, float, int, int]:
    unary = unary_costs(grid)
    chosen = np.take_along_axis(unary, labels[..., None], axis=-1)[..., 0]
    cells = labels.size
    data = float(np.sum(grid.normalized_measure() * chosen))
    temporal_switches = int(np.count_nonzero(labels[1:] != labels[:-1]))
    frequency_switches = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    total = (
        data
        + temporal * temporal_switches / cells
        + frequency * frequency_switches / cells
    )
    return total, data, temporal_switches, frequency_switches


def _basis_facts(candidates: Sequence[LoadedCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "canonical_name": item.row.canonical_name,
            "aliases": list(item.row.aliases),
            "canonical_recipe_id": item.row.canonical_recipe_id,
            "artifact_pcm_sha256": item.row.artifact_pcm_sha256,
            "container_sha256": item.row.container_sha256,
        }
        for item in candidates
    ]


def run_work_resolution(
    *,
    truth: ExactTruth,
    candidate_rows: Sequence[DeduplicatedCandidate],
    output_root: Path,
    resolution_seconds: float,
    config: CertifiedRoutingRunConfig,
    code_commit: str,
    preregistration_facts: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Run one work/resolution and return report, route plans, and runtime facts."""

    rows = rows_for_work(candidate_rows, truth.work_id)
    loaded = load_candidates(rows, frames=truth.frames)
    values, accompaniment, vocal = validate_exact_audio_basis(
        [item.audio for item in loaded], truth.accompaniment, truth.vocal
    )
    spectral_config = replace(
        config.spectral, time_cell_seconds=float(resolution_seconds)
    )
    candidate_spectra = stft_stack(values, spectral_config)
    truth_spectra = stft_stack([accompaniment, vocal], spectral_config)
    grid, cell_report = build_quadratic_grid(
        candidate_spectra, truth_spectra[0], truth_spectra[1], spectral_config
    )
    penalties = config.penalties(resolution_seconds)
    o1_index, o1_costs = best_whole_track(grid)
    o2 = solve_o2_global(
        grid,
        temporal_switch_penalty=penalties["temporal_switch"],
        frequency_switch_penalty=penalties["frequency_switch"],
        time_limit_seconds=config.o2_time_limit_seconds,
        mip_relative_gap=config.o2_mip_relative_gap,
    )
    o3 = solve_o3_convex(
        grid,
        temporal_smoothness=penalties["temporal_smoothness"],
        frequency_smoothness=penalties["frequency_smoothness"],
        tolerance=config.o3_tolerance,
        max_iterations=config.o3_max_iterations,
        o1_index=o1_index,
        o2_labels=o2.labels,
    )
    plans = {
        "O1": one_hot(np.full(grid.Q.shape[:2], o1_index, dtype=np.int64), len(loaded)),
        "O2": one_hot(o2.labels, len(loaded)),
        "O3": o3.weights,
    }
    outputs = {
        "O1": values[o1_index].copy(),
        "O2": render_spectral_route(
            candidate_spectra,
            plans["O2"],
            cell_report,
            frames=truth.frames,
            config=spectral_config,
        ),
        "O3": render_spectral_route(
            candidate_spectra,
            plans["O3"],
            cell_report,
            frames=truth.frames,
            config=spectral_config,
        ),
    }
    median_row = row_for_alias(rows, "median_mdx_mel_bs")
    mdx_row = row_for_alias(rows, "residual_mdx23c")
    median = next(item for item in loaded if item.row == median_row)
    mdx = next(item for item in loaded if item.row == mdx_row)
    median_identity = roundtrip_control(median.audio, spectral_config)
    o1_identity = roundtrip_control(outputs["O1"], spectral_config)

    common_recipe = {
        "code_commit": code_commit,
        **dict(preregistration_facts or {}),
        "truth_pcm": truth.pcm_identities,
        "basis": _basis_facts(loaded),
        "spectral_config": asdict(spectral_config),
        "penalties": penalties,
        "cell_modes": cell_report["mode_counts"],
    }
    methods: dict[str, Any] = {}
    methods["median_raw"], _ = _evidence(median.audio, truth, _raw_artifact(median))
    methods["residual_mdx23c"], _ = _evidence(mdx.audio, truth, _raw_artifact(mdx))
    o1_loaded = loaded[o1_index]
    methods["O1_raw"], _ = _evidence(outputs["O1"], truth, _raw_artifact(o1_loaded))

    for name, control, source_index in (
        ("median_stft_identity", median_identity, loaded.index(median)),
        ("O1_stft_identity", o1_identity, o1_index),
    ):
        plan = one_hot(
            np.full(grid.Q.shape[:2], source_index, dtype=np.int64), len(loaded)
        )
        artifact = write_generated_artifact(
            output_root,
            work_id=truth.work_id,
            resolution_seconds=resolution_seconds,
            method=name,
            audio=control["audio"],
            sample_rate_hz=truth.sample_rate_hz,
            plan=plan.astype("float64"),
            recipe_facts={**common_recipe, "control": "stft_identity"},
        )
        methods[name] = {
            "artifact": artifact,
            "identity_roundtrip": {
                "max_abs": control["max_abs"],
                "rms": control["rms"],
            },
        }

    o2_total, o2_data, o2_temporal, o2_frequency = _o2_recompute(
        o2.labels,
        grid,
        penalties["temporal_switch"],
        penalties["frequency_switch"],
    )
    if abs(o2_total - o2.objective) > 1e-9:
        raise CertifiedRoutingRunError(
            f"O2 independent objective mismatch: {o2_total} != {o2.objective}"
        )
    o3_total, o3_data, o3_temporal, o3_frequency = route_objective(
        o3.weights,
        grid,
        temporal_smoothness=penalties["temporal_smoothness"],
        frequency_smoothness=penalties["frequency_smoothness"],
    )
    if abs(o3_total - o3.objective) > 1e-9:
        raise CertifiedRoutingRunError(
            f"O3 independent objective mismatch: {o3_total} != {o3.objective}"
        )

    route_specs = {
        "O2_global_medoid": (
            outputs["O2"],
            o2.labels.astype("int32"),
            {
                "kind": "global_potts_milp",
                "success": True,
                "mip_gap": 0.0 if o2.mip_gap is None else o2.mip_gap,
                "objective": o2.objective,
                "data_objective": o2_data,
                "temporal_switches": o2_temporal,
                "frequency_switches": o2_frequency,
                "solver_status": o2.solver_status,
                "objective_recomputed": True,
                "complete_grid": True,
            },
        ),
        "O3_certified_convex": (
            outputs["O3"],
            o3.weights.astype("float64"),
            {
                "kind": "convex_projected_gradient",
                "converged": True,
                "monotone_objective": True,
                "starts_agree": True,
                "projected_gradient_norm": o3.projected_gradient_norm,
                "max_simplex_error": float(np.max(np.abs(o3.weights.sum(-1) - 1.0))),
                "objective": o3.objective,
                "data_objective": o3_data,
                "temporal_smoothness": o3_temporal,
                "frequency_smoothness": o3_frequency,
                "iterations": o3.iterations,
                "start_objectives": o3.start_objectives,
                "objective_recomputed": True,
                "complete_grid": True,
            },
        ),
    }
    for name, (audio, plan, certificate) in route_specs.items():
        artifact = write_generated_artifact(
            output_root,
            work_id=truth.work_id,
            resolution_seconds=resolution_seconds,
            method=name,
            audio=audio,
            sample_rate_hz=truth.sample_rate_hz,
            plan=plan,
            recipe_facts={
                **common_recipe,
                "optimizer_certificate": certificate,
            },
        )
        evidence, _ = _evidence(audio, truth, artifact)
        evidence["optimizer_certificate"] = certificate
        evidence["seam_check"] = route_boundary_metrics(
            audio,
            truth.accompaniment,
            truth.sample_rate_hz,
            time_frame_ranges=cell_report["time_frame_ranges"],
            frequency_bin_ranges=cell_report["frequency_bin_ranges"],
            n_fft=spectral_config.n_fft,
            hop_length=spectral_config.hop_length,
            plan=plan,
        )
        methods[name] = evidence

    work_report = {
        "truth": {
            "schema": TRUTH_SCHEMA,
            "frames": truth.frames,
            "sample_rate_hz": truth.sample_rate_hz,
            "channels": ["FL", "FR"],
            "pcm_identities": truth.pcm_identities,
        },
        "basis": _basis_facts(loaded),
        "cell_report": cell_report,
        "O1": {
            "selected_index": o1_index,
            "selected_canonical_name": loaded[o1_index].row.canonical_name,
            "whole_track_costs": {
                loaded[index].row.canonical_name: float(value)
                for index, value in enumerate(o1_costs)
            },
        },
        "methods": methods,
    }
    runtime = {
        "loaded": loaded,
        "candidate_spectra": candidate_spectra,
        "cell_report": cell_report,
        "spectral_config": spectral_config,
        "o1_index": o1_index,
    }
    return work_report, plans, runtime


def _match_control_rows(
    voiced: Sequence[LoadedCandidate],
    control_rows: Sequence[DeduplicatedCandidate],
    *,
    frames: int,
) -> tuple[LoadedCandidate, ...]:
    controls = load_candidates(control_rows, frames=frames)
    result = []
    for candidate in voiced:
        matches = [
            control
            for control in controls
            if set(control.row.aliases) & set(candidate.row.aliases)
        ]
        if len(matches) != 1:
            raise CertifiedRoutingRunError(
                f"no-vocal counterpart for {candidate.row.canonical_name} "
                f"resolves to {len(matches)} rows"
            )
        result.append(matches[0])
    return tuple(result)


def apply_no_vocal_controls(
    *,
    truth: ExactTruth,
    voiced_runtime: Mapping[str, Any],
    plans: Mapping[str, np.ndarray],
    no_vocal_rows: Sequence[DeduplicatedCandidate],
    output_root: Path,
    resolution_seconds: float,
    code_commit: str,
    preregistration_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    voiced = tuple(voiced_runtime["loaded"])
    controls = _match_control_rows(
        voiced,
        rows_for_work(no_vocal_rows, truth.work_id),
        frames=truth.frames,
    )
    spectral_config = voiced_runtime["spectral_config"]
    control_spectra = stft_stack([item.audio for item in controls], spectral_config)
    median = next(item for item in controls if "median_mdx_mel_bs" in item.row.aliases)
    result: dict[str, Any] = {
        "median_raw": {
            "false_positive_energy_ratio": no_vocal_false_positive_energy_ratio(
                truth.accompaniment, median.audio
            ),
            "artifact": _raw_artifact(median),
        }
    }
    for report_name, plan_name in (
        ("O2_global_medoid", "O2"),
        ("O3_certified_convex", "O3"),
    ):
        plan = np.asarray(plans[plan_name])
        output = render_spectral_route(
            control_spectra,
            plan,
            voiced_runtime["cell_report"],
            frames=truth.frames,
            config=spectral_config,
        )
        artifact = write_generated_artifact(
            output_root,
            work_id=truth.work_id + "__no_vocal",
            resolution_seconds=resolution_seconds,
            method=report_name,
            audio=output,
            sample_rate_hz=truth.sample_rate_hz,
            plan=plan,
            recipe_facts={
                "code_commit": code_commit,
                **dict(preregistration_facts or {}),
                "control": "no_vocal",
                "source_pcm_sha256": truth.pcm_identities["accompaniment"],
                "basis": _basis_facts(controls),
                "spectral_config": asdict(spectral_config),
            },
        )
        result[report_name] = {
            "false_positive_energy_ratio": no_vocal_false_positive_energy_ratio(
                truth.accompaniment, output
            ),
            "artifact": artifact,
        }
    return result


def run_experiment(
    *,
    basis_rows: Sequence[DeduplicatedCandidate],
    no_vocal_basis_rows: Sequence[DeduplicatedCandidate] | None,
    truth_root: Path,
    output_root: Path,
    code_commit: str,
    preregistration: PreregisteredRun,
    run_input_path: Path | None = None,
    run_claim_path: Path | None = None,
    works: Sequence[str] = DEFAULT_WORKS,
    config: CertifiedRoutingRunConfig | None = None,
    decision_config: RoutingGateConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if run_input_path is None or run_claim_path is None:
        raise CertifiedRoutingRunError(
            "binding run requires a v3 run input and external pre-run claim"
        )
    # This must remain the first filesystem-dependent runner operation. It
    # verifies every input dependency and refuses an already-created output root.
    try:
        preflight = preflight_run(run_input_path, run_claim_path)
    except RunContractError as exc:
        raise CertifiedRoutingRunError(f"v3 run preflight failed: {exc}") from exc
    expected_output = Path(preflight.run_input.document["output_root"])
    if output_root.resolve(strict=False) != expected_output:
        raise CertifiedRoutingRunError(
            "runner output_root differs from the v3 precommitted output root"
        )
    if code_commit.lower() != preflight.run_input.document["source_commit"]:
        raise CertifiedRoutingRunError(
            "execution code commit differs from v3 run input"
        )
    if tuple(works) != V3_REQUIRED_WORKS:
        raise CertifiedRoutingRunError(
            "runner works differ from the v3 frozen work set/order"
        )
    if not isinstance(preregistration, PreregisteredRun):
        raise CertifiedRoutingRunError(
            "binding run requires a preexisting verified preregistration witness"
        )
    initial = load_preregistration(Path(preregistration.path))
    if initial != preregistration:
        raise CertifiedRoutingRunError(
            "preregistration changed between caller verification and runner entry"
        )
    legacy_record = preflight.run_input.document["legacy_preregistration"]
    if (
        legacy_record["path"] != initial.path
        or legacy_record["sha256"] != initial.container_sha256
    ):
        raise CertifiedRoutingRunError(
            "v3 run input binds a different legacy preregistration witness"
        )
    v3_source_groups = {
        group: [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in preflight.run_input.document["source_manifests"][group]
        ]
        for group in ("voiced", "no_vocal")
    }
    legacy_source_groups = {
        group: [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in initial.document["source_manifests"][group]
        ]
        for group in ("voiced", "no_vocal")
    }
    if v3_source_groups != legacy_source_groups:
        raise CertifiedRoutingRunError(
            "legacy witness source groups differ from v3 run input"
        )
    if code_commit.lower() != initial.document["source_commit"]:
        raise CertifiedRoutingRunError(
            "execution code commit differs from preregistration"
        )
    _verify_clean_source_tree(code_commit)
    verify_source_manifests(initial)
    cfg = config or CertifiedRoutingRunConfig()
    legacy_resolution_keys = canonical_resolution_sequence(
        initial.document["resolutions_seconds"]
    )
    resolution_keys = tuple(
        item["seconds_decimal"] for item in preflight.run_input.document["resolutions"]
    )
    v3_resolutions = require_canonical_resolutions(resolution_keys)
    if {float(value) for value in legacy_resolution_keys} != set(v3_resolutions):
        raise CertifiedRoutingRunError(
            "legacy witness resolution set differs from v3 run input"
        )
    cfg = replace(
        cfg,
        resolutions_seconds=v3_resolutions,
    )
    cfg.validate()
    effective_decision_config = decision_config or RoutingGateConfig()
    effective_decision_config.validate()
    expected_routing_config = {
        "schema": "audio-extract/oracle-routing-config-binding/v1",
        "works": list(works),
        "run_config": {
            **asdict(cfg),
            "spectral": asdict(cfg.spectral),
        },
        "decision_config": asdict(effective_decision_config),
    }
    try:
        bound_routing_config = load_bound_json_artifact(
            preflight.run_input, "routing_config"
        )
    except RunContractError as exc:
        raise CertifiedRoutingRunError(
            f"cannot reopen v3 routing config: {exc}"
        ) from exc
    if bound_routing_config != expected_routing_config:
        raise CertifiedRoutingRunError(
            "runner config differs from the v3 bound routing config"
        )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "run_schema": RUN_SCHEMA,
        "diagnostic_only": True,
        "code_commit": code_commit,
        "config": {
            **asdict(cfg),
            "spectral": asdict(cfg.spectral),
        },
        "works": list(works),
        "resolutions": {},
        "run_input_binding": dict(preflight.binding),
        **binding_fields(initial),
    }
    bind_report(report, initial)
    recipe_preregistration = {
        "preregistration": binding_fields(initial),
        "run_input_binding": dict(preflight.binding),
    }
    truths = {
        work: load_exact_truth(
            truth_root,
            work,
            identity_tolerance=cfg.truth_identity_tolerance,
        )
        for work in works
    }
    for resolution_key, resolution in zip(
        resolution_keys, cfg.resolutions_seconds, strict=True
    ):
        resolution_report = {"works": {}}
        for work in works:
            work_report, plans, runtime = run_work_resolution(
                truth=truths[work],
                candidate_rows=basis_rows,
                output_root=output_root,
                resolution_seconds=resolution,
                config=cfg,
                code_commit=code_commit,
                preregistration_facts=recipe_preregistration,
            )
            if work == "aalto_mozart_dry" and no_vocal_basis_rows is not None:
                work_report["no_vocal"] = apply_no_vocal_controls(
                    truth=truths[work],
                    voiced_runtime=runtime,
                    plans=plans,
                    no_vocal_rows=no_vocal_basis_rows,
                    output_root=output_root,
                    resolution_seconds=resolution,
                    code_commit=code_commit,
                    preregistration_facts=recipe_preregistration,
                )
            resolution_report["works"][work] = work_report
        report["resolutions"][resolution_key] = resolution_report

    decision = evaluate_report_strict(
        report,
        metric_config=effective_decision_config,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    decision_payload = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    report_path = output_root / "oracle-routing-envelope-v2.json"
    decision_path = output_root / "binding-decision-v2.json"
    output_root.mkdir(parents=True, exist_ok=True)
    for path, text in (
        (report_path, payload),
        (decision_path, decision_payload),
    ):
        if path.exists() and path.read_text() != text:
            raise CertifiedRoutingRunError(
                f"refusing to replace differing completed report: {path}"
            )
        if not path.exists():
            path.write_text(text)
    final = load_preregistration(Path(initial.path))
    if final != initial:
        raise CertifiedRoutingRunError(
            "preregistration changed while routing outputs were being written"
        )
    verify_source_manifests(final)
    bind_report(report, final)
    try:
        verify_v3_report_binding(report, run_input_path, run_claim_path)
    except RunContractError as exc:
        raise CertifiedRoutingRunError(
            f"v3 report binding changed during execution: {exc}"
        ) from exc
    return report, decision
