#!/usr/bin/env python3
"""Export the accepted exact O2 route as masked training-only teacher labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from audio_extract import identity
from audio_extract.oracle_convex import ConvexOracleConfig
from audio_extract.oracle_routing import (
    RoutingConfig,
    source_coordinate_statistics,
    stft_stack,
)
from audio_extract.oracle_tail import corrected_quadratic
from tools.oracle_routing_envelope import (
    BASIS_ORDER,
    _pcm,
    _read_exact,
    _route_plan_hash,
    _sha_file,
)


def _config(payload: dict) -> RoutingConfig:
    values = dict(payload)
    values["band_edges_hz"] = tuple(values["band_edges_hz"])
    return RoutingConfig(**values)


def _metadata_dir(root: Path, work: str, recipe_id: str) -> Path:
    return root / work / "candidates" / recipe_id.replace(":", "_")


def run(args: argparse.Namespace) -> dict:
    report = json.loads(args.routing_report.read_text())
    if (report.get("schema") != "audio-extract/oracle-routing-envelope/v1"
            or report.get("status") != "final"):
        raise ValueError("wrong final oracle routing report")
    if tuple(report.get("candidate_basis", [])) != BASIS_ORDER:
        raise ValueError("oracle basis order changed")
    config = _config(report["config"])
    convex = ConvexOracleConfig()
    output_root = args.output_root
    if output_root.exists():
        raise RuntimeError(f"refusing to rewrite O2 teacher export: {output_root}")
    output_root.mkdir(parents=True)
    works = []
    for work, work_report in report["works"].items():
        basis = work_report["basis"]
        if tuple(row["name"] for row in basis) != BASIS_ORDER:
            raise ValueError(f"{work} basis order changed")
        values = []
        for row in basis:
            path = Path(row["path"])
            value = _read_exact(path, int(work_report["frames"]))
            if (_sha_file(path) != row["container_sha256"]
                    or _pcm(value) != row["artifact_pcm_sha256"]):
                raise ValueError(f"{work}/{row['name']} basis identity mismatch")
            values.append(value)
        truth = args.truth_root / work
        accompaniment = _read_exact(
            truth / "orchestra_only.wav", int(work_report["frames"])
        )
        vocal = _read_exact(truth / "voice_ref.wav", int(work_report["frames"]))
        spectra = stft_stack(values, config)
        truth_spectra = stft_stack([accompaniment, vocal], config)
        stats = source_coordinate_statistics(
            spectra, truth_spectra[0], truth_spectra[1], config
        )
        _, corrected_unary = corrected_quadratic(stats, convex)
        stats = replace(stats, unary_risk=corrected_unary)

        output = work_report["outputs"]["O2"]
        directory = _metadata_dir(args.execution_root, work, output["recipe_id"])
        recipe = json.loads((directory / "recipe.json").read_text())
        execution = json.loads((directory / "execution.json").read_text())
        if identity.recipe_id(recipe) != output["recipe_id"]:
            raise ValueError(f"{work} O2 recipe ID mismatch")
        labels = np.asarray(execution["routing_plan"], dtype=np.int32)
        if labels.shape != stats.available.shape:
            raise ValueError(f"{work} O2 label grid mismatch: {labels.shape}")
        if np.any(labels < 0) or np.any(labels >= len(BASIS_ORDER)):
            raise ValueError(f"{work} O2 labels are out of range")
        solver_config = recipe["effective_config"]["solver_config"]
        plan_hash = _route_plan_hash("O2", labels, config, solver_config)
        declared_hash = recipe["effective_config"]["routing_plan_sha256"]
        if plan_hash != declared_hash or plan_hash != output["routing_plan_sha256"]:
            raise ValueError(f"{work} O2 routing-plan hash mismatch")
        available_count = int(stats.available.sum())
        if available_count != int(work_report["identifiability"]["available_cells"]):
            raise ValueError(f"{work} identifiability mask count mismatch")
        chosen = np.take_along_axis(
            corrected_unary, labels[..., None], axis=-1
        )[..., 0]
        data_objective = float(chosen[stats.available].mean())
        if not np.isclose(
            data_objective, float(work_report["O2"]["data_objective"]),
            rtol=1e-10, atol=1e-12,
        ):
            raise ValueError(f"{work} O2 corrected data objective mismatch")
        o1_index = int(work_report["O1"]["selected_index"])
        reductions = corrected_unary[..., o1_index] - chosen
        contributions = {}
        for index, name in enumerate(BASIS_ORDER):
            selected = stats.available & (labels == index)
            count = int(selected.sum())
            total = float(reductions[selected].sum()) if count else 0.0
            contributions[name] = {
                "selected_available_cells": count,
                "corrected_risk_reduction_total": total,
                "corrected_risk_reduction_per_selected_cell": (
                    total / count if count else 0.0
                ),
                "corrected_risk_reduction_per_available_cell": total / available_count,
            }
        work_dir = output_root / work
        work_dir.mkdir()
        labels_path = work_dir / "o2-labels.i32.npy"
        available_path = work_dir / "identifiable-mask.bool.npy"
        np.save(labels_path, labels, allow_pickle=False)
        np.save(available_path, stats.available.astype(bool), allow_pickle=False)
        works.append({
            "work_id": work, "basis_order": list(BASIS_ORDER),
            "labels_path": str(labels_path.resolve()),
            "labels_sha256": _sha_file(labels_path),
            "available_path": str(available_path.resolve()),
            "available_sha256": _sha_file(available_path),
            "shape": list(labels.shape), "available_cells": available_count,
            "masked_cells": int((~stats.available).sum()),
            "route_recipe_id": output["recipe_id"],
            "route_plan_sha256": plan_hash,
            "o1_selected_member": work_report["O1"]["selected_member"],
            "selection_counts": work_report["O2"]["selection_counts"],
            "temporal_switches": work_report["O2"]["temporal_switches"],
            "frequency_switches": work_report["O2"]["frequency_switches"],
            "corrected_data_objective": data_objective,
            "per_member_corrected_risk_reduction": contributions,
        })
    result = {
        "schema": "audio-extract/o2-teacher-labels/v1", "status": "complete",
        "training_only": True, "truth_is_not_an_inference_input": True,
        "source_routing_report": str(args.routing_report.resolve()),
        "source_routing_report_sha256": _sha_file(args.routing_report),
        "candidate_basis": list(BASIS_ORDER), "config": report["config"],
        "works": works,
    }
    path = output_root / "teacher-report.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_root / "COMPLETE").write_text("")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-report", required=True, type=Path)
    parser.add_argument("--execution-root", required=True, type=Path)
    parser.add_argument("--truth-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps({
        "status": report["status"], "works": len(report["works"]),
        "output": str(args.output_root / "teacher-report.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
