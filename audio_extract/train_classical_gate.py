"""Deterministic 100-step trainer for the bounded classical routing gate.

All separator estimates are immutable precomputed inputs.  The only optimized
parameters belong to :class:`SmoothResidualGate`; separators never appear in
the optimizer or forward graph.  Training and complete-work evaluation use the
same gate, exact FLOAT grids, explicit A-only/V-only controls, and the versioned
single-residual loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from . import identity, recipe as recipe_mod
from .classical_baselines import _ensure_exact_source
from .classical_gate import (
    SmoothGateConfig,
    SmoothResidualGate,
    gate_bundle_identity,
    gate_regularization,
)
from .classical_loss_v2 import ClassicalResidualLossConfig, classical_residual_loss_v2
from .classical_release import exact_metrics
from .storage import ImmutableWriteError, TrackLayout


SR = 44_100
MANIFEST_SCHEMA = "audio-extract/classical-gate-inputs/v1"
CONFIG_SCHEMA = "audio-extract/train/classical-smooth-gate/v1"
REPORT_SCHEMA = "audio-extract/classical-smooth-gate-run/v1"


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - owned by the train extra
        raise RuntimeError("classical gate training requires PyTorch") from exc
    return torch


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def audio_record(path: Path, *, recipe_id: str | None = None,
                 executed_bundle_hash: str | None = None) -> dict[str, Any]:
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype) != (SR, 2, "FLOAT"):
        raise ValueError(f"gate input must be 44.1-kHz stereo FLOAT: {path}: {info}")
    value, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid gate input samples: {path}")
    result = {
        "path": str(path.resolve()), "container_sha256": _sha_file(path),
        "artifact_pcm_sha256": identity.artifact_pcm_sha256(
            value, SR, ["FL", "FR"], len(value)
        ),
        "frames": len(value), "sample_rate_hz": SR,
        "channels": ["FL", "FR"], "subtype": "FLOAT",
    }
    if recipe_id is not None:
        result["recipe_id"] = recipe_id
    if executed_bundle_hash is not None:
        result["executed_bundle_hash"] = executed_bundle_hash
    return result


def _verify_record(record: dict[str, Any], label: str) -> Path:
    required = {
        "path", "container_sha256", "artifact_pcm_sha256", "frames",
        "sample_rate_hz", "channels", "subtype",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"{label} lacks record fields: {sorted(missing)}")
    path = Path(record["path"])
    actual = audio_record(path)
    for key in required - {"path"}:
        if actual[key] != record[key]:
            raise ValueError(
                f"{label}/{key} mismatch: {actual[key]} != {record[key]}"
            )
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"wrong gate manifest schema: {document.get('schema')}")
    members = document.get("members")
    if not isinstance(members, dict) or set(members) != {"conservative", "aggressive"}:
        raise ValueError("gate manifest must name conservative and aggressive members")
    if members["conservative"] == members["aggressive"]:
        raise ValueError("gate members must be distinct")
    works = document.get("works")
    if not isinstance(works, list) or not works:
        raise ValueError("gate manifest has no works")
    ids = [row.get("work_id") for row in works]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("gate manifest has an invalid work ID")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate gate work IDs: {ids}")
    for row in works:
        work = row["work_id"]
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError(f"{work} lacks an independent group ID")
        truth = row.get("truth")
        if not isinstance(truth, dict) or set(truth) != {
            "mixture", "accompaniment", "vocal"
        }:
            raise ValueError(f"{work} must have exact M/A/V truth")
        estimates = row.get("estimates")
        if not isinstance(estimates, dict) or set(estimates) != {
            "mixture", "no_vocal", "vocal_only"
        }:
            raise ValueError(f"{work} must have mixture/A-only/V-only estimates")
        records = [*truth.values()]
        for control, pair in estimates.items():
            if not isinstance(pair, dict) or set(pair) != {
                "conservative", "aggressive"
            }:
                raise ValueError(f"{work}/{control} must contain both gate members")
            records.extend(pair.values())
        grids = set()
        for index, record in enumerate(records):
            _verify_record(record, f"{work}/record[{index}]")
            grids.add((record["frames"], record["sample_rate_hz"],
                       tuple(record["channels"]), record["subtype"]))
        if len(grids) != 1:
            raise ValueError(f"{work} gate inputs do not share one exact grid: {grids}")
        activity_path = Path(row["vocal_activity_path"])
        activity = np.load(activity_path, mmap_mode="r")
        if activity.ndim != 1 or len(activity) != int(records[0]["frames"]):
            raise ValueError(f"{work} vocal activity is off-grid: {activity.shape}")
        if not np.all(np.isfinite(activity)) or np.any(activity < 0):
            raise ValueError(f"{work} vocal activity must be finite and non-negative")
    return document


def _read(path: Path, *, start: int = 0, frames: int | None = None):
    stop = None if frames is None else start + frames
    value, sr = sf.read(
        path, dtype="float32", always_2d=True, start=start, stop=stop
    )
    if sr != SR or (frames is not None and len(value) != frames):
        raise ValueError(f"short or off-grid gate read: {path}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"non-finite gate read: {path}")
    return _torch().from_numpy(value.T.copy())


class GateDataset:
    def __init__(self, manifest: dict[str, Any], work_ids: list[str], crop_frames: int):
        by_id = {row["work_id"]: row for row in manifest["works"]}
        missing = sorted(set(work_ids) - set(by_id))
        if missing:
            raise ValueError(f"gate works missing from manifest: {missing}")
        self.items = []
        self.crop_frames = crop_frames
        for work in work_ids:
            row = by_id[work]
            frames = int(row["truth"]["mixture"]["frames"])
            if frames < crop_frames:
                raise ValueError(f"{work} is shorter than one gate crop")
            self.items.append((work, row, frames))

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        work, row, total = self.items[int(rng.integers(len(self.items)))]
        start = int(rng.integers(0, total - self.crop_frames + 1))
        return self.read(row, start=start, frames=self.crop_frames) | {
            "work_id": work, "start_frame": start,
        }

    @staticmethod
    def read(row: dict[str, Any], *, start: int = 0,
             frames: int | None = None) -> dict[str, Any]:
        truth = {
            role: _read(Path(record["path"]), start=start, frames=frames)
            for role, record in row["truth"].items()
        }
        estimates = {
            control: {
                member: _read(Path(record["path"]), start=start, frames=frames)
                for member, record in pair.items()
            }
            for control, pair in row["estimates"].items()
        }
        activity = np.asarray(
            np.load(row["vocal_activity_path"], mmap_mode="r")[
                start:None if frames is None else start + frames
            ], dtype="float32",
        )
        if frames is not None and len(activity) != frames:
            raise ValueError(f"short activity read for {row['work_id']}")
        return {
            "truth": truth, "estimates": estimates,
            "activity": _torch().from_numpy(activity.copy()).view(1, -1),
            "row": row,
        }


def _forward_loss(gate: SmoothResidualGate, batch: dict[str, Any], *, device: str,
                  loss_config: ClassicalResidualLossConfig):
    torch = _torch()
    truth = {key: value.unsqueeze(0).to(device) for key, value in batch["truth"].items()}
    estimates = {
        control: {
            member: value.unsqueeze(0).to(device) for member, value in pair.items()
        }
        for control, pair in batch["estimates"].items()
    }
    mixture_a, mixture_diag = gate(
        truth["mixture"], estimates["mixture"]["conservative"],
        estimates["mixture"]["aggressive"],
    )
    no_vocal_a, no_vocal_diag = gate(
        truth["accompaniment"], estimates["no_vocal"]["conservative"],
        estimates["no_vocal"]["aggressive"],
    )
    vocal_only_a, vocal_only_diag = gate(
        truth["vocal"], estimates["vocal_only"]["conservative"],
        estimates["vocal_only"]["aggressive"],
    )
    mixture_v = truth["mixture"] - mixture_a
    no_vocal_v = truth["accompaniment"] - no_vocal_a
    vocal_only_v = truth["vocal"] - vocal_only_a
    activity = batch["activity"].unsqueeze(0).to(device)
    loss, components = classical_residual_loss_v2(
        mixture_a, mixture_v, truth["mixture"], truth["accompaniment"],
        truth["vocal"], no_vocal_vocal_estimate=no_vocal_v,
        vocal_only_vocal_estimate=vocal_only_v, event_weights=activity,
        config=loss_config,
    )
    regularizers = []
    regularizer_parts = {}
    for control, diagnostics, pair in (
        ("mixture", mixture_diag, estimates["mixture"]),
        ("no_vocal", no_vocal_diag, estimates["no_vocal"]),
        ("vocal_only", vocal_only_diag, estimates["vocal_only"]),
    ):
        value, parts = gate_regularization(
            diagnostics, pair["aggressive"], pair["conservative"], gate.config
        )
        regularizers.append(value)
        regularizer_parts[control] = parts
    regularization = torch.stack(regularizers).mean()
    total = loss + regularization
    if not bool(torch.isfinite(total)):
        raise ValueError("gate training objective is non-finite")
    return total, components, regularization, regularizer_parts, {
        "accompaniment": mixture_a, "vocal": mixture_v,
        "diagnostics": mixture_diag,
    }


def _json_numbers(values: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        if value is None:
            result[key] = None
        elif hasattr(value, "detach") and getattr(value, "numel", lambda: 0)() == 1:
            result[key] = float(value.detach().cpu())
        else:
            result[key] = value
    return result


def _save_checkpoint(gate: SmoothResidualGate, directory: Path, *, step: int,
                     probe: dict[str, Any], device: str) -> dict[str, Any]:
    torch = _torch()
    if directory.exists():
        raise RuntimeError(f"refusing to rewrite gate checkpoint: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.parent / f".tmp-{directory.name}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    state_path = temporary / "state.pt"
    torch.save({"state_dict": gate.state_dict()}, state_path)
    loaded = torch.load(state_path, map_location=device, weights_only=True)["state_dict"]
    clone = SmoothResidualGate(gate.config).to(device)
    clone.load_state_dict(loaded)
    clone.eval(); gate.eval()
    with torch.no_grad():
        before, _ = gate(
            probe["mixture"], probe["conservative"], probe["aggressive"]
        )
        after, _ = clone(
            probe["mixture"], probe["conservative"], probe["aggressive"]
        )
    if not torch.equal(before, after):
        raise RuntimeError("gate export/reload output parity failed")
    if gate_bundle_identity(clone) != gate_bundle_identity(gate):
        raise RuntimeError("gate export/reload identity mismatch")
    record = {
        "schema": "audio-extract/classical-smooth-gate-checkpoint/v1",
        "step": step, "bundle": gate_bundle_identity(gate),
        "state_container_sha256": _sha_file(state_path),
        "reload_max_abs": float((before - after).abs().max().cpu()),
        "reload_exact": True,
    }
    (temporary / "checkpoint.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    (temporary / "COMPLETE").write_text("")
    temporary.rename(directory)
    gate.train()
    return record


def _input_pcm(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sha256": record["artifact_pcm_sha256"],
        "sample_rate_hz": record["sample_rate_hz"],
        "channel_layout": record["channels"], "frames": record["frames"],
        "sample_format": "float32-le-interleaved",
    }


def _write_audio_node(layout: TrackLayout, source: dict[str, Any], output: np.ndarray,
                      recipe: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    rid = identity.recipe_id(recipe)
    cdir = layout.candidate_dir(rid)
    if (cdir / "COMPLETE").is_file():
        path = cdir / "output.f32.wav"
        record = audio_record(path, recipe_id=rid)
        if (cdir / "output.pcm.sha256").read_text().strip() != record["artifact_pcm_sha256"]:
            raise RuntimeError(f"cached gate candidate PCM marker mismatch: {cdir}")
        return record | {"cached": True}
    artifact = identity.artifact_pcm_sha256(
        output, SR, ["FL", "FR"], len(output)
    )
    fd, name = tempfile.mkstemp(suffix=".f32.wav"); os.close(fd)
    temporary = Path(name)
    sf.write(temporary, output.astype("float32"), SR, subtype="FLOAT")
    try:
        layout.write_candidate(rid, recipe, execution, temporary, artifact)
    except ImmutableWriteError:
        pass
    path = cdir / "output.f32.wav"
    return audio_record(path, recipe_id=rid) | {"cached": False}


def _render_evaluation(gate: SmoothResidualGate, batch: dict[str, Any], *, step: int,
                       device: str, loss_config: ClassicalResidualLossConfig,
                       output_store: Path, code_commit: str,
                       checkpoint: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    gate.eval()
    with torch.no_grad():
        total, components, regularization, regularizer_parts, output = _forward_loss(
            gate, batch, device=device, loss_config=loss_config
        )
    accompaniment = output["accompaniment"][0].cpu().T.numpy().astype("float32")
    vocal = output["vocal"][0].cpu().T.numpy().astype("float32")
    conservative = batch["estimates"]["mixture"]["conservative"].T.numpy()
    if step == 0 and not np.array_equal(accompaniment, conservative):
        raise RuntimeError("step-0 decoded accompaniment differs from conservative parent")

    row = batch["row"]
    work = row["work_id"]
    layout = TrackLayout(output_store, work)
    source = _ensure_exact_source(layout, Path(row["truth"]["mixture"]["path"]))
    parents = [
        row["estimates"]["mixture"][name]["recipe_id"]
        for name in ("conservative", "aggressive")
    ]
    bundle = checkpoint["bundle"]
    recipe = {
        "schema": recipe_mod.SCHEMA, "canon": recipe_mod.CANON,
        "input_pcm": _input_pcm(row["truth"]["mixture"]),
        "operation": {"type": "phrase_route", "target": "instrumental"},
        "model": {
            "model_id": "classical-smooth-residual-gate-v1",
            "weights_sha256": bundle["weights_sha256"].removeprefix("sha256:"),
            "config_sha256": bundle["config_sha256"],
            "executed_bundle_id": bundle["bundle_sha256"],
            "adapter": "audio_extract.classical_gate.SmoothResidualGate",
            "adapter_revision": bundle["adapter_revision"], "members": parents,
        },
        "effective_config": {
            **gate.config.identity_dict(), "training_step": step,
            "parent_recipe_ids": parents, "frozen_separator_members": True,
            "alignment": "source-grid-exact",
            "loss_implementation": (
                "audio_extract.classical_loss_v2:classical_residual_loss_v2"
            ),
        },
        "software": {"audio_extract_commit": code_commit},
    }
    execution = {
        **identity.execution_fingerprint(), "parents": parents,
        "gate_bundle": bundle, "checkpoint": checkpoint,
        "step_zero_exact_parent": step == 0,
    }
    accompaniment_record = _write_audio_node(
        layout, source, accompaniment, recipe, execution
    )
    removed_recipe = {
        "schema": recipe_mod.SCHEMA, "canon": recipe_mod.CANON,
        "input_pcm": _input_pcm(row["truth"]["mixture"]),
        "operation": {"type": "mixture_minus_source", "target": "vocals",
                      "construction": "mixture_minus_source"},
        "model": {
            "model_id": "deterministic-complement-residual",
            "weights_sha256": hashlib.sha256(
                accompaniment_record["recipe_id"].encode()
            ).hexdigest(),
            "adapter": "audio_extract.train_classical_gate._render_evaluation",
            "adapter_revision": "exact-grid-complement/v1",
            "members": [accompaniment_record["recipe_id"]],
        },
        "effective_config": {
            "parent_recipe_ids": [accompaniment_record["recipe_id"]],
            "parent_artifact_pcm_sha256": accompaniment_record["artifact_pcm_sha256"],
            "alignment": "source-grid-exact",
        },
        "software": {"audio_extract_commit": code_commit},
    }
    removed_record = _write_audio_node(
        layout, source, vocal, removed_recipe,
        {**identity.execution_fingerprint(),
         "parents": [accompaniment_record["recipe_id"]]},
    )
    metrics = exact_metrics(
        accompaniment,
        batch["truth"]["accompaniment"].T.numpy().astype("float32"),
        batch["truth"]["vocal"].T.numpy().astype("float32"), SR,
    )
    gate.train()
    return {
        "work_id": work, "step": step,
        "objective": float(total.cpu()),
        "loss_components": _json_numbers(components),
        "regularization": float(regularization.cpu()),
        "control_regularization": {
            name: _json_numbers(parts) for name, parts in regularizer_parts.items()
        },
        "metrics": metrics, "accompaniment": accompaniment_record,
        "removed_vocal": removed_record,
        "step_zero_exact_parent": step == 0,
        "raw_gate_amplitude": float(
            output["diagnostics"]["raw_amplitude"].cpu()
        ),
        "effective_gate_amplitude": float(
            output["diagnostics"]["amplitude"].cpu()
        ),
    }


def _config_objects(config: dict[str, Any]) -> tuple[SmoothGateConfig, ClassicalResidualLossConfig]:
    model = dict(config["model"])
    if "band_edges_hz" in model:
        model["band_edges_hz"] = tuple(model["band_edges_hz"])
    gate_config = SmoothGateConfig(**model)
    gate_config.validate()
    loss = dict(config["loss"])
    if "stft_ffts" in loss:
        loss["stft_ffts"] = tuple(loss["stft_ffts"])
    loss_config = ClassicalResidualLossConfig(**loss)
    return gate_config, loss_config


def run_training(config_path: Path, output_root: Path, *, device: str,
                 seed: int) -> dict[str, Any]:
    torch = _torch()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"wrong gate config schema: {config.get('schema')}")
    manifest_path = Path(config["data"]["manifest"])
    manifest = load_manifest(manifest_path)
    if config["members"] != manifest["members"]:
        raise ValueError("gate config/manifest member mismatch")
    train_ids = list(config["data"]["train_work_ids"])
    eval_ids = list(config["data"]["eval_work_ids"])
    if set(train_ids) & set(eval_ids):
        raise ValueError("gate train/eval work overlap")
    group_by_work = {row["work_id"]: row["group_id"] for row in manifest["works"]}
    train_groups = {group_by_work[work] for work in train_ids}
    eval_groups = {group_by_work[work] for work in eval_ids}
    if train_groups & eval_groups:
        raise ValueError(f"source derivatives cross gate fold roles: {train_groups & eval_groups}")
    if config["data"].get("task") != "soloist_vs_rest":
        raise ValueError("gate task must be canonical soloist_vs_rest")
    if config["data"].get("task_roles") != {
        "removed": ["featured_soloists"],
        "retained": ["orchestra", "chorus", "non_target_soloists"],
    }:
        raise ValueError("gate task roles are not canonical")
    if config.get("loss_implementation") != (
        "audio_extract.classical_loss_v2:classical_residual_loss_v2"
    ):
        raise ValueError("gate must use the versioned residual-loss v2 contract")
    if config["optim"].get("name") != "adam":
        raise ValueError("bounded gate optimizer must be Adam")
    if int(config["optim"].get("batch", 0)) != 1:
        raise ValueError("the first bounded gate uses batch=1")
    if float(config["optim"].get("weight_decay", 0.0)) != 0.0:
        raise ValueError("the first bounded gate uses zero weight decay")
    steps = int(config["optim"]["steps"])
    if steps != 100 or config.get("evaluation_steps") != [0, 100]:
        raise ValueError("the bounded first gate requires identical steps 0 and 100")
    crop_frames = round(float(config["optim"]["crop_s"]) * SR)
    dataset = GateDataset(manifest, train_ids, crop_frames)
    eval_rows = {
        row["work_id"]: row for row in manifest["works"]
        if row["work_id"] in eval_ids
    }
    if set(eval_rows) != set(eval_ids):
        raise ValueError("not every gate eval work is in the manifest")

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    gate_config, loss_config = _config_objects(config)
    gate = SmoothResidualGate(gate_config).to(device)
    optimizer = torch.optim.Adam(
        list(gate.parameters()), lr=float(config["optim"]["lr"]),
        betas=tuple(float(value) for value in config["optim"].get("betas", [0.9, 0.999])),
        eps=float(config["optim"].get("eps", 1e-8)),
        weight_decay=float(config["optim"].get("weight_decay", 0.0)),
    )
    rng = np.random.default_rng(seed)
    probe_row = eval_rows[eval_ids[0]]
    probe_batch = GateDataset.read(probe_row, frames=min(8_192, int(
        probe_row["truth"]["mixture"]["frames"]
    )))
    probe = {
        "mixture": probe_batch["truth"]["mixture"].unsqueeze(0).to(device),
        "conservative": probe_batch["estimates"]["mixture"]["conservative"].unsqueeze(0).to(device),
        "aggressive": probe_batch["estimates"]["mixture"]["aggressive"].unsqueeze(0).to(device),
    }

    if output_root.exists():
        raise RuntimeError(f"refusing to reuse gate run directory: {output_root}")
    output_root.mkdir(parents=True)
    code_commit = _git_commit()
    history = []
    checkpoints = {}
    evaluations = {}

    for step in range(steps + 1):
        if step in (0, 100):
            checkpoint = _save_checkpoint(
                gate, output_root / "checkpoints" / f"step-{step:06d}",
                step=step, probe=probe, device=device,
            )
            checkpoints[str(step)] = checkpoint
            evaluations[str(step)] = []
            for work in eval_ids:
                batch = GateDataset.read(eval_rows[work])
                evaluations[str(step)].append(_render_evaluation(
                    gate, batch, step=step, device=device,
                    loss_config=loss_config,
                    output_store=output_root / "candidates" / f"step-{step:06d}",
                    code_commit=code_commit, checkpoint=checkpoint,
                ))
        if step == steps:
            break
        batch = dataset.sample(rng)
        optimizer.zero_grad(set_to_none=True)
        total, components, regularization, _, _ = _forward_loss(
            gate, batch, device=device, loss_config=loss_config
        )
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            list(gate.parameters()), float(config["optim"]["gradient_clip"])
        ).detach().cpu())
        if not np.isfinite(gradient_norm):
            raise ValueError("gate gradient norm is non-finite")
        optimizer.step()
        gate.project_parameters()
        history.append({
            "step": step + 1, "work_id": batch["work_id"],
            "start_frame": batch["start_frame"], "objective": float(total.detach().cpu()),
            "regularization": float(regularization.detach().cpu()),
            "gradient_norm_before_clip": gradient_norm,
            "raw_gate_amplitude": float(gate.correction_amplitude.detach().cpu()),
            "effective_gate_amplitude": float(gate.correction_amplitude.detach().cpu()),
            "components": _json_numbers(components),
        })

    report = {
        "schema": REPORT_SCHEMA, "status": "needs_human_ab",
        "experiment": config["experiment"], "fold": config["fold"],
        "code_commit": code_commit, "config_path": str(config_path),
        "config_sha256": _sha_file(config_path), "manifest_path": str(manifest_path),
        "manifest_sha256": _sha_file(manifest_path), "members": config["members"],
        "seed": seed, "device": device, "train_work_ids": train_ids,
        "eval_work_ids": eval_ids, "checkpoints": checkpoints,
        "evaluations": evaluations, "history": history,
        "terminal_claim": "bounded_engineering_gate_requires_independent_review_and_human_ab",
    }
    (output_root / "run-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "COMPLETE").write_text("")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("python -m audio_extract.train_classical_gate")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260809)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_training(
        args.config, args.output_root, device=args.device, seed=args.seed
    )
    print(json.dumps({
        "status": report["status"], "fold": report["fold"],
        "eval_works": len(report["eval_work_ids"]),
        "output": str(args.output_root),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
