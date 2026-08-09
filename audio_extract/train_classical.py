"""HTDemucs experiments for classical/operatic vocal removal.

The first production pilot preserves the released four-source topology.  Only
the vocal output is task-facing: ``V_hat = output[vocals]`` and the delivered
accompaniment is the mixture-consistent residual ``A_hat = M - V_hat``.

The explicitly labelled random two-source path exists only for the preregistered
Fold-V A2 control.  It uses the model's direct accompaniment source and cannot
be mistaken for, resumed into, or promoted as a pretrained continuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .classical_loss import ClassicalLossConfig, classical_separation_loss

SR = 44_100
DEFAULT_EVAL_WORKS = ("bologna_verdi", "bologna_puccini", "bologna_donizetti", "aalto_mozart_dry")


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on the GPU host
        raise RuntimeError("install the 'train' extra to run classical training") from exc
    return torch


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _state_dict_sha256(model) -> str:
    """Canonical identity for an initialized model state, independent of filenames."""
    h = hashlib.sha256()
    h.update(b"audio-extract/model-state/v1\0")
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload = value.view(_torch().uint8).numpy().tobytes()
        h.update(len(metadata).to_bytes(8, "big")); h.update(metadata)
        h.update(len(payload).to_bytes(8, "big")); h.update(payload)
    return "sha256:" + h.hexdigest()


def _read_exact(path: Path, *, start: int = 0, frames: int | None = None):
    torch = _torch()
    info = sf.info(path)
    if info.samplerate != SR or info.channels != 2 or info.subtype != "FLOAT":
        raise ValueError(
            f"training audio must be 44.1 kHz stereo float32 WAV: {path} ({info})"
        )
    stop = None if frames is None else start + frames
    audio, sr = sf.read(path, dtype="float32", always_2d=True, start=start, stop=stop)
    if sr != SR or (frames is not None and len(audio) != frames):
        raise ValueError(f"short or off-grid read: {path}, wanted {frames}, got {len(audio)}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"non-finite audio: {path}")
    return torch.from_numpy(audio.T.copy())


class ClassicalDataset:
    """Random exact-grid crops from immutable materializations."""

    def __init__(self, root: Path, works: list[str], crop_frames: int):
        self.root = root
        self.crop_frames = crop_frames
        self.items = []
        for work in works:
            directory = root / work
            if not directory.is_dir():
                continue
            reports = json.loads((directory / "report.json").read_text())
            infos = {role: sf.info(directory / f"{role}.f32.wav") for role in "MAV"}
            grids = {(i.frames, i.samplerate, i.channels, i.subtype) for i in infos.values()}
            if len(grids) != 1:
                raise ValueError(f"materialized grid mismatch for {work}: {infos}")
            frames, sr, channels, subtype = next(iter(grids))
            if sr != SR or channels != 2 or subtype != "FLOAT":
                raise ValueError(f"invalid materialized grid for {work}: {next(iter(grids))}")
            if frames < crop_frames:
                raise ValueError(f"work {work} is shorter than one training crop")
            self.items.append((work, directory, frames, reports["recipe_id"]))
        if not self.items:
            raise ValueError("no eligible immutable training works were materialized")

    def sample(self, rng: np.random.Generator):
        work, directory, total_frames, recipe_id = self.items[int(rng.integers(len(self.items)))]
        start = int(rng.integers(0, total_frames - self.crop_frames + 1))
        audio = {role: _read_exact(directory / f"{role}.f32.wav", start=start,
                                   frames=self.crop_frames) for role in "MAV"}
        mask = np.load(directory / "vocal_activity.npy", mmap_mode="r")
        event = np.asarray(mask[start:start + self.crop_frames], dtype="float32")
        if len(event) != self.crop_frames:
            raise ValueError(f"short activity mask for {work}")
        return audio["M"], audio["A"], audio["V"], _torch().from_numpy(event.copy()), {
            "work_id": work, "recipe_id": recipe_id, "start_frame": start
        }


def _manifest_train_works(manifest_path: Path, splits_path: Path, train_splits: set[str]) -> list[str]:
    split_doc = json.loads(splits_path.read_text())
    works = []
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        declared = split_doc["group_split"].get(row["group_id"])
        if declared != row["split"]:
            raise ValueError(f"manifest/split disagreement for {row['work_id']}")
        if declared in train_splits:
            works.append(row["work_id"])
    return sorted(set(works))


def _exact_fold_works(manifest_path: Path, splits_path: Path, train_ids: list[str],
                      eval_ids: list[str]) -> tuple[list[str], list[str]]:
    rows = {row["work_id"]: row for row in
            (json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip())}
    split_doc = json.loads(splits_path.read_text())
    requested = list(dict.fromkeys([*train_ids, *eval_ids]))
    missing = sorted(set(requested) - set(rows))
    if missing:
        raise ValueError(f"fold works missing from manifest: {missing}")
    train_groups, eval_groups = set(), set()
    for work in requested:
        row = rows[work]
        if split_doc["group_split"].get(row["group_id"]) != row["split"]:
            raise ValueError(f"manifest/split disagreement for {work}")
        if row.get("integrity_class") != "linear_exact":
            raise ValueError(f"exact fold refuses {work} integrity={row.get('integrity_class')}")
        if not {"M", "A", "V"}.issubset(row.get("files", {})):
            raise ValueError(f"exact fold requires explicit M/A/V for {work}")
        if "accompaniment_A" not in row.get("eligible_training_targets", []):
            raise ValueError(f"work is not eligible for accompaniment training: {work}")
        (train_groups if work in train_ids else eval_groups).add(row["group_id"])
    overlap = train_groups & eval_groups
    if overlap:
        raise ValueError(f"source derivatives cross fold roles: {sorted(overlap)}")
    return list(train_ids), list(eval_ids)


def _load_model(cfg: dict, device: str, seed: int):
    torch = _torch()

    checkpoint = cfg["base_checkpoint"]
    if checkpoint.get("mode") == "random_two_source_control":
        from demucs.htdemucs import HTDemucs

        expected_seed = int(checkpoint["initialization_seed"])
        if seed != expected_seed:
            raise ValueError(
                f"random control seed mismatch: CLI seed {seed} != pinned {expected_seed}"
            )
        if list(cfg["sources"]) != ["accompaniment", "vocals"]:
            raise ValueError("random A2 control requires [accompaniment, vocals] topology")
        model_cfg = cfg["model"]
        model = HTDemucs(
            sources=list(cfg["sources"]),
            audio_channels=int(cfg["audio_channels"]),
            samplerate=int(cfg["samplerate"]),
            channels=int(model_cfg["channels"]),
            depth=int(model_cfg["depth"]),
        ).to(device)
        actual = _state_dict_sha256(model)
        pinned = checkpoint.get("initial_state_sha256")
        if pinned and actual != "sha256:" + pinned.removeprefix("sha256:"):
            raise ValueError(f"random initial-state hash mismatch: {actual} != {pinned}")
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("random control model contains non-finite parameters")
        return model, {
            "mode": "random_two_source_control",
            "initialization_seed": seed,
            "initial_state_sha256": actual,
            "architecture": {
                "name": "HTDemucs",
                "sources": list(cfg["sources"]),
                "audio_channels": int(cfg["audio_channels"]),
                "samplerate": int(cfg["samplerate"]),
                "channels": int(model_cfg["channels"]),
                "depth": int(model_cfg["depth"]),
            },
        }

    from demucs.pretrained import get_model

    signature = checkpoint["signature"]
    model = get_model(signature)
    if hasattr(model, "models"):
        raise ValueError(f"{signature} resolved to a model bag; an individual model is required")
    if list(model.sources) != list(cfg["sources"]):
        raise ValueError(f"pretrained source topology mismatch: {model.sources} != {cfg['sources']}")
    candidates = list((Path.home() / ".cache/torch/hub/checkpoints").glob(f"{signature}-*.th"))
    if len(candidates) != 1:
        raise ValueError(f"cannot uniquely identify cached checkpoint for {signature}: {candidates}")
    actual = _sha_file(candidates[0])
    expected = "sha256:" + checkpoint["sha256"].removeprefix("sha256:")
    if actual != expected:
        raise ValueError(f"pretrained checkpoint hash mismatch: {actual} != {expected}")
    model.to(device)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("pretrained model contains non-finite parameters")
    return model, {"signature": signature, "path": str(candidates[0]), "sha256": actual}


def _build_optimizer(model, cfg: dict):
    torch = _torch()
    if cfg["base_checkpoint"].get("mode") == "random_two_source_control":
        if cfg["optim"]["name"] != "adam":
            raise ValueError("random A2 control optimizer must be Adam")
        return torch.optim.Adam(model.parameters(), lr=float(cfg["optim"]["lr"]))
    return torch.optim.AdamW(_parameter_groups(model, cfg["optim"]["schedule"]))


def _parameter_groups(model, schedule: list[dict]):
    """Freeze lower encoder initially and assign the preregistered low learning rates."""
    first = schedule[0]
    groups = {"lower_encoder": [], "upper_decoder": [], "transformer_decoder": []}
    for name, parameter in model.named_parameters():
        lower = False
        if name.startswith("encoder."):
            fields = name.split(".")
            lower = len(fields) > 1 and fields[1].isdigit() and int(fields[1]) < 3
        if lower:
            parameter.requires_grad_(False)
            groups["lower_encoder"].append(parameter)
        elif name.startswith("decoder."):
            groups["upper_decoder"].append(parameter)
        else:
            groups["transformer_decoder"].append(parameter)
    return [
        {"params": groups["lower_encoder"], "lr": float(first["lower_encoder_lr"]),
         "group_name": "lower_encoder"},
        {"params": groups["upper_decoder"], "lr": float(first["upper_decoder_lr"]),
         "group_name": "upper_decoder"},
        {"params": groups["transformer_decoder"], "lr": float(first["transformer_decoder_lr"]),
         "group_name": "transformer_decoder"},
    ]


def _set_learning_rates(model, optimizer, cfg: dict, step: int) -> None:
    schedule = cfg["optim"].get("schedule")
    if not schedule:
        return
    second_start, end = schedule[1]["steps"]
    if step < second_start:
        return
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    initial = float(schedule[1]["full_network_lr"])
    progress = min(1.0, max(0.0, (step - second_start) / max(1, end - second_start)))
    lr = initial * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr


def _loss_config(cfg: dict) -> ClassicalLossConfig:
    weights = cfg["loss"]
    return ClassicalLossConfig(
        waveform_l1=float(weights["waveform_l1"]),
        complex_stft=float(weights["complex_stft"]),
        mixture_consistency=float(weights["mixture_consistency"]),
        no_vocal_false_positive=float(weights["no_vocal_false_positive"]),
        vocal_only_false_negative=float(weights["vocal_only_false_negative"]),
        source_coordinate=float(weights["source_coordinate_alpha_beta_R"]),
        stereo_coherence=float(weights["stereo_coherence"]),
        event_weighted=float(weights["exact_event_weighting"]),
    )


def _separate_controls(model, mixture, accompaniment, vocals, vocal_index: int,
                       construction: str, accompaniment_index: int | None):
    torch = _torch()
    outputs = model(torch.cat((mixture, accompaniment, vocals), dim=0))
    batch = mixture.shape[0]
    vocal_estimates = outputs[:, vocal_index]
    mix_v = vocal_estimates[:batch]
    a_v = vocal_estimates[batch:2 * batch]
    v_v = vocal_estimates[2 * batch:]
    if construction == "mixture_residual":
        mix_a, a_a, v_a = mixture - mix_v, accompaniment - a_v, vocals - v_v
    elif construction == "direct_source":
        if accompaniment_index is None:
            raise ValueError("direct_source requires accompaniment_source_index")
        accompaniment_estimates = outputs[:, accompaniment_index]
        mix_a = accompaniment_estimates[:batch]
        a_a = accompaniment_estimates[batch:2 * batch]
        v_a = accompaniment_estimates[2 * batch:]
    else:
        raise ValueError(f"unknown accompaniment construction: {construction}")
    return {
        "A_hat": mix_a,
        "V_hat": mix_v,
        "A_only_A_hat": a_a,
        "A_only_V_hat": a_v,
        "V_only_A_hat": v_a,
        "V_only_V_hat": v_v,
    }


def _export_and_parity(model, run_dir: Path, cfg: dict, *, label: str,
                       reuse_existing: bool = False) -> dict:
    torch = _torch()
    from demucs import states
    from demucs.apply import apply_model
    from omegaconf import OmegaConf

    destination = run_dir / f"model-{label}.th"
    if destination.exists() and not reuse_existing:
        raise RuntimeError(f"refusing to rewrite immutable export: {destination}")
    if not destination.exists():
        package = states.serialize_model(model, OmegaConf.create({"audio_extract": cfg}), half=False)
        torch.save(package, destination)
    reloaded = states.load_model(destination, strict=True).to(next(model.parameters()).device).eval()
    generator = torch.Generator(device="cpu").manual_seed(8128)
    probe = torch.randn(1, 2, SR, generator=generator).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        original = apply_model(model, probe, shifts=0, split=False)
        restored = apply_model(reloaded, probe, shifts=0, split=False)
    difference = float((original - restored).abs().max().cpu())
    if difference != 0.0 or not torch.equal(original, restored):
        raise RuntimeError(f"export/reload parity failed: max_abs={difference}")
    return {"path": str(destination), "sha256": _sha_file(destination),
            "zero_step_or_checkpoint_parity": True, "max_abs_difference": difference,
            "shifts": 0, "split": False}


def _source_metrics(candidate, accompaniment, vocals) -> dict:
    from .judge_labels import local_source_coordinate_labels
    from .judge_train import label_targets

    y = candidate.T.detach().cpu().numpy()
    a = accompaniment.T.detach().cpu().numpy()
    v = vocals.T.detach().cpu().numpy()
    if y.shape != a.shape or y.shape != v.shape:
        raise ValueError(f"exact evaluation grid mismatch: {y.shape}, {a.shape}, {v.shape}")
    labels = local_source_coordinate_labels(
        y, a, v, tile_frames=round(0.5 * SR), hop_frames=round(0.25 * SR)
    )
    metrics = label_targets(labels)
    from .metrics_v2 import stereo_v2
    stereo = stereo_v2(y, a)
    for observation in stereo:
        if observation["available"]:
            metrics[observation["metric"]] = observation["value"]
    return metrics


def _evaluate(model, truth_root: Path, run_dir: Path, step: int, vocal_index: int,
              device: str, eval_works: list[str], construction: str,
              accompaniment_index: int | None) -> dict:
    torch = _torch()
    from demucs.apply import apply_model

    step_dir = run_dir / "evaluation" / f"step-{step:06d}"
    if step_dir.exists():
        raise RuntimeError(f"refusing to rewrite immutable evaluation: {step_dir}")
    step_dir.mkdir(parents=True)
    result = {"step": step, "works": {}}
    model.eval()
    for work in eval_works:
        directory = truth_root / work
        mixture = _read_exact(directory / "mix_with_voice.wav")
        accompaniment = _read_exact(directory / "orchestra_only.wav")
        vocals = _read_exact(directory / "voice_ref.wav")
        if mixture.shape != accompaniment.shape or mixture.shape != vocals.shape:
            raise ValueError(f"truth grid mismatch for {work}")
        with torch.no_grad():
            output = apply_model(model, mixture.unsqueeze(0), device=device, shifts=0,
                                 split=True, overlap=0.25)[0].cpu()
            no_vocal_output = apply_model(model, accompaniment.unsqueeze(0), device=device,
                                          shifts=0, split=True, overlap=0.25)[0].cpu()
        if output.shape[-1] != mixture.shape[-1]:
            raise ValueError(f"inference frame mismatch for {work}: {output.shape}, {mixture.shape}")
        vocal_hat = output[vocal_index]
        if construction == "mixture_residual":
            accompaniment_hat = mixture - vocal_hat
        elif construction == "direct_source" and accompaniment_index is not None:
            accompaniment_hat = output[accompaniment_index]
        else:
            raise ValueError(f"invalid accompaniment construction: {construction}")
        no_vocal_hat = no_vocal_output[vocal_index]
        false_positive_ratio = float(
            no_vocal_hat.square().sum() / accompaniment.square().sum().clamp_min(1e-12)
        )
        work_path = step_dir / work
        work_path.mkdir()
        sf.write(work_path / "accompaniment.f32.wav", accompaniment_hat.T.numpy(), SR,
                 subtype="FLOAT")
        sf.write(work_path / "removed-vocal.f32.wav", vocal_hat.T.numpy(), SR, subtype="FLOAT")
        result["works"][work] = {
            "frames": int(mixture.shape[-1]),
            "metrics": _source_metrics(accompaniment_hat, accompaniment, vocals),
            "no_vocal_false_positive_energy_ratio": false_positive_ratio,
            "accompaniment_pcm_sha256": hashlib.sha256(
                np.ascontiguousarray(accompaniment_hat.T.numpy(), dtype="float32").tobytes()
            ).hexdigest(),
        }
    (step_dir / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _existing_evaluation(run_dir: Path, step: int, eval_works: list[str]) -> dict | None:
    step_dir = run_dir / "evaluation" / f"step-{step:06d}"
    if not step_dir.exists():
        return None
    report_path = step_dir / "report.json"
    if not report_path.exists():
        raise RuntimeError(f"incomplete immutable evaluation requires inspection: {step_dir}")
    report = json.loads(report_path.read_text())
    if report.get("step") != step or set(report.get("works", {})) != set(eval_works):
        raise RuntimeError(f"invalid existing evaluation report: {report_path}")
    for work, facts in report["works"].items():
        for filename in ("accompaniment.f32.wav", "removed-vocal.f32.wav"):
            info = sf.info(step_dir / work / filename)
            if (info.frames, info.samplerate, info.channels, info.subtype) != (
                    facts["frames"], SR, 2, "FLOAT"):
                raise RuntimeError(f"existing evaluation grid verification failed: {work}/{filename}")
    return report


def run_training(args: argparse.Namespace) -> dict:
    torch = _torch()
    manifest = Path(args.manifest).resolve()
    splits = Path(args.split_manifest).resolve()
    config_path = Path(args.config).resolve()
    run_dir = Path(args.run_dir).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    requested_steps = int(cfg["optim"]["steps_first_run"] if args.steps is None else args.steps)
    if requested_steps < 0:
        raise ValueError("steps must be non-negative")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = run_dir / "resolved-config.yaml"
    if resolved_path.exists() and yaml.safe_load(resolved_path.read_text()) != cfg:
        raise RuntimeError("run directory belongs to a different resolved config")
    if not resolved_path.exists():
        resolved_path.write_text(yaml.safe_dump(cfg, sort_keys=True))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    if cfg["data"].get("train_work_ids"):
        train_works, eval_works = _exact_fold_works(
            manifest, splits, list(cfg["data"]["train_work_ids"]),
            list(cfg["data"]["eval_work_ids"])
        )
    else:
        train_works = _manifest_train_works(manifest, splits, set(cfg["data"]["train_splits"]))
        eval_works = list(cfg["data"].get("eval_work_ids", DEFAULT_EVAL_WORKS))
    model, base = _load_model(cfg, device, int(args.seed))
    requested_crop = round(float(cfg["optim"]["crop_s"]) * SR)
    crop_frames = int(model.valid_length(requested_crop))
    dataset = ClassicalDataset(Path(cfg["data"]["materialized_root"]), train_works, crop_frames)
    optimizer = _build_optimizer(model, cfg)
    loss_cfg = _loss_config(cfg)
    vocal_index = int(cfg["vocal_source_index"])
    construction = cfg.get("accompaniment_construction", "mixture_residual")
    accompaniment_index = cfg.get("accompaniment_source_index")
    accompaniment_index = None if accompaniment_index is None else int(accompaniment_index)
    eval_steps = {int(s) for s in cfg["optim"]["evaluation_steps"] if int(s) <= requested_steps}

    zero_export = _export_and_parity(
        model, run_dir, cfg, label="step-000000", reuse_existing=bool(args.resume)
    )
    evaluations = []
    if 0 in eval_steps:
        existing = _existing_evaluation(run_dir, 0, eval_works)
        evaluations.append(existing or _evaluate(
            model, Path(args.truth_root), run_dir, 0, vocal_index, device, eval_works,
            construction, accompaniment_index
        ))

    history = []
    sampled = []
    start_step = 0
    resumed_from = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("config") != cfg or state.get("base_checkpoint") != base:
            raise RuntimeError("resume checkpoint config or base identity mismatch")
        start_step = int(state["step"])
        if not 0 < start_step <= requested_steps:
            raise RuntimeError(f"resume step {start_step} is outside requested run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["numpy_rng_state"]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        history = list(state.get("history", []))
        sampled = list(state.get("sampled", []))
        resumed_from = {"path": str(resume_path), "sha256": _sha_file(resume_path),
                        "step": start_step}
        for eval_step in sorted(s for s in eval_steps if 0 < s <= start_step):
            existing = _existing_evaluation(run_dir, eval_step, eval_works)
            if existing is None:
                raise RuntimeError(f"checkpoint step {start_step} lacks evaluation step {eval_step}")
            evaluations.append(existing)
    started = time.time()
    for step in range(start_step, requested_steps):
        _set_learning_rates(model, optimizer, cfg, step)
        samples = [dataset.sample(rng) for _ in range(int(cfg["optim"]["batch"]))]
        mixture = torch.stack([s[0] for s in samples]).to(device)
        accompaniment = torch.stack([s[1] for s in samples]).to(device)
        vocals = torch.stack([s[2] for s in samples]).to(device)
        activity = torch.stack([s[3] for s in samples]).to(device).unsqueeze(1)
        event_weights = 1.0 + 2.0 * activity
        sampled.extend(s[4] for s in samples)

        model.train()
        estimates = _separate_controls(
            model, mixture, accompaniment, vocals, vocal_index,
            construction, accompaniment_index
        )
        loss, components = classical_separation_loss(
            estimates["A_hat"], estimates["V_hat"], mixture, accompaniment, vocals,
            no_vocal_accompaniment_estimate=estimates["A_only_A_hat"],
            no_vocal_vocal_estimate=estimates["A_only_V_hat"],
            vocal_only_accompaniment_estimate=estimates["V_only_A_hat"],
            vocal_only_vocal_estimate=estimates["V_only_V_hat"],
            event_weights=event_weights,
            config=loss_cfg,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg["optim"]["gradient_clip"])
        )
        if not torch.isfinite(gradient):
            raise RuntimeError(f"non-finite gradient at step {step + 1}")
        optimizer.step()
        values = {name: (None if value is None else float(value.detach().cpu()))
                  for name, value in components.items()}
        values.update({"step": step + 1, "total": float(loss.detach().cpu()),
                       "gradient_norm": float(gradient.detach().cpu()),
                       "learning_rates": {g["group_name"]: g["lr"] for g in optimizer.param_groups}})
        history.append(values)
        if (step + 1) % 20 == 0 or step == 0:
            print(json.dumps(values), file=sys.stderr, flush=True)

        completed = step + 1
        if completed in eval_steps:
            checkpoint = run_dir / f"checkpoint-step-{completed:06d}.pt"
            if checkpoint.exists():
                raise RuntimeError(f"refusing to rewrite immutable checkpoint: {checkpoint}")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": completed, "config": cfg, "base_checkpoint": base,
                        "numpy_rng_state": rng.bit_generator.state,
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": (torch.cuda.get_rng_state_all()
                                           if device.startswith("cuda") else None),
                        "history": history, "sampled": sampled}, checkpoint)
            evaluations.append(_evaluate(model, Path(args.truth_root), run_dir, completed,
                                         vocal_index, device, eval_works, construction,
                                         accompaniment_index))

    final_export = zero_export if requested_steps == 0 else _export_and_parity(
        model, run_dir, cfg, label=f"step-{requested_steps:06d}",
        reuse_existing=bool(args.resume and start_step == requested_steps)
    )
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()
    report = {
        "schema": "audio-extract/classical-training-run/v1",
        "status": "parity_complete" if requested_steps == 0 else "pilot_complete",
        "source_commit": commit,
        "base_checkpoint": base,
        "steps": requested_steps,
        "device": device,
        "train_works": [item[0] for item in dataset.items],
        "eval_works": eval_works,
        "accompaniment_construction": construction,
        "materialization_recipes": {item[0]: item[3] for item in dataset.items},
        "manifest_sha256": _sha_file(manifest),
        "split_manifest_sha256": _sha_file(splits),
        "config_sha256": _sha_file(config_path),
        "zero_step_export": zero_export,
        "final_export": final_export,
        "evaluation_steps": [e["step"] for e in evaluations],
        "all_losses_finite": all(math.isfinite(h["total"]) for h in history),
        "first_loss": None if not history else history[0]["total"],
        "last_loss": None if not history else history[-1]["total"],
        "elapsed_s": round(time.time() - started, 3),
        "sample_count": len(sampled),
        "resumed_from": resumed_from,
        "repro_command": " ".join(sys.argv),
    }
    (run_dir / "training-log.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history)
    )
    (run_dir / "sample-log.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sampled)
    )
    (run_dir / "run-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("audio-extract train classical")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--truth-root", default="/home/mickg/truth_pairs")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--resume", help="immutable checkpoint-step-XXXXXX.pt to resume")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run_training(build_parser().parse_args(argv))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
