"""Strict optimizer construction and provenance for separator training.

A resolved training config must describe the optimizer that is actually used.
Silently accepting ``name: adam`` while constructing ``AdamW`` changes the
regularized objective and makes checkpoints irreproducible. This module keeps
optimizer selection and all identity-bearing defaults explicit.
"""
from __future__ import annotations

from typing import Any, Iterable


class OptimizerConfigError(ValueError):
    """The resolved optimizer contract is incomplete or unsupported."""


def _pair(value: Any, *, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise OptimizerConfigError(f"{field} must be a two-element list")
    pair = (float(value[0]), float(value[1]))
    if not (0.0 <= pair[0] < 1.0 and 0.0 <= pair[1] < 1.0):
        raise OptimizerConfigError(f"{field} values must lie in [0,1)")
    return pair


def validate_optimizer_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized, fully explicit optimizer contract."""
    name = str(config.get("name", "")).lower()
    if name not in {"adam", "adamw"}:
        raise OptimizerConfigError(f"unsupported optimizer: {name!r}")
    missing = [field for field in ("betas", "eps", "weight_decay") if field not in config]
    if missing:
        raise OptimizerConfigError(f"optimizer contract missing fields: {missing}")
    betas = _pair(config["betas"], field="betas")
    eps = float(config["eps"])
    weight_decay = float(config["weight_decay"])
    if eps <= 0.0:
        raise OptimizerConfigError("eps must be positive")
    if weight_decay < 0.0:
        raise OptimizerConfigError("weight_decay must be non-negative")
    normalized = dict(config)
    normalized.update(name=name, betas=list(betas), eps=eps, weight_decay=weight_decay)
    return normalized


def build_optimizer(torch_module, parameter_groups: Iterable, config: dict[str, Any]):
    """Construct exactly the optimizer named by the resolved contract."""
    resolved = validate_optimizer_config(config)
    kwargs = {
        "betas": tuple(resolved["betas"]),
        "eps": resolved["eps"],
        "weight_decay": resolved["weight_decay"],
    }
    if resolved["name"] == "adam":
        return torch_module.optim.Adam(parameter_groups, **kwargs)
    return torch_module.optim.AdamW(parameter_groups, **kwargs)


def optimizer_provenance(optimizer) -> dict[str, Any]:
    """Serialize effective identity-bearing facts from a live optimizer."""
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        groups.append({
            "index": index,
            "group_name": group.get("group_name", f"group_{index}"),
            "lr": float(group["lr"]),
            "betas": [float(x) for x in group["betas"]],
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "amsgrad": bool(group.get("amsgrad", False)),
            "maximize": bool(group.get("maximize", False)),
            "capturable": bool(group.get("capturable", False)),
            "differentiable": bool(group.get("differentiable", False)),
            "fused": group.get("fused"),
        })
    return {
        "class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__qualname__}",
        "groups": groups,
    }
