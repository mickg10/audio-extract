"""Shared identity for the binding opera-routing work and basis scope."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

SCHEMA = "audio-extract/oracle-routing-work-contract/v3"

REQUIRED_WORKS = (
    "bologna_verdi",
    "bologna_donizetti",
    "bologna_puccini",
    "aalto_mozart_dry",
    "aalto_mozart_hall",
)

REQUIRED_BASIS_ALIASES = (
    "median_mdx_mel_bs",
    "geomedian_mdx_mel_bs",
    "convex_fusion_uniform",
    "residual_mdx23c",
    "residual_melband",
    "residual_bs_roformer",
    "htdemucs_04573f0d",
    "htdemucs_955717e8",
)

NO_VOCAL_CONTROL_WORK_ID = "aalto_mozart_dry"
HALL_REFERENCE_WORK_ID = "aalto_mozart_hall"


def identity_dict() -> dict[str, Any]:
    """Return the one closed work/basis/source-role identity."""

    return {
        "schema": SCHEMA,
        "required_voiced_works": list(REQUIRED_WORKS),
        "required_basis_aliases": list(REQUIRED_BASIS_ALIASES),
        "source_roles": {
            "voiced_mixture": list(REQUIRED_WORKS),
            "no_vocal_accompaniment": [NO_VOCAL_CONTROL_WORK_ID],
            "hall_reference": [HALL_REFERENCE_WORK_ID],
        },
    }


def semantic_sha256(value: Mapping[str, Any] | None = None) -> str:
    document = dict(value) if value is not None else identity_dict()
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


WORK_CONTRACT_SHA256 = semantic_sha256()


def validate_identity(value: Mapping[str, Any]) -> None:
    """Refuse any alternate work order, basis order, or source-role scope."""

    if dict(value) != identity_dict():
        raise ValueError("routing work contract differs from the compiled identity")
    if semantic_sha256(value) != WORK_CONTRACT_SHA256:
        raise ValueError("routing work contract semantic digest differs")
