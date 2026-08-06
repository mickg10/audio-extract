"""Planner contracts + deterministic experiment validation (v2.1 WP10 / §15).

Two hard rules from the plan:

* **The planner never finalizes.** Its only statuses are ``experiments`` and
  ``no_more_useful_probes`` (§15.4); ``final`` / ``no_acceptable_candidate`` belong
  to the deterministic selector alone.
* **The model cannot self-certify** ``changes_one_variable`` (§15.5). The
  controller constructs the child recipe, normalizes both, diffs them, and counts
  the changed *material* fields itself.
"""

from __future__ import annotations

from typing import Any

from . import recipe as recipe_mod
from .probes import recommend_probe

# §15.3 production planner vocabulary. request_human_comparison and directly-
# finalizing stop_with_reason are REMOVED.
PLANNER_ACTIONS = frozenset({
    "inspect_run", "inspect_uncertainty",
    "propose_track_remix", "propose_causal_probe",
    "propose_model_variant", "propose_construction", "propose_ensemble",
    "run_registered_experiment", "rescore",
})

PLANNER_STATUSES = ("experiments", "no_more_useful_probes")

# Fields that count as material, experiment-changing variables in a recipe diff.
# software.* (code commit, lib versions) is provenance, not an experiment variable.
_MATERIAL_PREFIXES = ("model.", "effective_config.", "operation.")


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, (list, tuple)):
        out[prefix[:-1]] = list(obj)
    else:
        out[prefix[:-1]] = obj
    return out


def recipe_diff(parent: dict, child: dict) -> set[str]:
    """Dotted paths that differ between two NORMALIZED recipes. Normalization
    materializes defaults first, so an omitted default vs an explicit default is
    not a difference."""
    fa = _flatten(recipe_mod.normalize_recipe(parent))
    fb = _flatten(recipe_mod.normalize_recipe(child))
    changed = {k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}
    return changed


def material_changes(parent: dict, child: dict) -> set[str]:
    return {p for p in recipe_diff(parent, child)
            if p.startswith(_MATERIAL_PREFIXES)}


def validate_experiment(parent: dict, child: dict, *, allow_combined: bool = False
                        ) -> tuple[bool, str, set[str]]:
    """§15.5: the controller counts changed material fields itself. One material
    variable per experiment unless explicitly combined."""
    try:
        changed = material_changes(parent, child)
    except recipe_mod.RecipeError as exc:
        return False, f"child recipe invalid: {exc}", set()
    if not changed:
        return False, "experiment changes nothing material (duplicate of parent)", changed
    if len(changed) > 1 and not allow_combined:
        return False, f"experiment changes {len(changed)} material variables: {sorted(changed)}", changed
    return True, "", changed


def validate_planner_response(d: dict) -> tuple[bool, str]:
    status = d.get("status")
    if status == "experiments":
        actions = d.get("actions")
        if not isinstance(actions, list):
            return False, "experiments requires an actions list"
        for a in actions:
            if a.get("type") not in PLANNER_ACTIONS:
                return False, f"unknown planner action: {a.get('type')!r}"
            if a.get("type").startswith(("propose_", "run_")) and not a.get("parent_recipe_id"):
                return False, f"action {a.get('type')!r} missing parent_recipe_id (§15.5)"
        return True, ""
    if status == "no_more_useful_probes":
        if not d.get("reason"):
            return False, "no_more_useful_probes requires a reason"
        return True, ""
    if status in ("final", "no_acceptable_candidate", "needs_human_ab"):
        return False, f"planner may not emit {status!r}: terminal authority belongs to the selector"
    return False, f"unknown planner status: {status!r}"


class DeterministicFallbackPlanner:
    """§14.5 glue: no DeepSeek/Pi available → recommend the highest-value probe for
    the two closest candidates; declare no_more_useful_probes when nothing helps."""

    def __init__(self, executed: set[str] | None = None):
        self.executed: set[str] = executed or set()

    def propose(self, report: dict) -> dict:
        pair = report.get("top_pair")
        if not pair or len(pair) < 2:
            return {"status": "no_more_useful_probes",
                    "reason": "fewer than two comparable candidates"}
        rec = recommend_probe(pair[0], pair[1], executed=self.executed)
        if rec.value <= 0 and "fallback" in rec.reason and rec.probe in self.executed:
            return {"status": "no_more_useful_probes",
                    "reason": "probe vocabulary exhausted for this pair"}
        self.executed.add(rec.probe)
        return {"status": "experiments",
                "actions": [{"type": "run_registered_experiment",
                             "probe": rec.probe, "axis": rec.axis,
                             "parent_recipe_id": report.get("best_recipe_id", "sha256:baseline"),
                             "reason": rec.reason}]}
