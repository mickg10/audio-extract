"""Bounded conductor (docs/v2 §6; oracle harness update: plain Pi + DeepSeek V4 Pro).

The deterministic job queue is the real harness; this conductor is the bounded
decision layer over typed audio tools. An LLM planner (DeepSeek V4 Pro via Pi)
never touches waveforms — it proposes from a typed action vocabulary, and the
controller validates every proposal (action exists, budget remains, not already
cached, one material variable per experiment) before executing. Two refinement
rounds, then a terminal decision: ``final`` or ``needs_human_ab``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

# Typed action vocabulary (docs/v2 §6.1). The planner may propose only these.
# Autonomous production action set (oracle follow-up): no human comparison. Ambiguity
# is resolved by a registered discriminating probe, not a person. The probe actions
# (run_track_remix_challenge / run_*_intervention_probe / expand_judge_committee) land
# with the autonomous-selection subsystem (challenges.py).
TYPED_ACTIONS = frozenset({
    "run_model_variant", "run_construction", "change_overlap", "build_weighted_ensemble",
    "render_full_track", "stop_with_reason",
})

_EXCERPT_ACTIONS = {"run_model_variant", "run_construction", "change_overlap"}


@dataclass
class Budget:
    # oracle-aligned limits (issue #1 design doc §15): full hard-budget set
    max_excerpt_candidates: int = 24
    max_full_renders: int = 3
    max_ensembles: int = 3
    max_rounds: int = 2
    max_new_candidates_per_round: int = 6
    max_cleanup_stages_per_candidate: int = 1


@dataclass
class ConductorState:
    round: int = 0
    excerpt_candidates: int = 0
    full_renders: int = 0
    ensembles: int = 0
    rendered: set = field(default_factory=set)   # dedup keys already executed

    def remaining(self, b: Budget) -> dict:
        return {
            "excerpt_candidates": b.max_excerpt_candidates - self.excerpt_candidates,
            "full_renders": b.max_full_renders - self.full_renders,
            "ensembles": b.max_ensembles - self.ensembles,
            "rounds": b.max_rounds - self.round,
        }


class Planner(Protocol):
    def propose(self, report: dict) -> dict:
        """Return either {"actions": [...]} or a terminal decision
        {"status": "final"|"needs_human_ab", ...}."""
        ...


class ScriptedPlanner:
    """Deterministic planner that replays a fixed list of proposals (one per
    round). Used for tests and as the reference planner shape."""

    def __init__(self, proposals: list[dict]):
        self._proposals = list(proposals)
        self.calls = 0

    def propose(self, report: dict) -> dict:
        self.calls += 1
        if self._proposals:
            return self._proposals.pop(0)
        return {"actions": []}


# System instruction for the real planner (docs/v2 §6.3): it does not hear audio.
DEEPSEEK_SYSTEM = (
    "You are the experiment planner for an operatic instrumental-extraction system. "
    "You do NOT hear the waveform. Use only the supplied measurements, configurations, "
    "learned-judge outputs, and human labels. Do not infer an acoustic property absent "
    "from the report. Prefer one-variable experiments, respect the supplied action and "
    "budget lists, and return STRICT JSON: either {\"actions\":[...]} using only the "
    "allowed action types, or a terminal {\"status\":\"final\",\"candidate_id\":...} or "
    "{\"status\":\"needs_human_ab\",...}."
)


class DeepSeekPlanner:
    """Real planner over DeepSeek V4 Pro (via Pi or the HTTP API). Requires
    ``DEEPSEEK_API_KEY``; replays ``reasoning_content`` across tool turns as the
    current API requires. Not exercised in unit tests (needs a live key)."""

    def __init__(self, api_key: str | None = None, model: str = "deepseek-reasoner",
                 base_url: str = "https://api.deepseek.com"):
        import os
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = base_url

    def propose(self, report: dict) -> dict:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set; use a live key on the GPU/host box")
        import json
        import urllib.request

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DEEPSEEK_SYSTEM},
                {"role": "user", "content": json.dumps(report)},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        return json.loads(body["choices"][0]["message"]["content"])


def validate_terminal(d: dict) -> tuple[bool, str]:
    """Validate the planner's only two allowed terminal shapes (docs/v2 §6.3)."""
    status = d.get("status")
    if status == "final":
        if not d.get("candidate_id"):
            return False, "final decision missing candidate_id"
        return True, ""
    if status == "needs_human_ab":
        if not (d.get("candidate_a") and d.get("candidate_b")):
            return False, "needs_human_ab requires candidate_a and candidate_b"
        return True, ""
    if status == "no_acceptable_candidate":
        if not d.get("reason"):
            return False, "no_acceptable_candidate requires a reason"
        return True, ""
    return False, f"unknown terminal status: {status!r}"


def _dedup_key(action: dict) -> str:
    body = {k: action[k] for k in sorted(action) if k not in ("reason", "note")}
    return repr(body)


class Conductor:
    """Drives the bounded loop. ``execute`` runs one validated action and returns a
    candidate record dict (must include ``recipe_id``); ``score`` maps the current
    candidate set to a report dict consumed by the planner."""

    def __init__(self, budget: Budget, execute: Callable[[dict], dict],
                 score: Callable[[list], dict], planner: Planner):
        self.budget = budget
        self.execute = execute
        self.score = score
        self.planner = planner
        self.log: list[dict] = []

    def validate_action(self, action: dict, state: ConductorState) -> tuple[bool, str]:
        atype = action.get("type")
        if atype not in TYPED_ACTIONS:
            return False, f"unknown action type: {atype!r}"
        if _dedup_key(action) in state.rendered:
            return False, "duplicate experiment (already cached)"
        rem = state.remaining(self.budget)
        if atype in _EXCERPT_ACTIONS and rem["excerpt_candidates"] <= 0:
            return False, "excerpt-candidate budget exhausted"
        if atype == "build_weighted_ensemble" and rem["ensembles"] <= 0:
            return False, "ensemble budget exhausted"
        if atype == "render_full_track" and rem["full_renders"] <= 0:
            return False, "full-render budget exhausted"
        if atype in _EXCERPT_ACTIONS and not action.get("changes_one_variable", True):
            return False, "experiment must change exactly one controlled variable"
        return True, ""

    def _account(self, action: dict, state: ConductorState) -> None:
        atype = action["type"]
        state.rendered.add(_dedup_key(action))
        if atype in _EXCERPT_ACTIONS:
            state.excerpt_candidates += 1
        elif atype == "build_weighted_ensemble":
            state.ensembles += 1
        elif atype == "render_full_track":
            state.full_renders += 1

    def run(self, candidates: list, report: dict) -> dict:
        """Run the bounded loop and return a validated terminal decision."""
        state = ConductorState()
        cands = list(candidates)

        while state.round < self.budget.max_rounds:
            report = {**report, "round": state.round, "budget_remaining": state.remaining(self.budget)}
            decision = self.planner.propose(report)

            if "status" in decision:
                ok, reason = validate_terminal(decision)
                if ok:
                    self.log.append({"event": "terminal", "round": state.round, "decision": decision})
                    return decision
                self.log.append({"event": "rejected_terminal", "reason": reason})
                break

            executed_any = False
            rendered_this_round = 0
            for action in decision.get("actions", []):
                ok, reason = self.validate_action(action, state)
                if not ok:
                    self.log.append({"event": "rejected_action", "action": action, "reason": reason})
                    continue
                if action["type"] == "stop_with_reason":
                    best = report.get("ranking", [{}])[0].get("recipe_id")
                    return {"status": "final", "candidate_id": best,
                            "confidence": action.get("confidence", 0.5),
                            "reason": action.get("reason", "planner stop")}
                is_new_candidate = (action["type"] in _EXCERPT_ACTIONS
                                    or action["type"] == "build_weighted_ensemble")
                if is_new_candidate and rendered_this_round >= self.budget.max_new_candidates_per_round:
                    self.log.append({"event": "rejected_action", "action": action,
                                     "reason": "per-round new-candidate budget exhausted"})
                    continue
                rec = self.execute(action)
                self._account(action, state)
                if rec:
                    cands.append(rec)
                    executed_any = True
                    if is_new_candidate:
                        rendered_this_round += 1

            report = self.score(cands)
            state.round += 1
            if not executed_any:
                break

        # Autonomous production: no human judge (oracle follow-up). At budget/round
        # exhaustion the run does NOT pause for a person and does NOT fabricate a
        # confident final — it returns the best available with no_acceptable_candidate.
        ranking = report.get("ranking", [])
        best = ranking[0]["recipe_id"] if ranking else None
        return {"status": "no_acceptable_candidate", "best_available": best,
                "reason": "probe/round budget exhausted without the selector accepting a finalist",
                "failed_gates": report.get("failed_gates", [])}
