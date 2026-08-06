from audio_extract import conductor as cd


def _mk(proposals, budget=None):
    calls = {"execute": 0}

    def execute(action):
        calls["execute"] += 1
        return {"recipe_id": f"r{calls['execute']}"}

    def score(cands):
        return {"ranking": [{"recipe_id": c["recipe_id"]} for c in cands]}

    c = cd.Conductor(budget or cd.Budget(), execute, score, cd.ScriptedPlanner(proposals))
    return c, calls


def test_reaches_final_decision():
    proposals = [
        {"actions": [
            {"type": "change_overlap", "overlap": 4, "changes_one_variable": True},
            {"type": "run_construction", "construction": "native_secondary", "changes_one_variable": True},
        ]},
        {"status": "final", "candidate_id": "cand-best", "confidence": 0.9},
    ]
    c, calls = _mk(proposals)
    decision = c.run([{"recipe_id": "base"}], {"ranking": [{"recipe_id": "base"}]})
    assert decision["status"] == "final"
    assert decision["candidate_id"] == "cand-best"
    assert calls["execute"] == 2


def test_excerpt_budget_enforced():
    big = [{"type": "change_overlap", "overlap": i, "changes_one_variable": True} for i in range(40)]
    # raise the per-round cap so this isolates the TOTAL excerpt budget
    c, calls = _mk([{"actions": big}], cd.Budget(max_excerpt_candidates=30, max_new_candidates_per_round=40))
    c.run([{"recipe_id": "base"}], {"ranking": []})
    assert calls["execute"] == 30  # 40 proposed, total budget caps at 30
    assert any(e["event"] == "rejected_action" for e in c.log)


def test_per_round_candidate_cap():
    big = [{"type": "change_overlap", "overlap": i, "changes_one_variable": True} for i in range(10)]
    c, calls = _mk([{"actions": big}], cd.Budget(max_new_candidates_per_round=6))
    c.run([{"recipe_id": "base"}], {"ranking": []})
    assert calls["execute"] == 6  # 10 proposed in one round, per-round cap is 6
    assert any("per-round" in e.get("reason", "") for e in c.log)


def test_invalid_action_rejected():
    c, calls = _mk([{"actions": [{"type": "teleport"}]}], )
    c.run([{"recipe_id": "base"}], {"ranking": [{"recipe_id": "base"}]})
    assert calls["execute"] == 0
    assert any(e["event"] == "rejected_action" and "unknown action" in e["reason"] for e in c.log)


def test_duplicate_experiment_rejected():
    dup = {"type": "change_overlap", "overlap": 4, "changes_one_variable": True}
    c, calls = _mk([{"actions": [dup, dict(dup)]}])   # same experiment twice
    c.run([{"recipe_id": "base"}], {"ranking": []})
    assert calls["execute"] == 1  # second is a cache/dup hit


def test_needs_human_ab_path():
    proposals = [{"actions": [{"type": "request_human_comparison", "candidate_a": "a",
                               "candidate_b": "b", "passages": ["seg_004"], "question": "which?"}]}]
    c, _ = _mk(proposals)
    decision = c.run([{"recipe_id": "base"}], {"ranking": []})
    assert decision["status"] == "needs_human_ab"
    assert decision["candidate_a"] == "a" and decision["candidate_b"] == "b"


def test_two_round_cap_defers_to_human():
    proposals = [
        {"actions": [{"type": "change_overlap", "overlap": 2, "changes_one_variable": True}]},
        {"actions": [{"type": "change_overlap", "overlap": 4, "changes_one_variable": True}]},
        {"actions": [{"type": "change_overlap", "overlap": 6, "changes_one_variable": True}]},  # never reached
    ]
    c, calls = _mk(proposals, cd.Budget(max_rounds=2))
    decision = c.run([{"recipe_id": "base"}], {"ranking": [{"recipe_id": "base"}]})
    # No fabricated final at the cap: with >=2 candidates, hand off to human A/B.
    assert decision["status"] == "needs_human_ab"
    assert calls["execute"] == 2                  # exactly two rounds ran


def test_validate_terminal_schema():
    assert cd.validate_terminal({"status": "final", "candidate_id": "x"})[0]
    assert not cd.validate_terminal({"status": "final"})[0]
    assert cd.validate_terminal({"status": "needs_human_ab", "candidate_a": "a", "candidate_b": "b"})[0]
    assert not cd.validate_terminal({"status": "needs_human_ab", "candidate_a": "a"})[0]
    assert cd.validate_terminal({"status": "no_acceptable_candidate", "reason": "all leak badly"})[0]
    assert not cd.validate_terminal({"status": "no_acceptable_candidate"})[0]   # reason required
    assert not cd.validate_terminal({"status": "bogus"})[0]
