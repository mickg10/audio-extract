from audio_extract.oracle_binding_gate_abstract import (
    AbstractCell,
    AbstractDecision,
    AbstractReport,
    METHODS,
    PRIMARY_RESOLUTION,
    RESOLUTIONS,
    SENSITIVITY_RESOLUTIONS,
    decide,
    exhaustive_assignments,
)


def _all(valid=True, passes=False):
    return {
        (method, resolution): AbstractCell(valid=valid, passes=passes)
        for method in METHODS
        for resolution in RESOLUTIONS
    }


def test_exhaustive_finite_state_invariants():
    visited = 0
    decisions = set()
    for selected in METHODS:
        for report in exhaustive_assignments(selected):
            visited += 1
            decision = decide(report)
            decisions.add(decision)
            all_valid = all(cell.valid for cell in report.cells.values())
            primary = report.cells[(selected, PRIMARY_RESOLUTION)].passes
            sensitivity = any(
                report.cells[(selected, resolution)].passes
                for resolution in SENSITIVITY_RESOLUTIONS
            )

            if decision is AbstractDecision.INVALID_EVIDENCE:
                assert not all_valid
            elif decision is AbstractDecision.ACTIONABLE:
                assert all_valid
                assert primary
            elif decision is AbstractDecision.RESOLUTION_SENSITIVE:
                assert all_valid
                assert not primary
                assert sensitivity
            elif decision is AbstractDecision.NO_ACTIONABLE_GAP:
                assert all_valid
                assert not primary
                assert not sensitivity
            else:  # pragma: no cover
                raise AssertionError(decision)

    assert visited == (
        len(METHODS)
        * 2 ** (2 * len(METHODS) * len(RESOLUTIONS))
    )
    assert decisions == set(AbstractDecision)


def test_other_method_primary_pass_cannot_promote_selected_method():
    cells = _all(valid=True, passes=False)
    cells[("O3", PRIMARY_RESOLUTION)] = AbstractCell(True, True)
    report = AbstractReport(selected_method="O2", cells=cells)
    assert decide(report) is AbstractDecision.NO_ACTIONABLE_GAP


def test_cross_method_sensitivity_fragments_cannot_form_actionable():
    cells = _all(valid=True, passes=False)
    cells[("O2", "sensitivity_a")] = AbstractCell(True, True)
    cells[("O3", "sensitivity_b")] = AbstractCell(True, True)
    assert decide(AbstractReport("O2", cells)) is (
        AbstractDecision.RESOLUTION_SENSITIVE
    )
    assert decide(AbstractReport("O3", cells)) is (
        AbstractDecision.RESOLUTION_SENSITIVE
    )


def test_sensitivity_success_never_becomes_actionable():
    cells = _all(valid=True, passes=False)
    cells[("O2", "sensitivity_a")] = AbstractCell(True, True)
    assert decide(AbstractReport("O2", cells)) is (
        AbstractDecision.RESOLUTION_SENSITIVE
    )


def test_truthy_gate_bit_on_invalid_evidence_is_ignored_fail_closed():
    cells = _all(valid=True, passes=False)
    cells[("O2", PRIMARY_RESOLUTION)] = AbstractCell(False, True)
    assert decide(AbstractReport("O2", cells)) is (
        AbstractDecision.INVALID_EVIDENCE
    )


def test_closed_assignment_refuses_missing_or_extra_cells():
    cells = _all()
    cells.pop(("O3", "sensitivity_b"))
    report = AbstractReport("O2", cells)
    try:
        decide(report)
    except ValueError as exc:
        assert "cell set differs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing cell accepted")

    cells = _all()
    cells[("O4", "primary")] = AbstractCell(True, False)
    report = AbstractReport("O2", cells)
    try:
        decide(report)
    except ValueError as exc:
        assert "cell set differs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("extra cell accepted")
