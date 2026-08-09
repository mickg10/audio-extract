# Oracle binding-gate finite-state model

This model isolates the final architecture-decision state machine from audio and
floating-point metric validity. Concrete code first validates every artifact,
metric, transform control, optimizer certificate, basis identity, and route
plan. The abstract state then contains one closed Boolean assignment for every
required `(method, resolution)` pair.

The promotion method and primary resolution are constants, not values selected
from the observed results. Promotion also requires the concrete v3 chronology:
a prewritten run input, valid semantic/container/sidecar dependency bindings, an
exact public issue-comment claim while the output root is absent, a successful
runner preflight, and an exact input/claim binding in the final report. The
state machine permits four terminal outcomes:

- `INVALID_EVIDENCE`: at least one required cell is invalid;
- `ACTIONABLE`: all evidence is valid and the preregistered method passes at the
  preregistered primary resolution;
- `RESOLUTION_SENSITIVE`: all evidence is valid, the primary fails, and the same
  preregistered method passes only at a frozen sensitivity resolution;
- `NO_ACTIONABLE_GAP`: all evidence is valid and the selected method passes
  neither primary nor sensitivities.

The invariants prove that missing evidence, another method's success,
sensitivity-only success, a post-output claim, a missing public anchor, or an
unbound v3 input/claim cannot become `ACTIONABLE`.

## TLC

With `tla2tools.jar` available:

```bash
java -XX:+UseParallelGC -cp tla2tools.jar tlc2.TLC \
  -config formal/oracle_binding_gate/OracleBindingGate.cfg \
  formal/oracle_binding_gate/OracleBindingGate.tla
```

The committed preregistration Python mirror exhaustively enumerates all
`2^17 = 131072` publication/evidence assignments in ordinary CI. The separate
closed method-resolution gate still enumerates its complete cell matrix:

```bash
python -m pytest -q tests/test_oracle_binding_gate_abstract.py
python -m pytest -q tests/test_oracle_binding_preregistration_abstract.py
```

The model does not prove that an audio metric is perceptually valid or correctly
computed. Those claims remain separate property tests and exact-artifact gates.
