# Counterfactual-Risk Router for Opera Stem Separation

> **Status:** dormant research prototype  
> **Product task:** `soloist_vs_rest`  
> **Immediate dependency:** certified PR #21 exact-routing result  
> **Inference contract:** no clean accompaniment, vocal truth, or oracle route is available at runtime

---

## 1. Decision

The first learned router should not be trained only to imitate one hard O2 label.

The preferred formulation is:

```text
mixture + frozen candidate estimates + optional target-singer query
    → per-candidate defect distributions
    → calibrated upper risks
    → deterministic constrained Potts decoder
    → route or abstention
```

Direct route imitation remains a baseline.

The risk formulation is preferable because one exact time/frequency cell with
`K` candidates and `D` defect axes supplies roughly `K × D` supervised values
rather than one class label. It also preserves close calls, permits threshold
changes without retraining the representation, and separates learned perception
from the auditable structured optimizer.

---

## 2. Counterfactual target tensor

For exact work `j`, cell `i`, candidate `k`, and defect `d`, compute:

\[
r_{j,i,k,d}.
\]

The first defect vector is:

```text
accompaniment transfer / event-hole damage
retained featured-solo voice
orthogonal artifact
stereo and hall deviation
transient loss and excess
direct-target fallback error
```

The tensor is counterfactual because every candidate is scored at every exact
cell, not only the candidate selected by O2.

Training metadata remain group-atomic:

```text
work
singer
session
ensemble
room
mastering chain
candidate family
```

---

## 3. Student outputs

For inference-available features `x_i`, candidate `k`, and optional target singer
embedding `q`, predict a defect distribution:

\[
(\hat\mu_{i,k,d},\hat Q_{i,k,d}(0.5),
 \hat Q_{i,k,d}(0.9),\hat Q_{i,k,d}(0.95)).
\]

The structured route uses a calibrated upper value:

\[
U_{i,k,d}=\hat Q_{i,k,d}(1-\delta_d).
\]

A useful first model is a small shared encoder followed by candidate- and
defect-specific heads. Candidate identity, feature schema, query encoder, audio
grid, and model checkpoint are all content-bound.

The model predicts **defects**, not one opaque quality score.

---

## 4. Why direct imitation is insufficient

O2 supplies one label:

\[
y_i=\arg\min_k C_i(k)
\]

after a global smoothness solve.

That target discards:

- the exact costs of the `K-1` unselected candidates;
- whether the winner beat the runner-up by `1e-6` or by 10 dB;
- which defect caused the choice;
- whether every candidate was unacceptable;
- how a changed leakage ceiling would alter the decision;
- whether the same label was selected mostly because of its neighbors.

Hard-label training should therefore mask cells whose exact best/second-best
margin is below a frozen threshold. It remains a useful auxiliary loss, not the
sole objective.

---

## 5. Deterministic constrained decoder

Let `x_{i,k} ∈ {0,1}` choose one candidate in cell `i`.

A candidate is locally feasible only when every calibrated critical upper risk
passes:

\[
U_{i,k,d}\le\tau_d
\quad\forall d\in\mathcal D_{\rm critical}.
\]

The first decoder solves:

\[
\min_x
\sum_{i,k} c_{i,k}x_{i,k}
+
\lambda_t\sum_{(i,j)\in E_t}\mathbf1[z_i\ne z_j]
+
\lambda_f\sum_{(i,j)\in E_f}\mathbf1[z_i\ne z_j],
\]

subject to:

\[
\sum_kx_{i,k}=1
\]

and:

\[
x_{i,k}=0
\quad\text{when candidate `k` is not confidently feasible.}
\]

The unary `c_{i,k}` combines distance below the critical thresholds and
secondary fidelity damage.

If any cell has no confidently feasible candidate, the learned route abstains
and returns the unchanged conservative whole-track parent. It does not force the
least-bad local label.

The reference prototype solves this finite Potts problem globally with SciPy
MILP and binds the plan to:

```text
ordered candidate IDs
risk-model ID
feature-contract ID
query-encoder ID
router policy
plan shape and int32 bytes
```

There is no channel axis in the plan. One label owns both stereo channels.

---

## 6. Later tail-aware decoder

Hard per-cell thresholds may be unnecessarily strict. A later preregistered
decoder can use weighted violation budgets or CVaR.

For defect `d`:

\[
\operatorname{CVaR}_{\alpha_d}(R_d)
=
\eta_d+
\frac{1}{(1-\alpha_d)W}
\sum_i w_i s_{i,d},
\]

with:

\[
s_{i,d}\ge
\sum_k U_{i,k,d}x_{i,k}-\eta_d,
\qquad s_{i,d}\ge0.
\]

The Potts labels remain integer; `η_d,s_{i,d}` are continuous. This yields a
mixed-integer linear program when the upper risks are constants.

Use lexicographic solves:

1. minimize critical violation/CVaR;
2. freeze the optimum within tolerance;
3. minimize secondary distortion and route complexity.

Do not introduce this until the simpler hard-feasibility decoder is validated.

---

## 7. Pairwise transition costs

A fixed switch penalty is only the first model.

Candidate transitions have different audible risks. An inference-available edge
cost can use the local candidate difference:

\[
p_{ij}(k,l)
=
\lambda_0\mathbf1[k\ne l]
+
\lambda_\Delta
\left\|
W_{ij}(\hat Y_k-\hat Y_l)
\right\|.
\]

Use a true metric norm if an alpha-expansion solver is desired. Otherwise retain
the exact MILP.

Transition features never use clean truth at inference.

---

## 8. Query conditioning

A target-singer query is useful when candidates differ in which voice material
they remove.

Inputs may include:

```text
target singer enrollment embedding
candidate-to-query identity evidence
removed-component/query coherence
query-absent evidence
```

But a router cannot produce audio outside its frozen candidate bank.

Therefore:

```text
generic candidates contain target-specific alternatives
    → query-conditioned risk router can select them

every candidate removes chorus/non-target soloists
    → query routing cannot restore those sources
    → query-conditioned correction/full separator is required
```

This distinction must be measured on exact duet/chorus scenes.

---

## 9. Training losses

For each defect head:

- heteroscedastic or quantile regression;
- threshold-violation BCE;
- within-cell candidate ranking;
- work-balanced sampling.

Auxiliary structured losses:

- masked hard O2 label loss;
- pairwise best-versus-runner-up preference;
- route-regret loss evaluated through a frozen decoder on small scenes.

Do not backpropagate through the exact MILP in v0. Train risks first; use
end-to-end route regret only as evaluation.

---

## 10. Calibration and abstention

Calibration is work-grouped, not crop-random.

Required checks:

```text
upper-quantile empirical coverage
catastrophic false-safe rate
coverage versus abstention
candidate-ranking accuracy
complete-route exact regret
worst-work regret
unseen singer/session/room
unseen candidate parameter setting
one held-out separator family
```

As the uncertainty threshold tightens:

```text
coverage must decrease
accepted-route failure must also decrease
```

If selective reliability does not improve, uncertainty is not useful.

---

## 11. Direct-gate versus risk-router experiment

Equal-compute arms:

### D0 — direct route imitation

```text
features (+ query)
→ label probabilities / convex weights
```

Training target:

```text
O2 label on high-margin exact cells
```

### R0 — counterfactual risk student

```text
features (+ query)
→ K × D defect upper risks
→ fixed deterministic Potts decoder
```

### Evaluation

On group-held-out works:

```text
per-defect prediction/calibration
hard-label accuracy on high-margin cells
complete-route regret
catastrophic retained-voice violations
catastrophic event-hole violations
artifact/stereo/hall regressions
abstention and coverage
```

Promotion rule:

```text
R0 materially lower exact route regret or catastrophic false-safe rate:
    use risk router

D0 and R0 fail on the same cells because all members fail:
    candidate bank is insufficient

query swap changes the correct target but all members are unchanged:
    build target-singer correction/separator
```

---

## 12. Reference prototype

The accompanying module:

```text
audio_extract/counterfactual_risk_router.py
```

implements:

- identity-bound candidate/metric panel;
- upper-risk feasibility;
- near-tie-aware exact teacher targets;
- global Potts MILP;
- shared-stereo labels;
- deterministic plan hash;
- fail-closed whole-track abstention.

Its synthetic suite covers:

- near-tie masking;
- complementary routing;
- switch regularization;
- missing/unsafe predictions;
- no-feasible-candidate abstention;
- candidate-order identity;
- upper-bound threshold crossing;
- exact lack of a channel route axis.

It is a dormant research component. It does not authorize learned routing before
the certified PR #21 report is complete.
