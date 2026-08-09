# Counterfactual risk student with deterministic Potts decode

Status: synthetic, CPU-only research prototype; not wired into production and not
authorized for a GPU run. The accepted frozen median remains the conservative
step-zero fallback.

## Decision being tested

A direct router learns one hard label per exact time/frequency cell. That label
discards the measured outcomes of the other candidates and becomes arbitrary when
two candidates are nearly tied. The alternative is a risk student:

```text
features (+ optional singer query)
    -> calibrated per-candidate defect upper bounds
    -> deterministic constrained Potts decoder
    -> shared-stereo route or abstention
```

For each exact development cell, the offline target is

\[
r_{t,b,k,d},
\]

where `t` is time, `b` is frequency band, `k` is a frozen candidate, and `d`
indexes retained voice, accompaniment hole/transfer, orthogonal artifact, and
secondary stereo/hall/transient defects. This supplies `K x D` counterfactual
labels per cell instead of one route class. Clean references are used only to
construct these development labels; the inference API accepts no truth waveform.

## Calibrated upper heads

The executable prototype fits one linear head per `(candidate, defect)`, then
uses a held-out split-conformal residual order statistic for an upper quantile:

\[
\hat r^+_{k,d}(x)=\max(0,\hat r_{k,d}(x)+q_{k,d}).
\]

A production student may replace the linear model with shared features and
multi-head quantile loss, but it must retain group-held-out calibration. A head
that fails coverage calibration cannot become a route signal.

## Lexicographic route objective

Let `C` be the critical voice/hole/artifact heads and `tau_d` their frozen limits.
A candidate is feasible in a cell only when

\[
\hat r^+_{t,b,k,d} \leq \tau_d \quad \forall d \in C.
\]

Feasibility is a hard first stage. Among globally feasible routes, the decoder
minimizes

\[
\sum_{t,b,d\notin C}w_d\hat r^+_{t,b,z_{t,b},d}
+\lambda_t\sum_{t,b}[z_{t,b}\ne z_{t-1,b}]
+\lambda_f\sum_{t,b}[z_{t,b}\ne z_{t,b-1}].
\]

The route index has no channel dimension, so left and right cannot split. Fixed
candidate ordering provides deterministic tie-breaking. The prototype enumerates
small grids exactly; production would use the already certified MILP machinery.

If any cell has no feasible candidate, or its offline exact min-margin is below
the frozen confidence threshold, the decoder abstains and returns the frozen
fallback route. It never fills a failed cell with an uncertified local guess.
The plan identity hashes calibrated upper risks, policy, and the route/fallback.

## Near ties and sample efficiency

During label construction, feasible candidates are ranked by exact secondary
cost. Cells whose best-versus-second-best margin is below a frozen threshold are
masked rather than assigned an unstable hard label. A full MILP experiment may
replace this local margin with exact min-marginals. Either way, the risk student
still learns all reliable `K x D` outcomes; direct imitation receives at most one
unmasked class.

This is the sample-efficiency advantage: a rare catastrophic outcome for an
unselected candidate remains supervised instead of disappearing behind the O2
winner.

## Query conditioning limits

A singer query can help only if the frozen candidate bank contains distinct
outcomes for the requested identity. If every candidate removes chorus or
non-target singers together with the soloist, no router can synthesize the
missing semantic answer. That failure requires a query-conditioned separator or
new correction candidate, not a more elaborate route.

## Grouped evaluation and stopping rule

Split by work, singer, session, venue, and mastering chain; related variants stay
in one group. Compare direct imitation and the risk student on:

- exact selection regret versus the certified counterfactual optimum;
- catastrophic false-safe rate (predicted feasible, exact critical failure);
- upper-head calibration coverage by defect and candidate;
- abstention rate and fallback quality;
- route switching and plan determinism;
- complete-work retained voice, holes, artifacts, stereo, transients, and hall.

The decision is:

```text
large exact risk-prediction advantage over direct imitation -> risk router
same cells fail for every member                         -> correction/new query separator
all members remove non-target voice                      -> target-singer separator required
```

No student advances unless it starts at the frozen fallback, has a content-bound
route identity, never sees exact truth at inference, and passes the existing
complete-work gate. This lane does not change the current Cantolopera fine-tune
`STOP_AT_100 / NO PROMOTION` decision.
