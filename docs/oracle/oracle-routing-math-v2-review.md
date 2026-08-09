# Certified oracle-routing mathematics v2

This document records the binding corrections to the exploratory exact router introduced in `1c96556`.

## Scope

The router is an exact-reference diagnostic. It may use known accompaniment `A` and featured-solo voice `V` to determine whether immutable separator candidates contain locally complementary answers. It is not a production selector.

## Cell objective

For an identifiable cell, fit every candidate accompaniment estimate as

\[
Y_i=\alpha_iA+\beta_iV+R_i.
\]

For real simplex weights `w`, use

\[
L(w)=\lambda_\alpha|\alpha(w)-1|^2+
\lambda_\beta\frac{|\beta(w)|^2E_V}{E_A+\epsilon}+
\lambda_R\frac{\|R(w)\|^2}{E_A+\epsilon}.
\]

The denominators are independent of `w`, hence

\[
L(w)=w^TQw-2c^Tw+k
\]

is convex. The former one-sided `max(1-|alpha|,0)^2` term and `|alpha|`-dependent voice denominator must not be used for a claimed convex-hull envelope.

## Exact fallback

When the local `[A,V]` basis is unavailable, do not assign zero cost. Define

\[
e_i=Y_i-A,
\qquad
L_{direct}(w)=\lambda_d\frac{\|\sum_iw_ie_i\|^2}{\max(E_A+E_V,E_{floor})}.
\]

This covers no-vocal, vocal-only, silent, and ill-conditioned cells. Every exact non-silent cell contributes to O1/O2/O3.

## O2

O2 is a global finite Potts-labeling problem over one actual stereo candidate per time/frequency cell. A MILP result counts as an oracle envelope only when the solver reports an incumbent with the requested optimality gap. A time-limited incumbent without a certified gap is exploratory evidence only.

## O3

O3 uses simplex weights and squared temporal/frequency smoothness. Optimize the convex quadratic directly in weight space. Publish:

```text
simplex feasibility
objective monotonicity
projected-gradient norm or primal/dual gap
iteration count
convergence status
agreement across deterministic starts
```

An iteration-limit result without a convergence certificate is invalid evidence.

## Candidate basis

The binding basis contains all unique, hash-verified candidates, deduplicated by decoded PCM identity:

```text
current waveform median champion
STFT geometric median
uniform convex fusion
MDX23C residual
Mel-Band residual
BS-RoFormer residual
HTDemucs 04573f0d
HTDemucs 955717e8
```

O2/O3 must beat the current median and MDX fallback, not merely the best single member.

## Transform controls

Publish both raw and identity-routed controls:

```text
O1_raw
O1_stft_identity
median_raw
median_stft_identity
```

This separates route changes from STFT/ISTFT round-trip damage.

## Binding gate

A route is actionable only when it:

1. beats the current median champion;
2. improves one critical target axis by at least 1.5 dB;
3. regresses the opposing critical axis by no more than 0.5 dB;
4. preserves Aalto retained voice, event holes, stereo width/coherence, and hall behavior;
5. keeps no-vocal false-positive energy within 1.05x;
6. preserves orthogonal artifact, transient, seam, and worst-event limits;
7. reopens as immutable exact FLOAT.

Run a frozen resolution sensitivity table at 2.0, 1.0, and 0.5 seconds. A large gap on a coarse grid is meaningful; a small gap is not evidence of basis insufficiency until shorter operatic events are represented.
