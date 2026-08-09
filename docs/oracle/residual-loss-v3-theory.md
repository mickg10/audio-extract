# Stable residual-loss v3: theory, precision policy, and executable claims

> **Task:** `soloist_vs_rest`  
> **Production construction:** `A_hat = M - V_hat`  
> **Implementation:** `audio_extract/classical_loss_v3.py`  
> **Status:** version-separated and unwired; this document does not reopen whole-model HTDemucs continuation

## 1. One independent source error

For an exact training row:

\[
M=A+V,
\]

where \(A\) is retained accompaniment and \(V\) is the featured soloist to remove.

The production accompaniment is constructed as:

\[
\hat A=M-\hat V.
\]

Therefore:

\[
\hat A-A
=
M-\hat V-A
=
V-\hat V
=
-(\hat V-V).
\]

The accompaniment and vocal waveform errors are the same residual with opposite sign.

Consequences:

```text
score the primary source residual once

not:
    waveform(A_hat, A) + waveform(V_hat, V)
    STFT(A_hat, A) + STFT(V_hat, V)
```

Duplicating those terms changes only their weighting when the normalizers differ. It does not add independent source information.

The v3 primary residual is:

\[
e=\hat V-V.
\]

## 2. Why target-normalized MR-STFT failed

The failed v1 objective used a form equivalent to:

\[
\frac{\|\operatorname{STFT}(\hat V-V)\|}
{\operatorname{mean}|\operatorname{STFT}(V)|+\epsilon}.
\]

When \(V\) is nearly silent, the denominator can approach the absolute epsilon while the numerator remains ordinary model error. The Donizetti run reproduced this failure as an approximately `34k` complex-STFT term and a pre-clip gradient norm near `19M`.

V3 instead normalizes the one residual to the mixture/reference scale:

\[
L_{\mathrm{STFT}}
=
\frac1R
\sum_{r=1}^{R}
\log\left(
1+
\frac{
\operatorname{mean}
|\operatorname{STFT}_r(e)|
}{
\max(
\operatorname{mean}|\operatorname{STFT}_r(M)|,
F_r
)
}
\right).
\]

The reference floor \(F_r\) is explicit and identity-bearing.

This gives:

- zero loss at zero error;
- monotonicity in residual magnitude;
- bounded growth under nearly silent targets;
- approximate global-gain invariance whenever the floor is inactive.

## 3. Explicit analysis precision

Model tensors may be produced under float16, bfloat16, float32, or float64 training.

The loss must not inherit accidental backend/autocast behavior. V3 freezes:

```text
if any audio input is float64:
    analysis dtype = float64
else:
    analysis dtype = float32
```

The policy is recorded as:

```text
float64_if_any_input_float64_else_float32
```

Numerically sensitive analysis uses that dtype:

```text
waveform normalization
MR-STFT
Hann windows
mid/side preservation
event weighting
source-coordinate fitting
```

The cast from the model estimate is differentiable, so gradients return to the original model parameter dtype.

Identity checks are detached and evaluated in float64 over the represented operands. They are diagnostics, not loss terms.

## 4. Exact identity checks

### Target identity

Require:

\[
M=A+V.
\]

### Residual construction

Require the construction directly:

\[
\hat A=M-\hat V.
\]

Checking `A_hat + V_hat == M` adds another rounded operation and can reject a correct FP16/BF16 subtraction. V3 compares \(\hat A\) to a float64 subtraction of the already represented \(M\) and \(\hat V\).

The allowed error is the larger of:

- the configured relative tolerance;
- a dtype- and magnitude-aware roundoff bound.

A material mismatch is refused.

## 5. Primary residual terms

### Waveform term

\[
L_{\mathrm{wave}}
=
\frac{
\operatorname{mean}|e|
}{
\max(\operatorname{mean}|M|,F_w)
}.
\]

### Event-weighted term

For nonnegative event weights \(q\):

\[
L_{\mathrm{event}}
=
\frac{
\sum q|e|/\sum q
}{
\max(\sum q|M|/\sum q,F_w)
}.
\]

This is the same source residual under an explicitly different sampling measure. It is not a second accompaniment/vocal source copy.

### Stereo term

Mid and side are:

\[
X_m=(X_L+X_R)/2,
\qquad
X_s=(X_L-X_R)/2.
\]

V3 scores accompaniment mid/side error once, normalized by target mid/side magnitude. This intentionally reweights spatial structure; it does not claim to be an independent waveform source error.

## 6. Exact A-only and V-only controls

The mixture crop alone does not uniquely distinguish:

```text
missed soloist
from
orchestra stolen into the vocal estimate
```

V3 therefore requires actual control forwards in the function signature.

### A-only control

Input:

\[
M_A=A,
\qquad
V_{\mathrm{target}}=0.
\]

Require:

\[
F_V(A)\approx0.
\]

This directly penalizes orchestral false positives.

### V-only control

Input:

\[
M_V=V.
\]

Require:

\[
F_V(V)\approx V.
\]

This directly penalizes missed target voice.

Missing control outputs cannot silently become zero labels.

## 7. Local source-coordinate term

On a local exact crop, decompose the accompaniment estimate:

\[
\hat A=\alpha A+\beta V+R.
\]

A real two-column ridge fit estimates \(\alpha\), \(\beta\), and residual \(R\).

The intended interpretations are:

- \(|\alpha-1|\): accompaniment transfer error;
- \(|\beta|\): retained target coefficient;
- \(R\): unexplained artifact.

A raw \(|\beta|\) assigns the same cost to the same coefficient when the target voice is inaudibly quiet or dominant. V3 uses an audibility-weighted term:

\[
|\beta|
\sqrt{
\frac{\|V\|^2+\epsilon}
{\|A\|^2+\epsilon}
}.
\]

The residual term is:

\[
\sqrt{
\frac{\|R\|^2+\epsilon}
{\|A\|^2+\epsilon}
}.
\]

Ill-conditioned or zero-source crops are masked from this attribution term. They remain covered by the primary residual and explicit controls.

## 8. Analytic 2×2 solve

For flattened real signals, define:

\[
G=
\begin{bmatrix}
A^TA+\lambda & A^TV\\
A^TV & V^TV+\lambda
\end{bmatrix},
\]

\[
b=
\begin{bmatrix}
A^T\hat A\\
V^T\hat A
\end{bmatrix}.
\]

V3 computes the two coefficients through the analytic determinant formula rather than relying on half-precision `torch.linalg.solve` or `torch.linalg.cond` support.

The raw, detached Gram condition number is obtained from the closed-form eigenvalues of the symmetric 2×2 matrix.

## 9. Executable claim matrix

| Claim | Automated evidence |
|---|---|
| accompaniment and vocal errors are one residual | exact float64 identity fixture |
| the residual is scored once | waveform-value equality fixture |
| mixture-normalized loss is gain-invariant above the floor | 0.25× / 1× / 4× property test |
| A-only control pushes the vocal estimate toward zero | gradient-direction fixture |
| V-only control pushes the estimate toward the target | gradient-direction fixture |
| retained-voice coefficient is audibility weighted | low/high target-level comparison |
| float16/bfloat16 CPU forward/backward are supported | mixed-precision fixtures |
| float64 remains float64 | analysis-policy fixture |
| low-precision represented inputs match float32 analysis | value/gradient comparison |
| near-silent targets have finite bounded gradients | adversarial spectral fixture |
| material residual-construction mismatch is refused | identity-failure fixture |
| ambient CUDA autocast does not define the policy | conditional CUDA fixture |

## 10. What this theory does not prove

It does not prove that:

- the chosen scalar weights match human preference;
- waveform/STFT/event/spatial reweighting is Pareto-optimal;
- the current exact corpus covers all opera acoustics;
- the full separator should be fine-tuned;
- a source-coordinate fit is identifiable on every crop;
- the loss alone prevents catastrophic forgetting.

Those are experimental gates.

## 11. Authorized use

The v3 loss is eligible for a future low-capacity frozen-member gate or zero-initialized correction student after review.

It does **not** authorize:

```text
resuming the failed HTDemucs whole-body folds
rewriting historical v1/v2 experiment semantics
promoting from crop loss alone
omitting complete-work and no-vocal evaluation
```
