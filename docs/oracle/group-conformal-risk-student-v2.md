# Independent-family conformal risk student v2

This is a dormant CPU research contract. It does not train a production model,
render audio, or authorize routing.

## Coverage unit

The exchangeability unit is a canonical `source_family_sha256`, not a cell,
excerpt, mix variant, or caller-chosen work string. Training, tuning, and
calibration source-family sets must be pairwise disjoint.

For calibration family \(g\), the nonconformity score is

\[
S_g = \max_{i,k,d\in g}
\frac{R_{i,k,d}-\widehat R_{i,k,d}}{s_d},
\]

where \(s_d>0\) is the frozen metric scale. The rank is

\[
r=\lceil (G+1)q\rceil.
\]

A finite certificate is refused when `r > G`. The certificate recomputes both
`r` and `r/(G+1)` during validation; neither field is trusted from serialized
input.

## Frozen selection boundary

Model architecture and all hyperparameters are represented by a
`FrozenSelectionProtocolV2`. Its training and tuning manifests are distinct from
the calibration manifest, and `calibration_data_consulted` must be false.

The calibration certificate additionally binds:

- exchangeability-scope identity;
- query-quality stratum;
- calibration algorithm revision;
- selected model and hyperparameters;
- candidate, feature, and metric contracts;
- exact coefficients and metric scales;
- training/tuning/calibration/split manifests;
- code commit and numerical fit policy.

## Numerical fit

The linear baseline uses augmented least squares rather than explicitly forming
`X.T @ X`. The intercept is not regularized. The augmented design rank and
condition number are checked before coefficients can be certified.

## Inference boundary

`predict_point()` and `predict_upper()` accept features only. Clean references,
exact risks, source-family IDs, and split identities are absent from the
inference API.
