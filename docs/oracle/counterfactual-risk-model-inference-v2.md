# Identity-bound counterfactual-risk inference v2

This is a dormant inference contract. It does not train a model, render audio, or
authorize routing.

## Feature boundary

The feature batch carries the exact ordered candidate identities and feature
contract next to `(B,K,C,T,F)` values and the non-learned availability mask. A
same-shaped tensor in another candidate order is rejected before forward.

## Inference-time provenance

The returned object binds and clones:

- model configuration identity;
- exact sorted model-state tensor bytes;
- candidate and metric order;
- feature and hard-availability tensor identities;
- exact query-encoder identity;
- each query embedding's original dtype, shape, and bytes;
- the complete ordered quantile-level axis.

Panel construction recomputes the current model/config identities and refuses a
post-forward weight change or same-shaped output from another model.

## Selection boundary

A panel records the selected quantile level and index, the complete quantile
axis, availability-probability threshold, and calibration-policy identity.
Therefore a median prediction cannot be mistaken for a calibrated upper bound.
The exact upper-risk and availability arrays are also content-hashed.
