# Counterfactual-risk router v4

This remains dormant CPU research. It invokes the already versioned zero-gap v3
solver only after the stronger v4 identity contract passes.

## Query-calibrated upper risks

The calibration certificate now binds:

- exact query-encoder bundle;
- frozen query-quality stratum;
- selected risk-quantile level;
- calibration algorithm identity;
- model, feature, candidate-panel, policy, data and split identities;
- complete metric-specific coverage rows.

A certificate calibrated for a different singer-query encoder, missing/poor query
stratum, or different quantile cannot authorize the same risk panel.

## Route publication

`RouteDecisionV4.to_dict()` publishes the exact two-dimensional integer label
grid, its shape, dtype and content hash. The report is independently renderable;
a plan hash and label shape alone are insufficient.

## Exhaustive mirror

The tiny-grid mirror is two-pass:

1. enumerate and measure every feasible route;
2. compute the exact minimum;
3. collect routes within the frozen uniqueness tolerance of that minimum.

Tolerance therefore cannot ratchet the reported optimum upward as enumeration
proceeds.
