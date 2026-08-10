# Counterfactual-risk inference binding v1

The model-output and routing contracts are intentionally independent. This
bridge is the only admissible composition path between them.

Before a neural risk tensor can reach the v4 router, the bridge requires:

- model-axis candidate IDs equal hashes of the exact ordered recipe/decoded-PCM
  candidate identities;
- complete metric order equal the calibrated routing order;
- inference-time model state/config equal the verified model bundle;
- inference-time query encoder and condition equal the verified query bundle;
- selected quantile and calibration policy equal the v4 certificate;
- source PCM, feature contract, query-quality stratum and raw-parent certificate
  all validate under one panel identity.

The upper-risk and availability arrays are copied into read-only snapshots. The
binding record names the original inference payload and the resulting v4 panel.
The final route-plan identity additionally binds that bridge record and the exact
serialized label grid.

This module remains dormant research. It does not train a predictor, render
audio, relax abstention, authorize a GPU job, or enter the production selector.
