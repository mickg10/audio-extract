# Counterfactual D0/R0 dataset contract v2

This dormant contract stores one immutable work block per exact source grid. It
contains no training loop and does not authorize an audio or GPU run.

## Ordered axes

Every block uses the same ordered candidate panel and the same complete metric
axis:

```text
critical_metrics + secondary_metrics
```

Both `exact_risks` and `risk_available` therefore have shape:

```text
(cell, candidate, metric)
```

A dataset that declares hall, stereo, transient, or another secondary objective
but omits its exact label or availability is invalid.

## Group isolation

Every split must enforce the complete canonical group tuple:

```text
source_family_sha256
work_sha256
singer_sha256
session_sha256
ensemble_sha256
room_sha256
mastering_sha256
```

Callers cannot weaken the split to work identity alone. In particular, alternate
mixes or sessions derived from one source family cannot be divided between
training, calibration, and test.

## Privileged-label boundary

Inference features are restricted to the mixture and frozen candidate
accompaniments. Clean accompaniment/vocal references are used only for exact
offline labels. Their source PCM identities, the candidate lineage, array
payloads, cell grid, feature contract, risk contract, and group contract are all
identity-bearing.
