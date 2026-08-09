# Canonical baseline-delivery hardening contract

This document is the implementation contract for `tools/package_baseline_delivery.py`.
It supersedes the duplicate release implementation in draft PR #8. The canonical
release path remains the existing tool on `v2-design` because it already consumes
the canonical routing report, emits measurement and removed-source recipes, and
uses the repository's exact candidate identities.

## Required corrections

1. **Never hard-link immutable inputs into a mutable release tree.**
   Copy atomically to a temporary file in the destination directory, rename, then
   reopen and verify. Reject `os.path.samefile(source, destination)`.

2. **Freeze truth identities at measurement time.**
   The routing/exact-measurement report consumed by the packager must contain
   scored records for the mixture, exact accompaniment, and exact featured voice:

   ```text
   source path
   container SHA-256
   decoded-PCM SHA-256
   frames
   sample rate
   channel layout
   FLOAT subtype
   ```

   Before packaging, reverify all three files against those scored records. Never
   accept a newly computed hash as though it were the original scored truth.

3. **Reverify every candidate and destination.**
   Both primary and alternate must match the exact truth grid and their manifest
   container/PCM identities. Every copied destination must match the source's
   scored identities and occupy a distinct inode.

4. **Content-address the removed-vocal child.**
   `removed_vocal = mixture - primary_accompaniment` is a new immutable recipe
   node. Bind:

   ```text
   scored mixture PCM/grid
   primary candidate recipe ID
   primary candidate PCM identity
   operation=mixture_minus_source
   target=featured_solo_voice
   alignment=source-grid-exact
   audio-extract commit
   ```

   Persist normalized recipe JSON, recipe ID, explicit parents, execution
   fingerprint, recipe-file hash, container hash, decoded-PCM hash, and exact grid.
   Refuse any differing existing node.

5. **Use only repository terminal statuses.**
   Top-level and per-work JSON status is `final` when a candidate clears every
   declared screen, otherwise `needs_human_ab`. Use a separate `release_scope`
   field for `exact_benchmark_qualified` versus `engineering_preview`.

6. **Fail closed on unavailable evidence.**
   Require finite values for every declared metric, at least a configured number
   and fraction of identifiable exact-label tiles, and coherent
   `event_hole_db_max >= event_hole_db_p90`.

7. **Collapse byte-identical aliases before primary/alternate selection.**
   Keep logical recipes in the report, but primary and alternate must have
   different decoded-PCM identities.

8. **Inventory every emitted file.**
   The manifest records relative filename, container SHA-256, decoded-PCM SHA-256,
   frames, sample rate, channel layout, and subtype for:

   ```text
   primary accompaniment
   alternate accompaniment
   removed featured-solo voice
   exact accompaniment target
   exact voice target
   recipe/execution/measurement metadata
   ```

9. **Preserve immutable replay.**
   A second invocation with unchanged inputs returns byte-identical reports and
   package files. Any differing existing file causes refusal.

10. **Test all failure boundaries.**
    Include mutation tests for mixture, accompaniment, vocal target, primary,
    alternate, package destination, recipe parent identity, and evidence coverage.

## Claim boundary

The resulting package is an exact-linear-reference engineering release. It makes
no population-risk or arbitrary-master claim. Even `needs_human_ab` works should
still receive a clearly labelled primary preview plus byte-distinct alternate so
the concrete Verdi/Puccini/Donizetti/Aalto milestone is delivered.
