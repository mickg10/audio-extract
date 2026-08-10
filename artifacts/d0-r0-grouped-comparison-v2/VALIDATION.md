# Validation record — D0/R0 grouped comparison v2

Separate branch:

```text
bigoracle/d0-r0-grouped-comparison-artifacts-20260810
```

V2 adds content-bearing unit/arm/submission provenance, exact oracle-availability
semantics, and strict summary types.

The implementation and v2 tests were executed in an isolated package using
interface-compatible versions of the repository's v1 preregistration,
submission, and complete-route-evaluation contracts.

Command:

```bash
python -m pytest -q
```

Focused result before GitHub upload:

```text
.............                                                            [100%]
13 passed in 0.09s
```

Additional checks:

```text
python -m py_compile
no source or test line longer than 88 characters
```

This is focused artifact validation only. It is not a claim that the complete
repository suite has run on the GitHub head. The stacked draft remains blocked
on exact-head focused and full repository validation.
