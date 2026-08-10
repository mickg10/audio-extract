# Validation record — D0/R0 grouped comparison v1

Separate branch:

```text
bigoracle/d0-r0-grouped-comparison-artifacts-20260810
```

Base commit:

```text
2254c8e1bb797c3cedaa765d00950e0e3d496aa2
```

The implementation and tests were exercised in an isolated package containing
the repository's complete-route-evaluation and group-identity contracts.

Command:

```bash
python -m pytest -q
```

Focused result before GitHub upload:

```text
..............                                                           [100%]
14 passed in 0.16s
```

Additional local checks:

```text
python -m py_compile
no source or test line longer than 88 characters
```

This is focused artifact validation. It is not a claim that the complete
repository suite has run against the GitHub head. The stacked draft PR must
receive exact-head repository validation before integration.
