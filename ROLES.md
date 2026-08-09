# Roles & Communication Protocol

Four parties collaborate on this project. **Every cross-party message MUST begin with an explicit
`FROM:` / `TO:` header** (see *Message format*). No exceptions — it is how we keep the two-oracle
workbench from crossing wires (a message from the implementer must never be mistaken for an oracle
reply, and vice versa).

## Roles

### `mickg10` — the human
- Owns the product objective and release trade-offs; provides data (e.g. the Cantolopera purchase) and sets priorities.
- **Relays** messages between `mickg10implementer` and `mickg10bigoracle` (the big oracle is human-relayed).
- Final authority on scope and spend.

### `mickg10implementer` — Claude Code (Opus 4.8)
- **Leads the build and delivery**: writes code, runs GPU agents, stages data, ships instrumentals.
- Asks the two oracles explicit, bounded questions; folds their guidance in; reports honestly against gates.
- Owns execution and the repo. Does **not** scrape or credential-login to paid services — processes only user-provided data.

### `mickg10localoracle` — gpt56 (GPT-5.6)
- Second **independent** AI advisor with read access to the branch (works in a worktree on `gpt56mailbox`).
- Answers research questions inline in `local_oracle.md`; **may commit code**.
- Fresh, fast, code-capable eyes; critiques and disagrees where warranted.

### `mickg10bigoracle` — the GitHub issue #1 oracle (senior reviewer / driver)
- Holds the strategic **gates**: (1) label/split integrity, (2) judge validation, (3) end-to-end selection, (4) release/delivery semantics.
- Answers on **GitHub issue #1**: one verdict + ≤3 release-blocking findings + one next experiment per gate.
- Now also asked to **actively drive** the plan, not only review.

## Channels

| between | medium |
|---|---|
| `mickg10implementer` ↔ `mickg10bigoracle` | GitHub **issue #1** comments (human-relayed both directions) |
| `mickg10implementer` ↔ `mickg10localoracle` | **`gpt56mailbox`** branch, `local_oracle.md` (git commits) |
| `mickg10` ↔ everyone | the Claude Code chat (+ relaying to the big oracle) |

## Message format — MANDATORY

Every cross-party message begins, on its own lines, at the **very top**:

```
FROM: <role>
TO: <role>
```

- Roles are exactly one of: `mickg10` · `mickg10implementer` · `mickg10localoracle` · `mickg10bigoracle`.
- Address both oracles with `TO: mickg10bigoracle, mickg10localoracle`.
- Recommended optional lines: `RE: <topic>` and `DATE: <YYYY-MM-DD>`.

**Example**

```
FROM: mickg10implementer
TO: mickg10bigoracle
RE: separator fine-tune plan
DATE: 2026-08-08

<body>
```

A reply is not valid until it carries its own `FROM:`/`TO:` header. If a message arrives without one,
the recipient asks for it to be re-sent with the header rather than guessing the sender.
