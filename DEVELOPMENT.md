# NEXUS — Parallel Development Guide

**Version:** 0.3.0 | **Baseline:** M2C-FROZEN | **Tag:** `m2c-frozen`

---

## Branch Responsibilities

| Branch | Agent | Purpose |
|---|---|---|
| `main` | — | Stable baseline. Frozen M2C analytical state. |
| `product-layer` | Antigravity / Gemini | Product Layer implementation |
| `research-m2d` | Claude Code | M2D analytical / model research |

---

## Rules

1. **`main` is the stable branch.** Never push broken code directly to main.
2. **`product-layer` is for Antigravity/Gemini only.** Do not push research work here.
3. **`research-m2d` is for Claude Code only.** Do not push product work here.
4. **Never let both agents edit the same worktree.** Each agent has its own physical directory (`NEXUS-product/` or `NEXUS-research/`).
5. **Commit frequently.** Small, logical commits are preferred over large batches.
6. **Push logical milestones.** Push when a coherent unit of work is complete.
7. **Never force-push shared history.** `git push --force` is prohibited on `main`, `product-layer`, and `research-m2d`.
8. **Do not merge research changes directly into production without review.** All merges to `main` require explicit review.
9. **`m2c-frozen` is the analytical recovery point.** If analytical behavior regresses, reset to this tag.
10. **Frozen M2C analytical behavior must not be silently changed.** Any change to `src/` analytical logic requires explicit documentation and approval.

---

## Worktrees

```
<parent>/
    NEXUS/          ← main (stable baseline)
    NEXUS-product/  ← product-layer branch (Antigravity workspace)
    NEXUS-research/ ← research-m2d branch (Claude Code workspace)
```

To check worktree status:
```bash
git worktree list
```

---

## GitHub Remote

**Repository:** `https://github.com/Dr-Dre420/NEXUS`

---

## Version Metadata

See [`VERSION`](./VERSION) for:
- `NEXUS_VERSION` — current semantic version
- `ANALYTICS_VERSION` — analytical milestone identifier
- `DEMO_WORLD_SEED` — reproducibility seed for the demo world

---

## Next Steps

### Antigravity / Gemini (Product Layer)
- Work exclusively in `NEXUS-product/` worktree
- Branch: `product-layer`
- Do not touch `src/models/`, `src/mechanics.py`, `src/features*.py` analytical logic

### Claude Code (M2D Research)
- Work exclusively in `NEXUS-research/` worktree
- Branch: `research-m2d`
- All research findings must be documented before merging into main
