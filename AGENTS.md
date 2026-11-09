# Notes for agents working on this recipe

GLM-5.3-Flash MLX 4-bit on one 256 GB M5 Ultra, served by TensorFold's native engine as released (v1.2.5,
`8be9c408`, no patch). See README.md for use, `evidence/` for measurements. Published (see below).

## State (2026-11-09)

- Done:
  - Moved from 1.0.5 + patch to 1.2.5 unpatched. The patch's pool, wiring and greedy fallback have stock equivalents
    (the engine pools the window for its streams, holds a residency set, samples); its vision tower has none, so
    image input is unsupported (README, Not supported). `vision = true` is refused.
  - Defaults: port 8888, the full 1,048,576-token window shared by three streams, limit 215 GB
    (`GLM_LOAD_LIMIT_GB`, or the MTP head is dropped), thinking on at effort High, the checkpoint's sampling.
  - `up` refuses when `sysctl iogpu.wired_limit_mb` is under `wired_limit_mb` (225000): at the default the model is
    evicted mid-request and decode stalls (README, Steadiness).
  - Measured on 1.2.5 with the recipe harness (github.com/BobClawblaw/recipe-harness, target `glm53-m5`):
    `evidence/2026-11-09-1.2.5/`. Leaderboard entry on tensorfold.dev/benchmarks from the same build plus the
    board-suite port (community-v1).
- Open:
  - Published: https://github.com/BobClawblaw/glm53-flash-mlx4bit-m5-ultra (public).
  - The quality suite has not run on 1.2.5 (the user says when); the 1.0.5 line's receipts stay under
    `evidence/2026-10-10-*` and the README says which build they measured.
  - Structured output and `tool_choice` required/named: 400, the Metal GLM backend has no token grammar support.
  - Upstream from the 1.0.5 line: PRs ashhart/TensorFold#580 (load limit), #581 (wiring), #582 (greedy refusal);
    a #565 comment (GLM Metal vision); issues #583, #584.
- The server may be up on 0.0.0.0:8888 (`local.toml`, git-ignored, also turns on the request log).

## Measure before you claim

Every number in the README has a file under `evidence/`. Speed and serving checks go through the recipe harness
(`python3 -m harness run glm53-m5 --plan quick|speed3|prefill`); `python3 -m mac quality --label <what changed>` is
the quality suite and the ruler, run only when the user asks for the quality suite. Commit the evidence folder, then
update the README tables from it. A number with no file under `evidence/` stays out of the README.

## Rules learned the hard way

- **Never load while another model holds memory.** GLM wires ~191 GiB. Check for other `tensorfold-native`
  processes first and never kill someone else's server: ask the user.
- The user's hard constraint: do not OOM the Mac. An unwired load once made jetsam kill ~300 system daemons.
  Keep `residency = true` and memguard on. `up` refuses for 15 min after a memguard kill (`MEMGUARD_RETRY=1`
  overrides; use it only once the cause is gone).
- **A measurement needs a quiet Mac and the wired ceiling raised.** Other sessions' GPU jobs, Time Machine and a
  long-lived Chrome each produced 1-1.5 s stalls inside replies; the default `iogpu.wired_limit_mb` produced the
  same greedy reply at 7 and at 107 tok/s. Check `python3 -m mac status` (it prints the ceiling), `pgrep -fl
  [t]ensorfold-native`, `tmutil status`, and repeat a greedy request a few times before trusting a run.
- `pgrep -f` patterns appear in the watcher's own command line: use `[t]ensorfold-native`.
- A small window starves the kept prompt states: at `--context 16384` three 5.5k-token chats evict each other and
  the suite's soak fails on resumes. Keep the full window unless memory forces less.
- The user wants everything in the native engine, not the Python 0.6.6 path.

## Changing the engine

The worktree is `~/.cache/tensorfold-glm53/engine/<sha>/`, built unpatched. To move to a new release: set
`tensorfold_sha` in recipe.toml, `python3 -m mac build`, `python3 -m mac up`, then the harness `quick`, `speed3`
and `prefill` plans against it and a new `evidence/<date>-<version>/`. A patch, if one is ever needed again, goes
under `patches/` pinned by `patch` and `patch_sha256` (never edited in place), and `build` applies it to a clean
worktree.
