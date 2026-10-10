# Notes for agents working on this recipe

GLM-5.3-Flash MLX 4-bit on one 256 GB M5 Ultra, served by TensorFold's native engine (v1.0.5, `bec00ae6`) with
`patches/glm-one-mac-vision-pool-v1.0.5.patch`. See README.md for use, `evidence/` for measurements. Published (see
below).

## State (2026-10-10)

- Done:
  - Native `--vision` (tower, preprocessing, server), wired loading, memguard, greedy fallback for clients that send
    a temperature.
  - Defaults: port 8888, the full 1,048,576-token window shared by three streams (the patch's MLA pool in
    families/glm/slots.zig: a request takes prompt + max_tokens from it while it runs, admission waits in
    lane_host.zig when it does not fit), limit 215 GB, thinking on at effort High. Needles found to 927k tokens.
  - Engine on v1.0.5: `zig build test` passes, `zig build test-golden` 309/319 with 0 unexpected differences. The
    patch rewrites upstream's two greedy-refusal tests (stream_preflight_test.zig) for the greedy fallback. Pooled,
    1.0.5's resident prompt states (MLA prefixes left in their slot) hold their blocks after their stream leaves
    (`Snap.held`, slots.zig `holdPrefixes`/`evictHeld`); `tools/pool_reuse.py` checks it.
  - Vision checked against mlx-vlm (`evidence/2026-10-09-native-vision-1.0.3.md`).
  - Quality suite and decode ruler complete on 1.0.5, thinking on (`evidence/2026-10-10-1.0.5-thinking/`) and off
    (`evidence/2026-10-10-1.0.5/`); `python3 -m mac quality [--thinking 32768] [--resume]` reruns them. The user's hermes agent works
    against it (33 requests, all 200).
  - Publishing files: NOTICE, LICENSES/Apache-2.0.txt, CHANGELOG.md, `.github/workflows/ci.yml`.
- Open:
  - Published: https://github.com/BobClawblaw/glm53-flash-mlx4bit-m5-ultra (public).
  - Upstream contributions (as BobClawblaw, paths and names stripped); follow up on review:
    - Posted 2026-10-10: PRs ashhart/TensorFold#580 (load limit), #581 (wiring), #582 (greedy refusal);
      a #565 comment (GLM Metal vision); issues #583 (loader page cache), #584 (prompt-cache question).
  - `response_format` returns 400: the native engine has no structured output.
  - With thinking on, one 200k needle is missed (98% depth on 1.0.4, 50% on 1.0.5); thinking off finds every one.
  - A `powermetrics` GPU sampler from another session ran during the 1.0.5 runs; two early rulers had one slow phase
    each (cause not found), later ones agree.
  - Candidate upstream PR: the shared pool (with the held prefixes), against main.
  - The image-aware prompt-cache key is not exercised live.
- The server may be up on 0.0.0.0:8888 (`local.toml`, git-ignored, also turns on the request log).

## Measure before you claim

Every recipe ships the quality suite and the decode ruler, and the README carries both, as detailed as the Qwen3.8
Spark recipe's (its structure, not its numbers: a recipe reports only its own model): a measured decode table, a prompt-pass table, the quality table with what each check measures, and
the suite how-to. After any engine, patch or settings change, run `python3 -m mac quality --label <what changed>`.
Commit the folder it writes under `evidence/`, then update the README tables from it. A number with no file under
`evidence/` stays out of the README.

## Rules learned the hard way

- **Never load while another model holds memory.** GLM wires ~185 GiB. Check for other `tensorfold-native`
  processes first (another agent runs Qwen3.8-27B from ~/github/TensorFold on port 18392) and never kill someone
  else's server: ask the user.
- The user's hard constraint: do not OOM the Mac. An unwired load once made jetsam kill ~300 system daemons.
  Keep `residency = true` (GLM_RESIDENCY + GLM_WIRE) and memguard on. `up` refuses for 15 min after a memguard kill
  (`MEMGUARD_RETRY=1` overrides; use it only once the cause is gone).
- `pgrep -f` patterns appear in the watcher's own command line: use `[t]ensorfold-native`.
- The native GLM engine is greedy-only; the patch serves any temperature greedily.
- The user wants everything in the native engine, not the Python 0.6.6 path. The 0.6.6 + mlx-vlm venv
  (~/.cache/tensorfold-glm53/python/venv) is only a numeric reference for `tools/vision_reference.py` and
  `tools/vision_noise.py`.

## Changing the engine

The worktree is `~/.cache/tensorfold-glm53/engine/<sha>/`, where `.recipe-patch` holds the pinned sha256. After
editing it:

1. `zig build native -Dcpu=apple_m1`, then `zig build test`, then `zig build test-golden`.
2. `git add -N . && git diff HEAD -- . ':!.recipe-patch' > patches/<new name>.patch && git reset`.
3. Re-pin `patch` and `patch_sha256` in recipe.toml, and write the new sha into `.recipe-patch`.

Never edit a pinned patch in place. To move to a new TensorFold release, `git apply --3way` the patch onto a fresh
worktree at the tag. 1.0.2 to 1.0.3 conflicted in engine_api.zig (Info), lane_host.zig (keep/lookup args), cli.zig
(apply chain) and build.zig (frameworks). 1.0.4 to 1.0.5 conflicted in engine_api.zig and stream.zig (logprobs beside
images), lane_host.zig (lookupRewind), slots.zig (the ring), engine.zig (the load plan moved to load_plan.zig: drop our
limit), glm_host.zig and metal.zig (open's learn floor), server.zig (keep the greedy fallback).
