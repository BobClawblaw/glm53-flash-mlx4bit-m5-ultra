# Notes for agents working on this recipe

GLM-5.3-Flash MLX 4-bit on one 256 GB M5 Ultra, served by TensorFold's native engine (v1.0.4, `d4fc1964`) with
`patches/glm-one-mac-vision-v1.0.4.patch`. See README.md for use, `evidence/` for measurements. Nothing is
committed yet.

## State (2026-10-09)

- Done:
  - Native `--vision` (tower, preprocessing, server), wired loading, memguard, greedy fallback for clients that send
    a temperature.
  - Defaults: port 8888, thinking on.
  - Engine on v1.0.4: 119/119 build steps, 773/775 tests, golden suite unchanged.
  - Vision checked against mlx-vlm (`evidence/2026-10-09-native-vision-1.0.3.md`).
  - Quality suite and decode ruler complete (`evidence/2026-10-09-quality.md`). The user's hermes agent works
    against it (33 requests, all 200).
  - Publishing files: NOTICE, LICENSES/Apache-2.0.txt, CHANGELOG.md, `.github/workflows/ci.yml`.
- Open:
  - Not committed and no GitHub remote: the user decides whether and where to publish.
  - Possible upstream contributions, none posted yet (the user picks; post as BobClawblaw, strip paths and names):
    - ashhart/TensorFold #498: the GLM one-Mac load limit, with the wiring the higher limit needs.
    - #565 (ours): native image input; a comment with the GLM Metal tower and its interface comes first.
    - Greedy-only engines fail a sampled request with a 500 after the whole prefill.
    - The loader leaves checkpoint pages cached by an earlier download.
    - The prompt cache keeps no single-chunk prompt.
  - `response_format` returns 400: the native engine has no structured output.
  - IFEval is weak with thinking off (65%): it is the model, not the serving.
  - The image-aware prompt-cache key is not exercised live.
- The server may be up on 0.0.0.0:8888 (`local.toml`, git-ignored, also turns on the request log).

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
(apply chain) and build.zig (frameworks).
