# GLM-5.3-Flash MLX 4-bit on one M5 Ultra

This recipe serves [TensorFold/GLM-5.3-Flash-MLX-4bit-MTP](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP)
on one Mac Studio (M5 Ultra, 256 GB) with [TensorFold](https://github.com/ashhart/TensorFold)'s native engine,
release 1.2.5 as published (no patch). The checkpoint is Z.AI's GLM-5.3-Flash: 45 layers of Kimi delta attention
and sparse MLA with 288-expert MoE, affine 4-bit weights in groups of 64, and an MTP head.

You get:
- the full 1,048,576-token window, shared by three streams: one request alone can use all of it, and requests that
  arrive together run side by side instead of queueing;
- MTP drafts, with drafted replies equal to undrafted ones;
- reasoning on by default, at effort High;
- the checkpoint's sampling (temperature 1.0, top_p 0.95) for requests that send none, greedy on request;
- tool calls (`tool_choice: auto`);
- an OpenAI-compatible API on port 8888, a memory guard, a launchd job.

Not in this line, because the released engine has neither for GLM on Metal: image input and structured output
(see [Not supported](#not-supported)). The 1.0.5 line of this recipe carried a patch with an image tower; it is in
this repository's history, not here.

Two settings outside the engine's defaults make a one-Mac load work:
1. `GLM_LOAD_LIMIT_GB=215` (`memory_limit_gb`): the release refuses a load past 70% of RAM (179.2 GB here), which
   drops the MTP head (180.6 GB of weights with it). At 215 the whole model and the 1M-token pool load: 190.9 GiB
   wired at idle.
2. `sudo sysctl iogpu.wired_limit_mb=225000` (`wired_limit_mb`): macOS's GPU wired ceiling. At its default (0,
   about 75% of RAM) the driver's wired collector evicts parts of the model between and inside requests, and the
   same greedy reply runs anywhere from 7 to 107 tok/s. `up` refuses to start under the value and prints the
   command; it resets at reboot ([Steadiness](#steadiness)).

The orchestrator is stdlib-only Python (`python3 -m mac`). Speed and serving checks come from the
[recipe harness](https://github.com/BobClawblaw/recipe-harness) (target `glm53-m5`); the quality suite and decode
ruler are the [Qwen3.8 Spark recipe's](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks).
Every number below has a file under [`evidence/`](evidence/).

## Measured on this Mac

The defaults: three streams sharing the 1,048,576-token window, MTP drafts, thinking on, TensorFold 1.2.5, the
server started by `python3 -m mac up`. The ruler: greedy, thinking off, 256-token replies, a distinct prompt per
phase, tokens a second ([`evidence/2026-11-09-1.2.5/quick/ruler-after.json`](evidence/2026-11-09-1.2.5/quick/ruler-after.json)):

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 89.7 | 101.7 | 95.7 | 93.6 | 0.11-0.14 s |

The same ruler before the serving checks read 89.7 / 62.9 / 96.6 / 92.8: the code phase caught one stall, the kind
described under Steadiness. The suite's `drafts` check (12 prompts, longer replies) reports drafted 54 tok/s
against undrafted 36 ([`evidence/2026-11-09-1.2.5/quick/report.md`](evidence/2026-11-09-1.2.5/quick/report.md)).

Streams together share the engine's decode rounds: a request that arrives during another's reply starts at once.
They share the throughput rather than add to it: one stream's rounds already fill the GPU with MTP drafts. Tokens a
second, 256-token replies, ranges across the four phases
([`evidence/2026-11-09-1.2.5/speed3/ruler-fresh.summary.json`](evidence/2026-11-09-1.2.5/speed3/ruler-fresh.summary.json)):

| Users | each | aggregate | first token |
|---:|---:|---:|---:|
| 1 | 91-103 | 91-103 | 0.11-0.14 s |
| 2 | 58-68 | 116-137 | 0.19-0.23 s |
| 3 | 41-53 | 118-153 | 0.26-0.34 s |

A fourth concurrent request waits for a stream. `parallel = 1` restores one reply at a time.

Prompt passes, cold (the prompt cache resumed none of it), one prompt each, thinking off and on
([`evidence/2026-11-09-1.2.5/prefill/`](evidence/2026-11-09-1.2.5/prefill/)):

| Prompt | Time to first token | Rate | Thinking on |
|---:|---:|---:|---:|
| 2,216 tokens | 1.15 s | 1,929 tok/s | 1.13 s, 1,973 tok/s |
| 8,746 | 6.18 s | 1,415 tok/s | 4.41 s, 1,986 tok/s |
| 35,125 | 20.5 s | 1,712 tok/s | 20.6 s, 1,696 tok/s |
| 139,815 | 86.3 s | 1,621 tok/s | 80.8 s, 1,731 tok/s |

The engine reads prompts in chunks of up to 16,384 tokens. Longer prompts, to the 1M window, were not measured on
this build; the 1.0.5 line found needles at 485k and 927k tokens ([`evidence/2026-10-10-1.0.5-refresh/`](evidence/2026-10-10-1.0.5-refresh/)).

A multi-turn chat's kept prompt state: a 4,776-token conversation resumed 4,652 tokens from the cache, first token
in 0.20 s against 3.9 s read whole, with the same reply
([`evidence/2026-11-09-1.2.5/quick/turns.json`](evidence/2026-11-09-1.2.5/quick/turns.json)).

Memory: 190.9 GiB kept wired between requests, the load limit 215 GB, 27.7 GiB free after the load with the Mac's
usual programs beside it ([`evidence/2026-11-09-1.2.5/status-after-load.txt`](evidence/2026-11-09-1.2.5/status-after-load.txt)).
A smaller `context` takes less: the engine sizes the pool from it.

### tensorfold.dev/benchmarks

The community-v1 protocol (two fixtures, 256 tokens, five repeats at temperature 1 and 0, one stream), run by the
same 1.2.5 build plus TensorFold's board-suite port (`benchmark --suite community-v1`) on this Mac and published
([`evidence/2026-11-09-1.2.5/leaderboard/`](evidence/2026-11-09-1.2.5/leaderboard/)):

| Workload | Temperature | tok/s (median of 5) | first text |
|---|---:|---:|---:|
| code | 1 | 119.2 | 0.076 s |
| chat | 1 | 83.1 | 0.114 s |
| code | 0 | 107.1 | 0.077 s |
| chat | 0 | 88.1 | 0.114 s |

Greedy repeats were within 0.4 tok/s of each other; MTP accepted 2,862 of 4,006 drafted tokens over 2,246 rounds.

## Steadiness

With the GPU wired ceiling raised and the Mac otherwise idle, repeated greedy requests give the same time to within
a few hundredths of a second. Three things made replies stall by 1-2 s in the measurements for this release, each
with the same tokens and draft rounds:

- **macOS's default GPU wired ceiling** (`iogpu.wired_limit_mb` 0): the wired collector evicted the model (181 to
  85 GiB wired, no restart) and the same reply ran at 7, 37, 7, 40, 103, 62 tok/s in a row. Raising the ceiling to
  225,000 MB ended it: 103.8-106.8 tok/s over six repeats. The engine's residency set alone does not prevent it.
- **Other work on the Mac**: another model server's benchmark, a kernel-contract run, a compiler, Time Machine, and a
  days-old browser process each coincided with 1.2 s stalls at random points in replies.
- **The first request after an idle pause**: 1-1.8 s on the first token or inside the reply, once, as the GPU
  re-establishes the model's residency.

`python3 -m mac status` prints the ceiling. Before trusting a measurement, repeat one greedy request a few times.

## Quality

The quality suite has not been run on 1.2.5. The 1.0.5 line's results (TensorFold 1.0.5 plus this recipe's patch of
that time, three streams, the 1M pool, thinking on and off) remain under
[`evidence/2026-10-10-1.0.5-thinking/`](evidence/2026-10-10-1.0.5-thinking/) and
[`evidence/2026-10-10-1.0.5/`](evidence/2026-10-10-1.0.5/) as that build's: IFEval 131 loose / 121 strict thinking on
(96 / 89 off), GSM8K 248 (246), MGSM 226 (217), MMLU 308 (292), tools 57/60, JSON 30/30, long context 54/55 (55/55).
They describe the same weights on an older engine, not this one.

On 1.2.5 the suite's exactness checks pass ([`evidence/2026-11-09-1.2.5/quick/report.md`](evidence/2026-11-09-1.2.5/quick/report.md)):

| Check | Result |
|---|---|
| drafted equals undrafted, 12 prompts | 12/12 |
| four streams together equal each alone | 12/12 |
| soak: 3 chats x 20 turns beside a decoding stream, each turn against the same request alone | 60/60 |
| streaming = non-streaming, multi-turn recall, stop strings, thinking mode, verbatim copy, exact `max_tokens` | 4/4, 8/8, 3/3, 10/10, 2/2, 3/3 |
| seeded sampled replies repeat, and equal their undrafted run | 9/9 |
| a tool call with `tool_choice: auto` | 1/1 |

## Requirements

- A Mac with 256 GB of unified memory: an M5 Ultra here. The model wires about 191 GiB, so nothing else large may
  run beside it. **Stop any other model server first**; memguard stops the load if memory runs short.
- `sudo sysctl iogpu.wired_limit_mb=225000` before `up` (it resets at reboot; `up` refuses without it).
- macOS with Xcode's SDK and Metal toolchain, Zig 0.17.0 (`zig` in `recipe.toml`), Python 3.11+ for the
  orchestrator (stdlib only), and the `hf` CLI for the download (`pip install -U huggingface_hub`; public, no token).
- A checkout of TensorFold (`tensorfold_src`; `build` adds a worktree at the pinned commit) or network access to
  clone it.
- Disk: the checkpoint is 170 GiB, and the engine worktree and build about 2 GB.

## Quick start

```bash
git clone https://github.com/BobClawblaw/glm53-flash-mlx4bit-m5-ultra.git && cd glm53-flash-mlx4bit-m5-ultra
pip install -U huggingface_hub            # the hf CLI that downloads the weights (public, no token)
sudo sysctl iogpu.wired_limit_mb=225000   # the GPU wired ceiling (Steadiness); resets at reboot
python3 -m mac validate                   # the settings and the guards, touches nothing
python3 -m mac up                         # build the engine, download, start with memguard, wait for the API
python3 -m mac status                     # pid, memory, the ceiling, /health, /v1/models, stats
python3 -m mac chat "Say hello."          # one greedy reply
python3 -m mac down
```

The first `up`:
1. adds a worktree of TensorFold at the pinned commit under `~/.cache/tensorfold-glm53/engine/` and refuses one
   that carries changes;
2. builds `tensorfold-native` (about two minutes);
3. downloads the checkpoint at the pinned revision;
4. checks free memory against the engine's floor plus `memory_margin_gib`, and the GPU wired ceiling against
   `wired_limit_mb`;
5. drops the checkpoint's pages from the file cache;
6. starts the server with memguard beside it and waits for `/v1/models`.

Later starts take about 25 s.

To serve on the network, copy `local.example.toml` to `local.toml` (git-ignored), then set `host = "0.0.0.0"`, and
`api_key` too if the network is shared.

Autostart: `launchd/com.tensorfold.glm53-flash.plist` runs `python3 -m mac up --foreground --no-download`. After a
memguard kill, `up` refuses to start for 15 minutes, so launchd's `KeepAlive` can't retry a load into the same wall
(`MEMGUARD_RETRY=1` overrides). The sysctl is not persistent: set it again after a reboot, or the job refuses.

## Settings

`recipe.toml` holds the defaults and `local.toml` (git-ignored) this Mac's values. Any setting can be overridden by
an environment variable of the same name in upper case (`PORT=8890 python3 -m mac up`).

| Setting | Default | Notes |
|---|---|---|
| `host`, `port` | `127.0.0.1`, `8888` | `0.0.0.0` serves the network |
| `profile` | `chat` | `chat`: MTP drafts; `reference`: `--no-drafts`, the exactness reference |
| `context` | 1048576 | the window the streams share, the checkpoint's full one. A smaller window takes less memory, but starves the kept prompt states: at 16,384 three 5.5k-token chats evict each other |
| `parallel` | 3 | streams at once, sharing decode rounds and the window (see Measured). 1 queues requests |
| `thinking` | true | reasoning first, in `reasoning_content`. A request sends `chat_template_kwargs: {"enable_thinking": false}` to skip it |
| `reasoning_effort` | `high` | GLM-5.3's levels are `low`, `high` and `max`. A request's `"reasoning_effort"` overrides it; empty leaves the template's default, `max` |
| `temperature` | empty | the server's default for requests that send none: empty is the checkpoint's (1.0, top_p 0.95); `"0"` greedy. A request's own `temperature`, `top_p`, `top_k` and `seed` are honoured |
| `max_tokens` | 32768 | the reply cap when a request sets none. With thinking on, reasoning counts against it |
| `prompt_cache_gib` | 4 | kept prompt states, bounded so the load stays predictable |
| `vision` | false | refused when true: no `--vision` for GLM on Metal in 1.2.5 |
| `memory_limit_gb` | 215 | `GLM_LOAD_LIMIT_GB`, the engine's one-Mac load limit (unset: 70% of RAM, which drops the MTP head) |
| `residency` | true | the engine's residency set over the weights, kept wired between requests; false sets `GLM_RESIDENCY=0` |
| `wired_limit_mb` | 225000 | `up` refuses when `sysctl iogpu.wired_limit_mb` is under it; 0 skips the check |
| `memguard*` | 6 GiB free, 12 GiB swap growth, 24 GiB compressor growth | SIGKILL after two readings 2 s apart; 30 s at critical pressure |
| `memory_margin_gib` | 8 | `up` refuses with less free than the engine's floor plus this |
| `keep_warm`, `dashboard`, `learn*`, `compact_at` | 900, true, false | the engine's flags |
| `request_log` | empty | a file for each request and reply |
| `api_key`, `extra_args`, `extra_env` | | flags and variables the recipe owns are refused |
| `patch`, `patch_sha256` | empty | a patch under `patches/` applied to the worktree before the build, pinned by sha256; empty builds the release |

`python3 -m mac validate` refuses these before anything runs:
- `vision = true`;
- a context outside [256, 1,048,576] or not a multiple of 256;
- out-of-range limits, and zero-padded or non-decimal integers;
- a patch whose sha256 isn't the pinned one, or a sha256 with no patch;
- `extra_args` that re-set a flag the recipe builds (any prefix of them), or a flag for another platform or engine
  (`--vision*`, `--speed-up`, CUDA ranks);
- `extra_env` that sets `GLM_LOAD_LIMIT_GB` or `GLM_RESIDENCY`.

## Memory safety

The constraint this recipe is built around: **don't run the Mac out of memory.**

- **Before a load**, `up` refuses when free memory is under the engine's floor (171 GiB) plus the margin, and when
  the GPU wired ceiling is under `wired_limit_mb`. It also drops the checkpoint's pages from the file cache with
  `msync(MS_INVALIDATE)`, which needs no root. A cached copy beside the GPU copy once pushed a load to critical
  pressure.
- **During a load and after**, `mac/memguard.py` reads `vm_stat`, `vm.swapusage` and
  `kern.memorystatus_vm_pressure_level` every 2 s. It SIGKILLs the server after two readings of any of these:
  - free + inactive memory under 6 GiB;
  - swap grown 12 GiB;
  - the compressor grown 24 GiB;
  - or 30 s at critical pressure.

  With the model wired, any compressor growth means other memory is being squeezed.
- **After a kill**, `up` refuses for 15 minutes.
- **Two models don't fit.** A 47 GB model server beside this one made the compressor take over during a restart;
  memguard stopped that load at 16 GiB free, with nothing else harmed.

## Quality suite and ruler

```bash
python3 -m mac quality --label my-run      # all of the below into evidence/<date>-my-run/{suite,ruler}
python3 -m mac suite -- --only gsm8k tools --out evidence/my-run   # a subset (arguments after -- pass through)
python3 -m mac ruler -- --runs 3 --max-tokens 256
```

`quality` is the standard quality measurement; commit the folder it writes. It caps long-context documents at
`context` nominal tokens, runs HumanEval only with `--humaneval` and Docker, and skips `vision`. The suite and the
ruler come from the Qwen3.8 Spark recipe at a pinned commit (`spark_recipe_sha`), cloned under `.external/` and run
there. Its checks:

| Check | What it measures |
|---|---|
| `drafts` | drafted replies equal `"draft": false` ones, 12 prompts (MTP drafts must not change greedy text) |
| `concurrent` | four streams together give the same replies as each alone |
| `behaviour` | streamed equals non-streamed, facts recalled over six turns, stop strings, thinking mode (`reasoning_content`, no tag leaks), verbatim copy, exact `max_tokens` |
| `tools` | 60 tool calls: exact function and argument set, optional arguments left out unless asked, parallel calls, no call where none is wanted |
| `json` | 30 JSON-schema prompts checked by the suite's validator |
| `ifeval` | 150 IFEval prompts of the 24 instruction types implemented, strict and loose |
| `gsm8k` | 250 grade-school math problems, exact number |
| `mgsm` | the same kind of problems in 8 languages (de, es, fr, ja, zh, ru, sw, bn), 30 each |
| `mmlu` | 6 questions from each of the 57 MMLU subjects (342), a fixed sample, exact letter |
| `humaneval` | 164 Python functions, pass@1, in a container with no network |
| `repetition` | repeated 4-grams and early stops in eight 600-word replies |
| `long-context` | needles at 10/50/90% depth and at the edges of long documents; ten spread facts recalled together |
| `soak` | three 20-turn chats at once beside a decoding stream, each turn against the same request alone |
| `snapshot` | the reply set, for comparing two servers (`--compare a.json b.json`) |

The ruler runs four phases (prose, code, structured JSON, a long list), each with its own prompt, and reports the
per-stream decode rate and the time to the first content token.

## Not supported

- **Image input.** TensorFold 1.2.5 has no `--vision` for GLM on Metal; an `image_url` part answers 400
  ([`evidence/2026-11-09-1.2.5/quick/vision-400.md`](evidence/2026-11-09-1.2.5/quick/vision-400.md)).
- **Structured output** (`response_format` json_object/json_schema, `guided_*`, regex) and **`tool_choice`
  `required` or a named function**: 400, "backend 'glm-zig' cannot enforce ... it has no token grammar support"
  ([`evidence/2026-11-09-1.2.5/quick/structured-400.json`](evidence/2026-11-09-1.2.5/quick/structured-400.json)).
  `tool_choice: auto` works.
- Remote image URLs, video and audio.
- More than three streams at a time by default (`parallel` raises it).
- The two-Mac `--speed-up` pair: this recipe is one Mac.
- No `n > 1`, no logprobs. The reasoning field is `reasoning_content`.

## Logs and evidence

`python3 -m mac logs`, `python3 -m mac status`. The engine prints one `done` line per request (prompt and cached
tokens, reply tokens, drafts accepted, the reply's sha). memguard logs to `~/.cache/tensorfold-glm53/memguard.log`.
Every number in this README has a file under [`evidence/`](evidence/), written by the recipe harness, the suite or
the ruler.

Upstream TensorFold pull requests from the 1.0.5 line of this work: the one-Mac load limit
([#580](https://github.com/ashhart/TensorFold/pull/580)), wired loading
([#581](https://github.com/ashhart/TensorFold/pull/581)), and the early refusal of sampled requests on greedy-only
engines ([#582](https://github.com/ashhart/TensorFold/pull/582)).

## License

MIT ([LICENSE](LICENSE)). TensorFold is Apache-2.0 ([NOTICE](NOTICE)).
