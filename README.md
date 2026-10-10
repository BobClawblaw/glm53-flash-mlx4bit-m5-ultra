# GLM-5.3-Flash MLX 4-bit on one M5 Ultra, with native vision

This recipe serves [TensorFold/GLM-5.3-Flash-MLX-4bit-MTP](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP)
on one Mac Studio (M5 Ultra, 256 GB) with [TensorFold](https://github.com/ashhart/TensorFold)'s native engine. That
checkpoint is Z.AI's GLM-5.3-Flash: 45 layers of Kimi delta attention and sparse MLA with 288-expert MoE, affine
4-bit weights in groups of 64, an MTP head, and the vision tower.

You get:
- the full 1,048,576-token window, shared by three streams: one request alone can use all of it, and requests that
  arrive together run side by side instead of queueing;
- MTP drafts;
- reasoning on by default, at effort High;
- image input, run natively: the tower on Metal, decoding and preprocessing in the engine;
- tool calls;
- an OpenAI-compatible API on port 8888;
- a memory guard;
- a launchd job.

Stock TensorFold 1.0.4 refuses this checkpoint on one 256 GB Mac and serves GLM text only.
[`patches/glm-one-mac-vision-pool-v1.0.4.patch`](patches/), pinned by sha256, changes five things:

1. **The one-Mac load limit.** GLM's native engine allows 70% of RAM, 179.2 GB here. The language model alone is
   180.6 GB. `TENSORFOLD_MEMORY_LIMIT_GB` now sets the limit, and the recipe uses 215.
2. **Wired loading.** Past about 110 GB of freshly written buffers, the GPU driver's wired collector handed the
   buffers no command had touched yet to the VM compressor: 145 GiB at the peak, with pressure still normal. One
   unwired load ended with jetsam killing about 300 system daemons, then the server. With `GLM_RESIDENCY` +
   `GLM_WIRE`, every buffer is resident and wired (mlock) before the reads fill it. The full model then loads in 22 s
   with the compressor flat.
3. **Native image input** (`--vision`):
   - the checkpoint's 24-block tower on Metal;
   - images decoded through ImageIO;
   - GLM5-Next's image processor ported exactly, including PIL's 8-bit bicubic resampler;
   - `image_url` parts expanded to each image's tokens;
   - the tower's rows written over the placeholders' embeddings during the prompt pass.

   PNG patches equal TensorFold 0.6.6 + mlx-vlm's bit for bit. The tower's rows are as close to an fp32 run as
   mlx-vlm's own bf16 rows.
4. **Greedy fallback.** The native GLM engine decodes greedily only. A request that sends a temperature, as agent
   clients do, is served greedily instead of failing with a 500 after its prefill.
5. **A shared cache pool.** Stock, every stream gets its own cache for the whole window when the server starts, so
   three streams at 1M would need ~65 GB of caches: more than this Mac has beside the model. The patch keeps one
   1M-token pool, the engine's own caches, and gives each request a slice of it for its prompt plus `max_tokens` while
   it runs. One request alone can use the full window; several split it. A request that does not fit beside the
   running ones waits in line until they finish. Each extra stream costs only its fixed-size state, ~0.3 GB.

The orchestrator is stdlib-only Python (`python3 -m mac`). The quality suite and decode ruler are the
[Qwen3.8 Spark recipe's](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks), cloned at a
pinned commit and run against this server. Every number below has a file under [`evidence/`](evidence/).

## Measured on this Mac

The defaults: one stream, MTP drafts, vision on, model wired, measured on 131,072 and 1,048,576-token windows (decode does not change with the window). These were measured with
`bench/ruler.py`: greedy, thinking off, 256-token replies, a distinct prompt per phase, median of 3 runs. Receipts:
[`evidence/2026-10-09-ruler-1.0.4/`](evidence/2026-10-09-ruler-1.0.4/).

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 93.3 | 104.6 | 104.2 | 89.7 | 0.23-0.29 s |

Tokens a second. Undrafted, one stream decodes at 63 tok/s and drafted at 81, on the suite's `drafts` prompts.

Three streams by default (`parallel = 3`): concurrent requests share the engine's decode rounds, so a request that
arrives during another's long reply starts at once instead of queueing behind it. They share the throughput rather
than add to it: one stream's rounds already fill the GPU with MTP drafts, so a second stream mostly takes the draft
rows. Aggregate tokens a second, 2 runs, 256-token replies, measured with three streams on a 262k window before the
pool (the pool changes where the caches live, not the decode)
([`evidence/2026-10-10-parallel3/ruler/`](evidence/2026-10-10-parallel3/ruler/)):

| Users | each | aggregate | first token |
|---:|---:|---:|---:|
| 1 | 55-106 | 55-106 | 0.23-0.69 s |
| 2 | 46-55 | 93-110 | 0.39-0.68 s |
| 3 | 33-40 | 102-123 | 0.54-1.52 s |

(Ranges across the prose / code / structured / list phases; the list phase varies most between runs. A lone request
decodes as fast as with `parallel = 1`.) A fourth concurrent request waits for a stream. `parallel = 1` restores
one reply at a time.

Prompt passes, one prompt each, prompt cache off, with the needle found in every one. Up to 102k they come from the
long-context check's own timings
([`evidence/2026-10-09-quality-1.0.4-long128/`](evidence/2026-10-09-quality-1.0.4-long128/)); beyond that, from the
suite's needle function on the 1,048,576-token profile
([`evidence/2026-10-09-long-1m/`](evidence/2026-10-09-long-1m/)):

| Prompt | Time to first token | Rate |
|---:|---:|---:|
| 6,390 tokens | 5.7-6.9 s | 932-1,129 tok/s |
| 25,435 | 22.4 s | 1,134 tok/s |
| 50,981 | 48.6 s | 1,049 tok/s |
| 102,081 | 111.2 s | 918 tok/s |
| 200,465 | 248 s | 810 tok/s |
| 484,568 | 880 s | 551 tok/s |
| 926,599 | 2,533 s | 366 tok/s |

Images ([`evidence/2026-10-09-native-vision-1.0.3.md`](evidence/2026-10-09-native-vision-1.0.3.md)):

| Image | Tokens | Preprocess | Tower | Request to first token |
|---|---:|---:|---:|---:|
| 96x96 logo | 16 | 1.6 ms | 8 ms | 0.5 s |
| 517x389 drawing | 266 | 3.4 ms | 26 ms | 2.1 s |
| 2000x1500 photo-like, 1,024-token budget | 972 | 43 ms | 154 ms | |
| 2000x1500, the full 4,096-token budget | 3,909 prompt tokens | | | 6.4 s |

Memory, with the model and tower wired. GLM's caches are small: most of its layers keep constant-size state.

| window | caches | weights + caches | wired | free after load |
|---:|---:|---:|---:|---:|
| 131,072 | 6.3 GB | 186.9 GB | 183 GiB | ~38 GiB |
| 262,144 | 8.9 GB | 189.5 GB | | |
| 524,288 | 14.0 GB | 194.6 GB | | |
| 1,048,576 (default) | 24.2 GB | 204.8 GB | 202 GiB | ~27 GiB |

Measured with one stream, before the shared pool. The default now serves three streams from one 1,048,576-token
pool in the same memory as the old one-stream 1M load: 200-202 GiB wired, ~27 GiB free, load limit 215 GB
([`evidence/2026-10-10-pool/`](evidence/2026-10-10-pool/)).

The pool was checked three ways ([`evidence/2026-10-10-pool/`](evidence/2026-10-10-pool/), `tools/pool_*.py`):
- the suite's `concurrent` check: four streams together give the same replies as each alone, 12/12;
- 7k- and 10k-token prompts (the sparse attention path) running beside a 24k-token one, in slices at offsets in
  the pool, gave byte-identical replies to each alone;
- a request needing more room than was free waited for the running one to finish, then answered; a request larger
  than the window was refused with a 400.

The load takes 21.7-24 s from a warm disk (80 s unwired), with the compressor and pressure flat. Memory stayed flat
through the 485k-token prompt pass. The tower is 0.9 GiB, plus 1.2 GiB of scratch for 4,096 image tokens.

## Quality

The Spark recipe's suite on the defaults, greedy, datasets pinned. Each check ran twice: with thinking **on** at effort
High (the recipe's default; `python3 -m mac quality --thinking 32768`, up to 32,768 reasoning tokens a request) and
with thinking **off** (as the suite sends on its own). Receipts:
[`evidence/2026-10-10-thinking-on/`](evidence/2026-10-10-thinking-on/) and
[`evidence/2026-10-09-quality.md`](evidence/2026-10-09-quality.md).

These runs predate the shared pool: one stream, on a 262k (thinking on) and 131k (thinking off) window. The
thinking-on suite is being rerun on the pooled 1M default; so far drafts 12/12 and four streams together equal
alone 12/12, as before. This table is updated from that run when it finishes.

| Check | Thinking on (default) | Thinking off |
|---|---|---|
| drafted equals undrafted; four streams together equal alone | 12/12; 12/12 | 12/12; 12/12 |
| streaming = non-streaming, multi-turn recall, stop strings, thinking mode, verbatim copy | all pass | all pass |
| exact `max_tokens` | 0/3 (see below) | 3/3 |
| tool calls (60: exact function and arguments, optional arguments left out, parallel calls, no-call cases) | 58/60 | 57/60 |
| JSON schemas | 30/30 | 30/30 |
| IFEval (150 prompts, the 24 instruction types implemented) | 127 loose / 117 strict | 98 / 92 |
| GSM8K (250) | 248 (99.2%) | 245 (98.0%) |
| MGSM (8 languages x 30) | 221 (92.1%) | 220 (91.7%) |
| MMLU (6 a subject, 342) | 307 (89.8%) | 288 (84.2%) |
| HumanEval (164, pass@1, sandboxed) | not run (needs the suite's Docker sandbox) | not run |
| repetition (eight 600-word replies) | 8/8, 0.18% repeated 4-grams | 8/8, 0.06% |
| long context: needles at 10/50/90% of 6-51k and 2/50/98% of 102k and 200k; ten spread facts at 25k-200k | 54/55 (to 200k, 262k window) | 42/42 (to 102k, 131k window) |
| vision (12 drawn probes) | 12/12 | 12/12 |

Notes on the results:

- **Thinking pays where it should.** IFEval rises from 98 to 127, MMLU from 288 to 307, GSM8K from 245 to 248. The
  replies with thinking off were well formed but broke the instruction; with reasoning the model checks itself.
- **Exact `max_tokens`** fails with thinking on because of how the run is set up, not the server. To leave room for
  reasoning, the wrapper adds the 32,768-token budget to every reply limit, so the model finishes well under the
  raised limit. The thinking-off run (3/3) is the test of the limit.
- **Long context.** The one miss is a 200k-token document with the needle 98% of the way in. With thinking on, the
  model answered that the document held no secret. With thinking off, the same document gave the code at 98% and 95%
  depth ([`needle-200k-098-thinking-off.txt`](evidence/2026-10-10-thinking-on/needle-200k-098-thinking-off.txt)).
  For plain retrieval from a long document, send `"chat_template_kwargs": {"enable_thinking": false}`. The suite
  sizes documents nominally (8k-250k) and GLM's tokenizer makes them about 20% shorter, so "250k" is about 200k
  tokens. Beyond the suite, needles were found at 485k and 927k tokens on the 1M profile (see Measured).
- **Tools.** The misses are the checker's exact match. The model picked the right function with the right core
  values, then added an optional field (`label` on a timer) or a fuller place name ("louvre museum, paris").
- **Decode is the same either way.** The ruler on the thinking-on defaults (262k window, effort High) measured
  92.9 / 106.0 / 104.0 / 90.1 tok/s.
- **Serving checks beyond the suite.** Tool calls with `temperature` 0.6 (as agents send), thinking off and on,
  streamed and not, all returned valid `tool_calls`. Parallel calls with a `tool`-result round trip answered
  correctly. An agent (hermes) on another machine sent 33 requests, all 200.

### Vision against the reference

`zig-out/bin/tf-glm-vision` runs the tower on an image or a reference dump and compares each stage with
`tools/vision_reference.py`'s output: mlx-vlm's tower, which TensorFold 0.6.6's `--vision` serves.

| Image | Patches differing | Cosine to mlx-vlm (mean / worst row) |
|---|---|---|
| 517x389 drawing (pad only) | 0 of 1,251,264 | 0.9835 / 0.695 |
| 2000x1500 at 1,024 tokens (antialiased downscale) | 0 of 4,572,288 | 0.9935 / 0.722 |
| 96x96 RGBA logo (upscale, alpha over white) | 0 of 75,264 | 0.9984 / 0.992 |
| palette PNG with a transparent entry; gray+alpha PNG | 0; 0 | 0.983; 0.991 |
| JPEG with EXIF orientation 6 | 4% (ImageIO's decoder against libjpeg-turbo) | 0.982 |

A cosine of 0.98 is bf16 noise, not error. Against an fp32 run of the same tower (`tools/vision_noise.py`), the
native rows score 0.98720 on the drawing and mlx-vlm's bf16 rows 0.98718. mlx-vlm's own GPU and CPU runs, both
bf16, agree only to 0.98161.

## Requirements

- A Mac with 256 GB of unified memory: an M5 Ultra here. The model wires about 185 GiB, so nothing else large may
  run beside it. **Stop any other model server first**; memguard stops the load if memory runs short.
- macOS with Xcode's SDK and Metal toolchain, Zig 0.17.0 (`zig` in `recipe.toml`), Python 3.11+ for the
  orchestrator (stdlib only), and the `hf` CLI for the download (`pip install -U huggingface_hub`; public, no token).
- A checkout of TensorFold (`tensorfold_src`; `build` adds a worktree at the pinned commit) or network access to
  clone it.
- Disk: the checkpoint is 170 GiB, and the engine worktree and build about 2 GB.

## Quick start

```bash
git clone https://github.com/BobClawblaw/glm53-flash-mlx4bit-m5-ultra.git && cd glm53-flash-mlx4bit-m5-ultra
pip install -U huggingface_hub            # the hf CLI that downloads the weights (public, no token)
python3 -m mac validate                   # the settings and the guards, touches nothing
python3 -m mac up                         # build the patched engine, download, start with memguard, wait for the API
python3 -m mac status                     # pid, memory, /health, /v1/models, stats
python3 -m mac chat "Say hello."          # one greedy text reply
python3 -m mac see tests/images/shapes.png --prompt "What shapes and colours are in this image?"
python3 -m mac quality --label my-run     # the full suite and the ruler into evidence/<date>-my-run/
python3 -m mac down
```

The first `up`:
1. adds a worktree of TensorFold at the pinned commit under `~/.cache/tensorfold-glm53/engine/`;
2. applies the patch and checks its sha256;
3. builds `tensorfold-native` (about a minute);
4. downloads the checkpoint at the pinned revision;
5. checks free memory against the engine's floor plus `memory_margin_gib`;
6. drops the checkpoint's pages from the file cache;
7. starts the server with memguard beside it and waits for `/v1/models`.

Later starts take about 25 s.

To serve on the network, copy `local.example.toml` to `local.toml` (git-ignored), then set `host = "0.0.0.0"`, and
`api_key` too if the network is shared.

Autostart: `launchd/com.tensorfold.glm53-flash.plist` runs `python3 -m mac up --foreground --no-download`. After a
memguard kill, `up` refuses to start for 15 minutes, so launchd's `KeepAlive` can't retry a load into the same wall
(`MEMGUARD_RETRY=1` overrides).

### Images

Send `image_url` parts with `data:` URLs (PNG, JPEG, GIF, WebP, HEIC) in user messages:

```json
{"model": "GLM-5.3-Flash-MLX-4bit", "messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}},
  {"type": "text", "text": "Describe this image."}]}]}
```

The Anthropic (`/v1/messages`) and Responses endpoints take their own image forms and convert them the same way.

The images in a request share `vision_image_tokens` (4,096), split evenly; `"detail": "low"` caps an image at 256.
Up to `vision_max_images` (4) images are accepted across a request's history. The limits match TensorFold 0.6.6:
10 MB an image, 20 MB a request, 8192 px a side, 16 megapixels, a single frame. EXIF orientation is applied, and
alpha is composited over white.

## Settings

`recipe.toml` holds the defaults and `local.toml` (git-ignored) this Mac's values. Any setting can be overridden by
an environment variable of the same name in upper case (`PORT=8890 python3 -m mac up`).

| Setting | Default | Notes |
|---|---|---|
| `host`, `port` | `127.0.0.1`, `8888` | `0.0.0.0` serves the network |
| `profile` | `chat` | `chat`: MTP drafts; `reference`: `--no-drafts`, the exactness reference |
| `context` | 1048576 | the shared pool's tokens, the checkpoint's full window. A smaller window takes less memory (see Memory) |
| `parallel` | 3 | streams at once, sharing decode rounds and the pool (see Measured); ~0.3 GB each. 1 queues requests |
| `thinking` | true | reasoning first, in `reasoning_content`. A request sends `chat_template_kwargs: {"enable_thinking": false}` to skip it |
| `reasoning_effort` | `high` | GLM-5.3's levels are `low`, `high` and `max`. A request's `"reasoning_effort"` overrides it; empty leaves the template's default, `max` |
| `temperature` | `"0"` | greedy is all the native GLM engine decodes. Requests with a temperature are served greedily |
| `max_tokens` | 32768 | the reply cap when a request sets none. With thinking on, reasoning counts against it |
| `prompt_cache_gib` | 4 | kept prompt states, bounded so the load stays predictable |
| `vision`, `vision_image_tokens`, `vision_max_images` | true, 4096, 4 | see Images |
| `memory_limit_gb` | 215 | `TENSORFOLD_MEMORY_LIMIT_GB`, the patched one-Mac load limit |
| `residency` | true | `GLM_RESIDENCY` + `GLM_WIRE`: resident and wired before the reads. Keep it on |
| `memguard*` | 6 GiB free, 12 GiB swap growth, 24 GiB compressor growth | SIGKILL after two readings 2 s apart; 30 s at critical pressure |
| `memory_margin_gib` | 8 | `up` refuses with less free than the engine's floor plus this |
| `keep_warm`, `dashboard`, `learn*`, `compact_at` | 900, true, false | the engine's flags |
| `request_log` | empty | a file for each request and reply (image data redacted) |
| `api_key`, `extra_args`, `extra_env` | | flags and variables the recipe owns are refused |

`python3 -m mac validate` refuses these before anything runs:
- a temperature other than 0;
- a context outside [256, 1,048,576] or not a multiple of 256;
- out-of-range limits, and zero-padded or non-decimal integers;
- a patch whose sha256 isn't the pinned one;
- `extra_args` that re-set a flag the recipe builds (any prefix of them), or a flag for another platform
  (`--vision-urls`, `--speed-up`, CUDA ranks);
- `extra_env` that sets `TENSORFOLD_MEMORY_LIMIT_GB` or `GLM_RESIDENCY`.

## Memory safety

The constraint this recipe is built around: **don't run the Mac out of memory.**

- **Before a load**, `up` refuses when free memory is under the engine's floor (171 GiB) plus the margin. It also
  drops the checkpoint's pages from the file cache with `msync(MS_INVALIDATE)`, which needs no root. A cached copy
  beside the GPU copy once pushed a load to critical pressure.
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

`quality` is the standard measurement: run it after any engine, patch or settings change and commit the folder.
It caps long-context documents at `context` nominal tokens. It runs HumanEval only with `--humaneval` and Docker,
and vision only with `vision` on. The suite and the ruler come from the Qwen3.8 Spark recipe at a pinned commit
(`spark_recipe_sha`), cloned under `.external/` and run there. Its checks:

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
| `vision` | 12 generated images: shapes, colours, counts, left/right, above/below, a blank |
| `snapshot` | the reply set, for comparing two servers (`--compare a.json b.json`) |

The ruler runs four phases (prose, code, structured JSON, a long list), each with its own prompt, and reports the
per-stream decode rate and the time to the first content token.

## Not supported

- Sampled decoding: the native GLM engine is greedy only, and requests with a temperature are served greedily.
- Structured output (`response_format`, `guided_*`) is refused with a 400: the native engine has none yet.
- Remote image URLs (send `data:` URLs), video and audio.
- More than three streams at a time by default (`parallel` raises it; each stream costs ~0.3 GB).
- The two-Mac `--speed-up` pair with vision.
- The prompt cache keeps states at planned chunk starts only, so short image chats re-encode their images each turn.
- No `n > 1`, no logprobs, no presence or frequency penalties. The reasoning field is `reasoning_content`.

## Logs and evidence

`python3 -m mac logs`, `python3 -m mac status`. The engine prints one `done` line per request (prompt and cached
tokens, reply tokens, drafts accepted, the reply's sha). memguard logs to `~/.cache/tensorfold-glm53/memguard.log`.
Every number in this README has a file under [`evidence/`](evidence/), written by the suite, the ruler or
`tf-glm-vision`.

Upstream TensorFold pull requests from this work: the one-Mac load limit
([#580](https://github.com/ashhart/TensorFold/pull/580)), wired loading
([#581](https://github.com/ashhart/TensorFold/pull/581)), and the early refusal of sampled requests on greedy-only
engines ([#582](https://github.com/ashhart/TensorFold/pull/582)).

## License

MIT ([LICENSE](LICENSE)); the TensorFold patch is Apache-2.0 like the code it changes ([NOTICE](NOTICE)).
