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

TensorFold 1.0.5 serves GLM text on one 256 GB Mac when `GLM_LOAD_LIMIT_GB` raises its load limit past 70% of RAM
(179.2 GB here; the language model alone is 180.6 GB), and the recipe sets 215.
[`patches/glm-one-mac-vision-pool-v1.0.5.patch`](patches/), pinned by sha256, changes four things on top:

1. **Wired loading.** Past about 110 GB of freshly written buffers, the GPU driver's wired collector handed the
   buffers no command had touched yet to the VM compressor: 145 GiB at the peak, with pressure still normal. One
   unwired load ended with jetsam killing about 300 system daemons, then the server. With `GLM_RESIDENCY` +
   `GLM_WIRE`, every buffer is resident and wired (mlock) before the reads fill it. The full model then loads in 22 s
   with the compressor flat.
2. **Native image input** (`--vision`):
   - the checkpoint's 24-block tower on Metal;
   - images decoded through ImageIO;
   - GLM5-Next's image processor ported exactly, including PIL's 8-bit bicubic resampler;
   - `image_url` parts expanded to each image's tokens;
   - the tower's rows written over the placeholders' embeddings during the prompt pass.

   PNG patches equal TensorFold 0.6.6 + mlx-vlm's bit for bit. The tower's rows are as close to an fp32 run as
   mlx-vlm's own bf16 rows.
3. **Greedy fallback.** The native GLM engine decodes greedily only, and stock 1.0.5 refuses a request that sends a
   temperature with a 400. Agent clients always send one, so the patch serves it greedily instead.
4. **A shared cache pool.** Stock, every stream gets its own cache for the whole window when the server starts, so
   three streams at 1M would need ~65 GB of caches: more than this Mac has beside the model. The patch keeps one
   1M-token pool, the engine's own caches, and gives each request a slice of it for its prompt plus `max_tokens` while
   it runs. One request alone can use the full window; several split it. A request that does not fit beside the
   running ones waits in line until they finish. Each extra stream costs only its fixed-size state, ~0.3 GB.
   1.0.5 keeps a prompt state's cached prefix in the stream slot that wrote it; in the pool, a finished stream's
   kept prompt states hold just the blocks under their prefixes, and give them up (and go stale) only when a new
   request needs the room.

The orchestrator is stdlib-only Python (`python3 -m mac`). The quality suite and decode ruler are the
[Qwen3.8 Spark recipe's](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks), cloned at a
pinned commit and run against this server. Every number below has a file under [`evidence/`](evidence/).

## Measured on this Mac

The defaults: three streams sharing the 1,048,576-token pool, MTP drafts, vision on, model wired, TensorFold 1.0.5.
Measured with `bench/ruler.py`: greedy, thinking off, 256-token replies, a distinct prompt per phase, median of 3
runs ([`evidence/2026-10-10-1.0.5/ruler/`](evidence/2026-10-10-1.0.5/ruler/)):

| Users | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 92.4 | 104.7 | 99.3 | 96.3 | 0.11-0.15 s |

Tokens a second. The thinking-on run's ruler measured 92.3 / 105.0 / 99.3 / 97.2
([`evidence/2026-10-10-1.0.5-thinking/ruler/`](evidence/2026-10-10-1.0.5-thinking/ruler/)). Undrafted, one stream
decodes at 73 tok/s and drafted at 91, on the suite's `drafts` prompts. On 1.0.4 the same ruler measured 93.3 /
104.6 / 104.2 / 89.7 with one stream, and a lone request's list phase dropped to 55 tok/s with three streams; 1.0.5
does not show that. Two early ruler runs on 1.0.5 each had one slow phase (structured 61, then list 42) with the
same tokens and the same draft rounds; the four runs since agree with the table, and the cause was not found.

Concurrent requests share the engine's decode rounds, so a request that arrives during another's long reply starts
at once instead of queueing behind it. They share the throughput rather than add to it: one stream's rounds already
fill the GPU with MTP drafts, so a second stream mostly takes the draft rows. Tokens a second, 3 runs, 256-token
replies, ranges across the four phases
([`evidence/2026-10-10-pool-1.0.5/ruler/ruler-users.json`](evidence/2026-10-10-pool-1.0.5/ruler/ruler-users.json)):

| Users | each | aggregate | first token |
|---:|---:|---:|---:|
| 1 | 91-104 | 91-104 | 0.11-0.15 s |
| 2 | 51-68 | 101-136 | 0.19-0.24 s |
| 3 | 34-44 | 106-124 | 0.26-1.39 s |

A fourth concurrent request waits for a stream. `parallel = 1` restores one reply at a time.

Prompt passes, one prompt each, the needle found in every one, read whole (the prompt cache resumed at 0). To 200k
they come from the 1.0.5 long-context check with thinking off
([`evidence/2026-10-10-1.0.5/suite/long-context.json`](evidence/2026-10-10-1.0.5/suite/long-context.json)); the
two longest were measured on 1.0.4 with the suite's needle function
([`evidence/2026-10-09-long-1m/`](evidence/2026-10-09-long-1m/)):

| Prompt | Time to first token | Rate |
|---:|---:|---:|
| 6,390 tokens | 3.6-3.9 s | 1,633-1,761 tok/s |
| 25,435 | 14.1 s | 1,809 tok/s |
| 50,981 | 28.3-28.4 s | 1,794-1,800 tok/s |
| 102,081 | 57.7 s | 1,768 tok/s |
| 200,366 | 117.4-119.2 s | 1,681-1,707 tok/s |
| 484,568 (1.0.4) | 880 s | 551 tok/s |
| 926,599 (1.0.4) | 2,533 s | 366 tok/s |

1.0.5 reads prompts in chunks of up to 16,384 tokens (4,096 on 1.0.4), about twice as fast at 100-200k.

Images ([`evidence/2026-10-09-native-vision-1.0.3.md`](evidence/2026-10-09-native-vision-1.0.3.md)):

| Image | Tokens | Preprocess | Tower | Request to first token |
|---|---:|---:|---:|---:|
| 96x96 logo | 16 | 1.6 ms | 8 ms | 0.5 s |
| 517x389 drawing | 266 | 3.4 ms | 26 ms | 2.1 s |
| 2000x1500 photo-like, 1,024-token budget | 972 | 43 ms | 154 ms | |
| 2000x1500, the full 4,096-token budget | 3,909 prompt tokens | | | 6.4 s |

Memory, with the model and tower wired. GLM's caches are small: most of its layers keep constant-size state. The
default load on 1.0.5, three streams sharing the 1,048,576-token pool: 21.4 s, 201.4 GiB wired and 39.9 GiB free
after load, 200.5 GiB and 30.8 GiB after both suite runs, compressor ~5-6 GiB, pressure normal
([`evidence/2026-10-10-pool-1.0.5/server-load.txt`](evidence/2026-10-10-pool-1.0.5/server-load.txt)). Smaller
windows, measured with one stream on 1.0.4:

| window | caches | weights + caches | wired | free after load |
|---:|---:|---:|---:|---:|
| 131,072 | 6.3 GB | 186.9 GB | 183 GiB | ~38 GiB |
| 262,144 | 8.9 GB | 189.5 GB | | |
| 524,288 | 14.0 GB | 194.6 GB | | |
| 1,048,576 | 24.2 GB | 204.8 GB | 202 GiB | ~27 GiB |

The pool was checked on 1.0.5 ([`evidence/2026-10-10-pool-1.0.5/`](evidence/2026-10-10-pool-1.0.5/),
`tools/pool_*.py`):
- the suite's `concurrent` check: four streams together give the same replies as each alone, 12/12, thinking on
  and off;
- 7k- and 10k-token prompts (the sparse attention path) running beside a 24k-token one, in slices at offsets in
  the pool, gave byte-identical replies to each alone;
- a request needing more room than was free waited for the running one to finish, then answered; a request larger
  than the window was refused with a 400;
- prompt reuse after a stream leaves: a second question on a 6.4k-token system prompt, asked while another stream
  held slot 0, resumed at 6,362 of 6,376 tokens from the blocks its finished stream left held (0.55 s, against
  6.05 s read whole). A request then asking for nearly the whole pool did not wait: the held blocks gave way, and
  the same question read its prompt whole again with a byte-identical reply. After such an eviction the next turn
  reads its prompt once more before the cache keeps it again.

The tower is 0.9 GiB, plus 1.2 GiB of scratch for 4,096 image tokens.

## Quality

The Spark recipe's suite on the defaults (TensorFold 1.0.5, three streams, the 1M pool), greedy, datasets pinned.
Each check ran twice: with thinking **on** at effort High (the recipe's default; `python3 -m mac quality --thinking
32768`, up to 32,768 reasoning tokens a request) and with thinking **off** (as the suite sends on its own). Receipts:
[`evidence/2026-10-10-1.0.5-thinking/`](evidence/2026-10-10-1.0.5-thinking/) and
[`evidence/2026-10-10-1.0.5/`](evidence/2026-10-10-1.0.5/).

| Check | Thinking on (default) | Thinking off |
|---|---|---|
| drafted equals undrafted; four streams together equal alone | 12/12; 12/12 | 12/12; 12/12 |
| streaming = non-streaming, multi-turn recall, stop strings, thinking mode, verbatim copy | all pass | all pass |
| exact `max_tokens` | 0/3 (see below) | 3/3 |
| tool calls (60: exact function and arguments, optional arguments left out, parallel calls, no-call cases) | 57/60 | 57/60 |
| JSON schemas | 30/30 | 30/30 |
| IFEval (150 prompts, the 24 instruction types implemented) | 131 loose / 121 strict | 96 / 89 |
| GSM8K (250) | 248 (99.2%) | 246 (98.4%) |
| MGSM (8 languages x 30) | 226 (94.2%) | 217 (90.4%) |
| MMLU (6 a subject, 342) | 308 (90.1%) | 292 (85.4%) |
| HumanEval (164, pass@1, sandboxed) | not run (needs the suite's Docker sandbox) | not run |
| repetition (eight 600-word replies) | 8/8, 0.19% repeated 4-grams | 8/8, 0.18% |
| long context: needles at 10/50/90% of 6-51k and 2/50/98% of 102k and 200k; ten spread facts at 25k-200k | 54/55 | 55/55 |
| vision (12 drawn probes) | 12/12 | 12/12 |

On 1.0.4 (one stream, before the pool) the same suite gave IFEval 127/117, MMLU 307, GSM8K 248, MGSM 221 with
thinking on, and IFEval 98/92, MMLU 288, GSM8K 245, MGSM 220 with it off
([`evidence/2026-10-10-thinking-on/`](evidence/2026-10-10-thinking-on/),
[`evidence/2026-10-09-quality.md`](evidence/2026-10-09-quality.md)).

Notes on the results:

- **Thinking pays where it should.** IFEval rises from 96 to 131, MMLU from 292 to 308, MGSM from 217 to 226. The
  replies with thinking off were well formed but broke the instruction; with reasoning the model checks itself.
- **Exact `max_tokens`** fails with thinking on because of how the run is set up, not the server. To leave room for
  reasoning, the wrapper adds the 32,768-token budget to every reply limit, so the model finishes well under the
  raised limit. The thinking-off run (3/3) is the test of the limit.
- **Long context.** With thinking on, the one miss is a 200k-token document with the needle halfway in: the model
  answered that the document held no blue door (on 1.0.4 the miss was the 98% needle). With thinking off every
  needle and fact was found to 200k. For plain retrieval from a long document, send
  `"chat_template_kwargs": {"enable_thinking": false}`. The suite sizes documents nominally (8k-250k) and GLM's
  tokenizer makes them about 20% shorter, so "250k" is about 200k tokens. Beyond the suite, needles were found at
  485k and 927k tokens on 1.0.4 (see Measured).
- **Tools.** The misses are the checker's exact match, the same three both ways. The model picked the right function
  with the right core values, then added an optional `label` to two timers ("5 minutes", "timer") or a fuller place
  name ("notre-dame de paris").
- **Serving checks beyond the suite.** Tool calls with `temperature` 0.6 (as agents send), thinking off and on,
  streamed and not, all returned valid `tool_calls`. Parallel calls with a `tool`-result round trip answered
  correctly. An agent (hermes) on another machine sent 33 requests, all 200 (on 1.0.4).

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
| `memory_limit_gb` | 215 | `GLM_LOAD_LIMIT_GB`, 1.0.5's one-Mac load limit (unset: 70% of RAM). Past 70% the server logs the watchdog-panic risk |
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
- `extra_env` that sets `GLM_LOAD_LIMIT_GB` or `GLM_RESIDENCY`.

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
