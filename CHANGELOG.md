# Changelog

## 2026-10-10

- **TensorFold 1.0.5** (`bec00ae6`) with `patches/glm-one-mac-vision-pool-v1.0.5.patch`. 1.0.5's own
  `GLM_LOAD_LIMIT_GB` replaces the patched `TENSORFOLD_MEMORY_LIMIT_GB` (the recipe still sets 215), and its faster
  GLM decode and prompt kernels, out-of-page-cache loads and the prompt cache for prompts under 4,096 tokens come with
  it. The patch keeps wired loading, vision, the greedy fallback (stock 1.0.5 refuses a temperature with a 400) and
  the shared pool. 1.0.5 keeps a prompt state's prefix in its stream's slot; in the pool a finished stream's kept
  states now hold the blocks under their prefixes until a new request needs the room. Golden 309/319 with 0
  unexpected differences, `zig build test` passes.
- **Measured on 1.0.5, three streams, 1M pool**: one user 92 / 105 / 99 / 96 tok/s (prose / code / structured /
  list; the 55 tok/s list phase seen with three streams on 1.0.4 is gone), 101-136 tok/s aggregate at two users and
  106-124 at three. Prompts read at ~1,700-1,800 tok/s to 200k tokens (810-1,134 on 1.0.4): 200k in 118 s.
  Quality with thinking on: IFEval 131/121, MMLU 308, GSM8K 248, MGSM 226, long context 54/55; off: IFEval 96/89,
  MMLU 292, GSM8K 246, MGSM 217, long context 55/55. Tools 57/60 both ways (`evidence/2026-10-10-1.0.5*`,
  `evidence/2026-10-10-pool-1.0.5/`).
- **`python3 -m mac quality --resume`** continues an interrupted run: saved checks are not rerun and the report is
  rebuilt from every saved check.
- **The full 1,048,576-token window, shared by three streams, by default.** The patch
  (`glm-one-mac-vision-pool-v1.0.4.patch`) keeps one MLA cache pool of the window's size and gives each request a
  slice for its prompt plus `max_tokens` while it runs; a request that does not fit waits for room. One request alone
  can use the full window. Memory is the old one-stream 1M load's: ~200 GiB wired, ~27 GiB free (`memory_limit_gb`
  215). `PROFILE=long` is gone: the default is the full window. Replies with streams together equal each alone, also
  for long prompts at offsets in the pool (`evidence/2026-10-10-pool/`).
- **Three streams** (`parallel = 3`): concurrent requests share decode rounds, no queueing behind a long reply,
  aggregate 93-123 tok/s at two and three users, a lone request as fast as before
  (`evidence/2026-10-10-parallel3/ruler/`).
- **A 262,144-token window by default, and the full 1,048,576 with `PROFILE=long`.** GLM's caches are small (8.9 GB at
  262k, 24.2 GB at 1M, beside 180.6 GB of weights); the long profile raises `memory_limit_gb` to 215. At 1M the model
  loads in 24 s, 202 GiB wired, about 27 GiB free. Needles found at 200,465, 484,568 and 926,599 prompt tokens
  (810 / 551 / 366 tok/s), memory flat throughout (`evidence/2026-10-09-long-1m/`).
- **Reasoning effort High by default** (`reasoning_effort`); GLM's template defaults to Max when none is sent.
- **Quality with thinking on** (effort High, up to 32,768 reasoning tokens): IFEval 127/150 (98 off), GSM8K 248/250,
  MMLU 307/342 (288 off), MGSM 221/240, tools 58/60, long context 54/55 to 200k tokens; the one miss, a 200k needle at
  98% depth, is found with thinking off.
- **The suite with thinking on**: `python3 -m mac quality --thinking BUDGET` runs every check through
  `tools/suite_thinking.py`, which turns thinking on in each request and adds the reasoning budget to its reply limit.

## 2026-10-09

- **GLM-5.3-Flash MLX 4-bit on one 256 GB M5 Ultra, natively** on TensorFold v1.0.4 (`d4fc1964`) with
  `patches/glm-one-mac-vision-v1.0.4.patch`. Stock 1.0.4 refuses the 180.6 GB checkpoint on one Mac (70% of RAM);
  the patch makes the limit `TENSORFOLD_MEMORY_LIMIT_GB` (the recipe sets 200).
- **Native image input** (`--vision`): the checkpoint's 24-block tower on Metal (tensor-unit GEMMs, 2-D rotary
  attention, the 2x2 merge, the merger), image decoding through ImageIO with EXIF orientation and alpha over white
  as TensorFold 0.6.6 does, the processor's sizing and PIL's 8-bit bicubic resampler ported exactly, data-URL
  `image_url` parts expanded to each image's tokens, a prompt-cache key by image content. PNG patches equal 0.6.6 +
  mlx-vlm's bit for bit; the tower's rows are as close to an fp32 run as mlx-vlm's own bf16 rows.
- **Wired loading** (`GLM_RESIDENCY` + `GLM_WIRE`): every buffer resident and mlocked before it fills. Unwired, past
  ~110 GB the GPU driver's wired collector compressed the untouched buffers (145 GiB) and one load ended in a
  jetsam event. Wired: loaded in 22 s (80 s before), compressor flat, 183 GiB wired, ~38 GiB free.
- **Memory guard** (`mac/memguard.py`): SIGKILL on low free memory, swap growth or compressor growth (two readings),
  or 30 s of critical pressure; `up` drops the checkpoint's file cache first and refuses for 15 minutes after a kill.
- **Greedy fallback**: the native GLM engine decodes greedily only; a request with a temperature is served greedily
  instead of failed (hermes sends one with every request).
- **Defaults**: port 8888, thinking on, vision on (4096 image tokens, 4 images), context 131,072, one stream.
- **Quality** (thinking off, the Spark recipe's suite): GSM8K 245/250, MGSM 220/240, MMLU 288/342, IFEval 98/150, long context 42/42 to 102k tokens, vision 12/12,
  tools 57/60, JSON 30/30, repetition 8/8, every exactness and behaviour check passing.
- **Decode** (one stream, thinking off): 93.3 / 104.6 / 104.2 / 89.7 tok/s on prose / code / JSON / list, first token 0.23-0.29 s.
