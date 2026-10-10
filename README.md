# GLM-5.3-Flash MLX 4-bit on one M5 Ultra, with images

`TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` served by TensorFold's native engine (v1.0.4) on one 256 GB Mac, with image
input running natively: the checkpoint's vision tower on Metal, image decoding and preprocessing in the engine.
Stock 1.0.4 serves GLM-5.3-Flash text only and refuses this checkpoint on one Mac; `patches/` holds what changes that.

```bash
python3 -m mac up                       # build the patched engine if needed, find the weights, start, wait for the API
python3 -m mac chat "Say hello."        # one greedy text reply
python3 -m mac see tests/images/shapes.png --prompt "What shapes and colours are in this image?"
python3 -m mac status                   # pid, memory, /health, /v1/models, stats
python3 -m mac down
```

The API is OpenAI-compatible at `http://127.0.0.1:8888/v1`. Send images as `image_url` parts with `data:` URLs:

```json
{"model": "GLM-5.3-Flash-MLX-4bit", "temperature": 0, "messages": [{"role": "user", "content": [
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "auto"}},
  {"type": "text", "text": "Describe this image."}]}]}
```

The Anthropic and Responses endpoints take their own image forms and convert them the same way.

## What the patch does (`patches/glm-one-mac-vision-v1.0.4.patch`, pinned by sha256)

- **One-Mac load limit.** GLM's load limit is 70% of RAM (179 GB on this Mac) and the checkpoint needs 180.6 GB. The
  patch makes the limit overridable with `TENSORFOLD_MEMORY_LIMIT_GB`; the recipe sets 200.
- **Wired weights** (`GLM_WIRE`, `GLM_RESIDENCY`). Every buffer is wired with mlock and made resident before the
  reads fill it. Without this, past ~110 GB of fresh buffers the GPU driver's wired collector handed the buffers no
  command had touched yet to the compressor (up to 145 GiB). One such load ended with jetsam killing about 300 system
  daemons. Wired, the load takes 22 s instead of 80 s and nothing of the model compresses.
- **Native `--vision`:**
  - `families/glm/vision.zig` and `kernels/metal/glm_vision.metal`: the tower (24 blocks of 1024, 2-D rotary
    attention, the 2x2 merge, the merger) on the prefill GEMM template.
  - `families/glm/image.zig`: decoding through ImageIO, with EXIF orientation and alpha over white as TensorFold
    0.6.6 does. Then the processor's sizing, PIL's 8-bit bicubic resampler (ported exactly), padding, normalizing and
    patch order.
  - `server/images.zig`: data URLs, each `<|image|>` placeholder expanded to the image's tokens, and a prompt-cache
    key that tells images apart.
  - The tower's output rows overwrite the placeholders' embeddings during prefill (`forward.inject`, and the MTP
    head one position on).
  - Flags: `--vision`, `--vision-image-tokens`, `--vision-max-images`.

Accuracy and timing are in `evidence/`:
- PNG preprocessing is bit-exact against TensorFold 0.6.6 + mlx-vlm.
- The tower's output is as close to an fp32 run as mlx-vlm's own bf16 output.
- An image's first token takes 0.5-2 s for typical images, 6.4 s at the full 4,096-token budget. Text decodes at
  about 77 tok/s.

## Measured on this Mac (`evidence/2026-10-09-quality.md`)

Greedy, thinking off, one stream; the Spark recipe's suite and ruler.

| | |
| --- | --- |
| decode (prose / code / JSON / list) | 93.3 / 104.6 / 104.2 / 89.7 tok/s, first token 0.23-0.29 s |
| image to first token | 0.5-2 s for typical images, 6.4 s at the full 4,096-token budget |
| load | 22 s from a warm disk, 183 GiB wired, ~38 GiB left free |
| GSM8K / MGSM / MMLU | 245/250 / 220/240 / 288/342 |
| IFEval | 98/150 (thinking on fixed 6 of the first 10 misses) |
| tools / JSON / repetition / vision | 57/60 / 30/30 / 8/8 / 12/12 |
| exactness and behaviour | drafts, concurrent, streaming, multiturn, stop, copy, max-tokens, snapshot: all pass |
| long context | 42/42: needles 12/12 to 102k tokens, spread facts 10/10 at 25k, 51k and 102k; prefill 903 tok/s at the slowest |

The recipe now defaults to thinking on. The user's hermes agent runs against it from another machine.

## Settings

`recipe.toml` holds the defaults. `local.toml` overrides them (copy `local.example.toml`), and an upper-case
environment variable overrides both (`PORT=8090 python3 -m mac up`). `python3 -m mac validate` prints the settings in
effect. Vision settings:

| setting | default | meaning |
| --- | --- | --- |
| `thinking` | `true` | reasoning first (`reasoning_content`); send `chat_template_kwargs: {"enable_thinking": false}` to skip it |
| `vision` | `true` | load the tower (~0.9 GiB) and its scratch (~1.2 GiB at 4096 tokens) |
| `vision_image_tokens` | `4096` | visual tokens a request's images share. Each image gets an equal part; `"detail": "low"` gets at most 256 |
| `vision_max_images` | `4` | images a request may hold, across its whole history |

## Memory safety

- `up` refuses to start when free memory is under the engine's floor plus `memory_margin_gib`.
- `up` drops the checkpoint's pages from the file cache before loading. A cached copy beside the GPU copy was what
  pushed earlier loads into the compressor.
- `mac/memguard.py` runs beside the server and SIGKILLs it after two readings 2 s apart (4 s) of any of these:
  free+inactive memory under 6 GiB, swap grown 12 GiB, or the compressor grown 24 GiB. With the model wired, any
  compressor growth means other memory is being squeezed. macOS's critical pressure level kills after 30 s.
- After a memguard kill, `up` refuses to start for 15 minutes (`MEMGUARD_RETRY=1` overrides), so launchd's
  `KeepAlive` can't retry a load into the same wall.
- At rest, with the model and tower wired: 183 GiB wired, about 38 GiB free.

## Limits

- Greedy only. A request that sends a temperature is served greedily, because clients such as hermes always send
  one. `response_format` (structured output) is refused.
- Only one model fits. GLM wires ~185 GiB, so stop any other model server before `up`; memguard stops the load if
  memory runs short.
- Images arrive as data URLs. Remote URLs are refused (`--vision-urls` is not served natively).
- The limits match 0.6.6: 10 MB an image, 20 MB a request, 8192 px a side, 16 MP, single-frame only. Video and
  audio are refused.
- JPEGs decode through ImageIO, not libjpeg-turbo. Pixels differ slightly from PIL's, mostly at chroma edges.
- One Mac only: `--vision` with the `--speed-up` pair is refused.
- The prompt cache keeps states at planned chunk starts only. Short image chats re-encode their images each turn.

## Checks

```bash
python3 -m unittest discover tests        # the recipe's settings, guards and restart guard
# engine: in ~/.cache/tensorfold-glm53/engine/<sha>/ — zig build test, zig build test-golden
# the tower against mlx-vlm (needs the reference venv: uv venv + tensorfold[vision] at v0.6.6):
python -I tools/vision_reference.py tests/images/shapes.png REF_DIR
zig-out/bin/tf-glm-vision MODEL_DIR REF_DIR features tests/images/shapes.png
```

## License

MIT ([LICENSE](LICENSE)); the TensorFold patch is Apache-2.0 like the code it changes ([NOTICE](NOTICE)).
