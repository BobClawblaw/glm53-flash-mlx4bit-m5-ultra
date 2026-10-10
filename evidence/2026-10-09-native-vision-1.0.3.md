# Native vision on TensorFold 1.0.3, one M5 Ultra (256 GB) — 2026-10-09

Engine: TensorFold v1.0.3 (`24b009ff`) + `patches/glm-one-mac-vision-v1.0.4.patch` (sha256 `204735da…`).
Model: `TensorFold/GLM-5.3-Flash-MLX-4bit-MTP` @ `d2b17f52`. macOS: Darwin 27.0.0.

## Engine tests (patched tree)

- `zig build test`: 119/119 steps, 770/772 tests passed (2 skipped).
- `zig build test-golden`: 0 differ in every category (the same table as pristine 1.0.2; the printed "known" cases
  are the suite's own).
- The patch applies cleanly to a pristine v1.0.3 checkout.

## The image tower against mlx-vlm (the code 0.6.6's `--vision` serves)

`zig-out/bin/tf-glm-vision MODEL REF STAGE IMAGE` against `tools/vision_reference.py` dumps. Patches: native
decode + resize + pad + normalize, compared value by value. Features: rows' cosine against mlx-vlm's bf16 output.

| image | patches differing | cosine mean / worst |
| --- | --- | --- |
| shapes.png 517x389 (pad only) | 0 of 1,251,264 | 0.9835 / 0.695 |
| large.png 2000x1500 @1024 tokens (antialiased downscale) | 0 of 4,572,288 | 0.9935 / 0.722 |
| zai-logo.png 96x96 RGBA (upscale, alpha over white) | 0 of 75,264 | 0.9984 / 0.992 |
| tiny.png 20x15 (upscale to the 16-token floor) | 0 of 94,080 | 0.9277 / 0.620 |
| palette.png (transparent palette entry) | 0 of 1,251,264 | 0.9832 / 0.728 |
| gray-alpha.png (LA) | 0 of 225,792 | 0.9909 / 0.826 |
| rotated.jpg (EXIF orientation 6) | 17,726 of 413,952 (JPEG decoder) | 0.9815 / 0.856 |
| shapes.jpg | 116,000 of 1,251,264 (JPEG decoder) | 0.9706 / 0.528 |

The cosines are bf16 noise, not error. Against an fp32 run of the same tower (`tools/vision_noise.py`, same patches):

| image | mlx bf16 GPU vs fp32 | native vs fp32 | mlx GPU vs mlx CPU (both bf16) |
| --- | --- | --- | --- |
| shapes.png | 0.98718 / 0.767 | 0.98720 / 0.766 | 0.98161 / 0.613 |
| tiny.png | 0.91464 / 0.542 | 0.94063 / 0.754 | 0.89900 / 0.642 |
| zai-logo.png | 0.97624 / 0.863 | 0.98049 / 0.909 | 0.99049 / 0.910 |
| large.png | 0.99442 / 0.753 | 0.99385 / 0.648 | 0.99412 / 0.730 |

Tower time (encode only): 64 patches 8 ms, 1,064 patches 26 ms, 3,888 patches 154 ms. Preprocessing 2-43 ms.

## Served end to end (`python3 -m mac see`, greedy)

- shapes.png, "What shapes and colours…": "a red circle on the left, a blue square on the right, and a row of thin
  vertical stripes … along the bottom, … the text "GLM 7" at the top." 295 prompt tokens, 2.1 s.
- zai-logo.png: "a stylized letter Z in white on a dark rounded square background." 0.5 s.
- rotated.jpg (EXIF 6): circle on the left, text running vertically: the orientation applied.
- Two images in one message: described in order. A second turn about the first turn's image ("And what colour is the
  square?") answers "Blue." from the history.
- large.png at the full 4,096-token budget: 3,909 prompt tokens, first token in 6.4 s.
- Text: 383 tokens in 5.0 s (76.6 tok/s including prefill); /stats decode 77.5 tok/s.

## Memory

- Unwired loads (1.0.2, `GLM_RESIDENCY` only): past ~110 GB of fresh buffers the compressor took the untouched GPU
  buffers (123-145 GiB) even at pressure 1. One such load, with Time Machine's ~73 GiB file cache beside it, ended in
  a jetsam event at 19:22:42: ~300 system daemons killed for low swap, then the server.
- Wired loads (`GLM_WIRE`: mlock before the reads): 35 layers — wired 140 GiB, compressor flat (1.5-1.7 GiB),
  pressure 1 throughout. Full model with vision — loaded in 21.7 s (80 s unwired), wired 183 GiB, compressor +2.6
  GiB, pressure 1, 38 GiB free; after the image and text requests 39.5 GiB free, compressor 2.4 GiB.

## Not exercised live

The prompt cache kept no state in these tests, text or image (states are kept at planned chunk starts; none of the
prompts crossed one), so the image-aware cache key is covered only by `server/images.zig`'s unit test.
