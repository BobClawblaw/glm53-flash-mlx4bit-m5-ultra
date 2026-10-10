"""Reference dumps for the native GLM vision port: python -I tools/vision_reference.py IMAGE OUT_DIR [MAX_TOKENS].

Decodes the image as TensorFold 0.6.6's --vision does (vision/images.py: EXIF orientation, alpha over white), then
runs mlx-vlm's GLM5-Next processor and tower (the code 0.6.6 serves with) and writes what the native engine must
reproduce, as raw little-endian files plus a meta.json:
  patches.f32   (grid_h*grid_w, 1176) normalized patches, temporal frames duplicated
  embed.f32     (grid_h*grid_w, 1024) after patch_embed
  block0.f32    (grid_h*grid_w, 1024) after the first block
  blocks.f32    (grid_h*grid_w, 1024) after the last block, before post_layernorm
  features.f32  (tokens, 4096) the tower's output, the rows that replace <|image|> embeddings
Needs the venv with tensorfold[vision] (mlx-vlm 0.7.x). Loads only model.visual.* (about 1 GB).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from tensorfold.vision.images import DEFAULT_LIMITS, _decode
from mlx_vlm.models.glm5_next.config import VisionConfig
from mlx_vlm.models.glm5_next.processing import Glm5NextImageProcessor
from mlx_vlm.models.glm5_next.vision import VisionModel

SNAP = Path.home() / ".cache/huggingface/hub/models--TensorFold--GLM-5.3-Flash-MLX-4bit-MTP/snapshots/d2b17f5253440224422b150705c1109a9a3a0368"


def main() -> int:
    image_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((SNAP / "config.json").read_text())
    proc_cfg = json.loads((SNAP / "processor_config.json").read_text())["image_processor"]
    proc_cfg.pop("image_processor_type", None)
    processor = Glm5NextImageProcessor(**proc_cfg)
    decoded = _decode(image_path.read_bytes(), "auto", DEFAULT_LIMITS, DEFAULT_LIMITS.max_total_pixels)
    image = decoded.to_pil()
    done = processor([image], return_tensors="np", min_image_tokens=min(16, max_tokens), max_image_tokens=max_tokens)
    patches = np.asarray(done["pixel_values"], dtype=np.float32)
    grid = np.asarray(done["image_grid_thw"], dtype=np.int64)

    tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
    index = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for shard in sorted({f for k, f in index.items() if k.startswith("model.visual.")}):
        for k, v in mx.load(str(SNAP / shard)).items():
            if k.startswith("model.visual."):
                weights[k[len("model.visual."):]] = v
    tower.load_weights(list(tower.sanitize(weights).items()), strict=True)
    mx.eval(tower.parameters())

    dtype = tower.patch_embed.proj.weight.dtype
    x = mx.array(patches).astype(dtype)
    g = mx.array(grid, dtype=mx.int32)
    h = tower.patch_embed(x)
    embed = h
    pos = tower._rotary_embeddings(g)
    cu = mx.array([0, int(h.shape[0])])
    block0 = None
    for i, block in enumerate(tower.blocks):
        h = block(h, cu, pos)
        if i == 0:
            block0 = h
    blocks = h
    features = tower(x, g)
    mx.eval(embed, block0, blocks, features)
    for name, a in (("patches", patches), ("embed", embed), ("block0", block0), ("blocks", blocks), ("features", features)):
        np.asarray(a.astype(mx.float32) if isinstance(a, mx.array) else a, dtype=np.float32).tofile(out / f"{name}.f32")
    meta = {"image": str(image_path), "size": list(image.size), "mode": Image.open(image_path).mode, "grid_thw": grid[0].tolist(),
            "max_tokens": max_tokens, "tokens": int(features.shape[0]), "patch_width": int(patches.shape[1]),
            "dtype": str(dtype)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
