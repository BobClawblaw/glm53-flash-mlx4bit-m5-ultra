"""bf16 noise floor for the tower: python -I tools/vision_noise.py REF_DIR (after tf-glm-vision wrote native_features.f32).

Compares mlx-vlm on GPU in bf16, on CPU in bf16 and on GPU in fp32 with the native rows, by row cosine.
"""
import json, sys
from pathlib import Path
import numpy as np, mlx.core as mx
from mlx_vlm.models.glm5_next.config import VisionConfig
from mlx_vlm.models.glm5_next.vision import VisionModel
SNAP = Path.home() / ".cache/huggingface/hub/models--TensorFold--GLM-5.3-Flash-MLX-4bit-MTP/snapshots/d2b17f5253440224422b150705c1109a9a3a0368"
ref = Path(sys.argv[1]); meta = json.loads((ref / "meta.json").read_text())
config = json.loads((SNAP / "config.json").read_text())
tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
index = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
w = {}
for shard in sorted({f for k, f in index.items() if k.startswith("model.visual.")}):
    for k, v in mx.load(str(SNAP / shard)).items():
        if k.startswith("model.visual."): w[k[13:]] = v
tower.load_weights(list(tower.sanitize(w).items()), strict=True)
P = np.fromfile(ref / "patches.f32", dtype=np.float32).reshape(-1, 1176)
g = mx.array([meta["grid_thw"]], dtype=mx.int32)
def run(dtype, dev):
    with mx.stream(dev):
        t = tower
        if dtype != mx.bfloat16:
            t.set_dtype(dtype)
        out = t(mx.array(P).astype(dtype), g).astype(mx.float32); mx.eval(out)
        if dtype != mx.bfloat16: t.set_dtype(mx.bfloat16)
        return np.array(out)
def cos(a, b):
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
    return f"mean {c.mean():.6f} worst {c.min():.6f}"
gpu = run(mx.bfloat16, mx.gpu); cpu = run(mx.bfloat16, mx.cpu); f32 = run(mx.float32, mx.gpu)
nat = np.fromfile(ref / "native_features.f32", dtype=np.float32).reshape(gpu.shape)
print("mlx gpu bf16 vs mlx cpu bf16:", cos(gpu, cpu))
print("mlx gpu bf16 vs mlx gpu f32: ", cos(gpu, f32))
print("native bf16  vs mlx gpu f32: ", cos(nat, f32))
print("native bf16  vs mlx gpu bf16:", cos(nat, gpu))
