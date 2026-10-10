"""The checkpoint: a Hugging Face snapshot at the pinned revision, served from the cache by folder."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .config import Settings
from .log import log

REQUIRED = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
            "chat_template.jinja", "generation_config.json")


def complete(snapshot: Path) -> bool:
    """Every required file and every shard the index names resolves to a real file (the cache uses symlinks)."""
    if not all((snapshot / f).is_file() for f in REQUIRED):
        return False
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    return bool(shards) and all((snapshot / sh).is_file() for sh in shards)


def find_cached(s: Settings) -> tuple[str, Path] | None:
    """The pinned repository's snapshot, or a `same_as` repository's at the same revision (the same git tree)."""
    for repo in s.candidate_repos:
        snap = s.snapshot_dir(repo)
        if complete(snap):
            return repo, snap
    return None


def ensure_snapshot(s: Settings, download: bool = True) -> Path:
    found = find_cached(s)
    if found:
        repo, snap = found
        note = "" if repo == s.repo else f" (the same revision under {repo}; the trees are the same)"
        log(f"Checkpoint: {snap}{note}")
        return snap
    if not download:
        raise RuntimeError(f"no complete snapshot of {s.repo} at {s.revision[:8]} under {s.hf_cache} (--no-download)")
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if not hf:
        raise RuntimeError("the `hf` CLI is missing: pip install -U huggingface_hub")
    log(f"Downloading {s.repo}@{s.revision[:8]} (169 GiB) with {hf}")
    env = {**os.environ, "HF_HOME": str(s.hf_cache)}
    p = subprocess.run([hf, "download", s.repo, "--revision", s.revision], env=env, text=True)
    if p.returncode:
        raise RuntimeError(f"hf download exited {p.returncode}")
    snap = s.snapshot_dir()
    if not complete(snap):
        raise RuntimeError(f"{snap} is incomplete after the download")
    return snap
