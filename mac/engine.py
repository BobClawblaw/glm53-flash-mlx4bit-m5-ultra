"""The engine binary: a worktree of TensorFold at the pinned sha under tf_cache, the recipe's patch applied, built with Zig."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .config import Settings
from .log import log


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 3600) -> str:
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{' '.join(cmd)} (exit {p.returncode}):\n{p.stdout[-2000:]}{p.stderr[-4000:]}")
    return p.stdout


def ensure_source(s: Settings) -> Path:
    """A worktree at the pinned sha, from the checkout in `tensorfold_src` or a fresh clone of `tensorfold_repo`."""
    if (s.build_dir / "build.zig").is_file():
        head = _run(["git", "-C", str(s.build_dir), "rev-parse", "HEAD"]).strip()
        if head != s.tensorfold_sha:
            raise RuntimeError(f"{s.build_dir} is at {head[:8]}, not the pinned {s.tensorfold_sha[:8]}; remove it to rebuild")
        return s.build_dir
    s.build_dir.parent.mkdir(parents=True, exist_ok=True)
    src = s.tensorfold_src
    if (src / ".git").exists():
        have = subprocess.run(["git", "-C", str(src), "cat-file", "-e", f"{s.tensorfold_sha}^{{commit}}"], capture_output=True)
        if have.returncode:
            log(f"{src} lacks {s.tensorfold_sha[:8]}; fetching from {s.tensorfold_repo}")
            _run(["git", "-C", str(src), "fetch", s.tensorfold_repo, s.tensorfold_sha])
        log(f"Adding a worktree of {src} at {s.tensorfold_sha[:8]} under {s.build_dir}")
        _run(["git", "-C", str(src), "worktree", "add", "--detach", str(s.build_dir), s.tensorfold_sha])
    else:
        log(f"Cloning {s.tensorfold_repo} at {s.tensorfold_sha[:8]} into {s.build_dir}")
        _run(["git", "clone", "--no-checkout", s.tensorfold_repo, str(s.build_dir)])
        _run(["git", "-C", str(s.build_dir), "checkout", "--detach", s.tensorfold_sha])
    return s.build_dir


def ensure_patched(s: Settings, src: Path) -> bool:
    """The pinned patch applied to the worktree, once; a worktree carrying any other change is refused. True if just applied."""
    marker = src / ".recipe-patch"
    if marker.is_file() and marker.read_text().strip() == s.patch_sha256:
        if subprocess.run(["git", "-C", str(src), "apply", "--check", "--reverse", str(s.patch_path)], capture_output=True).returncode:
            raise RuntimeError(f"{src} carries changes beyond the pinned patch; remove the worktree to rebuild it")
        return False
    if subprocess.run(["git", "-C", str(src), "diff", "--quiet"], capture_output=True).returncode:
        raise RuntimeError(f"{src} has uncommitted changes that are not the pinned patch; remove the worktree to rebuild it")
    log(f"Applying patches/{s.patch} ({s.patch_sha256[:8]}) to the worktree")
    _run(["git", "-C", str(src), "apply", "--check", str(s.patch_path)])
    _run(["git", "-C", str(src), "apply", str(s.patch_path)])
    marker.write_text(s.patch_sha256 + "\n")
    return True


def ensure_binary(s: Settings, rebuild: bool = False) -> Path:
    """The binary to serve with: `binary` as given, or the pinned sha plus patch built once (zig build native)."""
    if s.binary:
        return s.run_binary
    src = ensure_source(s)
    patched_now = ensure_patched(s, src)
    if s.built_binary.is_file() and not rebuild and not patched_now:
        return s.built_binary
    if not s.zig.is_file():
        raise RuntimeError(f"zig={s.zig} is not a file: install Zig {(src / '.zig-version').read_text().strip()} there "
                           "or set `zig` in local.toml")
    want = (src / ".zig-version").read_text().strip()
    have = _run([str(s.zig), "version"]).strip()
    if have != want:
        raise RuntimeError(f"{s.zig} is Zig {have}; the checkout pins {want}")
    log(f"Building tensorfold-native at {s.tensorfold_sha[:8]}+patch (zig build native -Dcpu={s.zig_cpu}); about a minute")
    _run([str(s.zig), "build", "native", f"-Dcpu={s.zig_cpu}"], cwd=src, timeout=3600)
    if not s.built_binary.is_file():
        raise RuntimeError(f"the build left no {s.built_binary}")
    log(f"Built {s.built_binary}: {_run([str(s.built_binary), '--version']).strip()}")
    return s.built_binary


def info(s: Settings, model_dir: Path) -> str:
    """`tensorfold-native info <dir>`: family, format, weight bytes and the memory floor."""
    return _run([str(ensure_binary(s)), "info", str(model_dir)], timeout=300)


def binary_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
