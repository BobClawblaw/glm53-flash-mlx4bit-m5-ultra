"""The Spark recipe's quality suite and decode ruler, run from a pinned clone under .external/ (never copied here)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from .config import ROOT, Settings
from .log import log


def ensure_clone(s: Settings) -> None:
    d = s.external_dir
    if (d / "quality" / "suite.py").is_file():
        head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == s.spark_recipe_sha:
            return
        log(f"{d} is at {head[:8]}; fetching the pinned {s.spark_recipe_sha[:8]}")
    else:
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "remote", "add", "origin", s.spark_recipe_repo], check=True)
    subprocess.run(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin", s.spark_recipe_sha], check=True)
    subprocess.run(["git", "-C", str(d), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    log(f"Spark recipe at {s.spark_recipe_sha[:8]} under {d}")


def suite(s: Settings, passthrough: list[str]) -> int:
    """python3 -m quality.suite --url <ours> --model <ours> ...; HumanEval needs its Docker sandbox, skipped unless asked."""
    ensure_clone(s)
    args = [sys.executable, "-m", "quality.suite", "--url", s.url, "--model", s.served_name]
    if "--only" not in passthrough and "--skip" not in passthrough:  # HumanEval's sandbox is a Docker image
        args += ["--skip", "humaneval"] + ([] if s.vision else ["vision"])
    if "--out" not in passthrough:
        args += ["--out", str(s.state_dir / "quality-out")]
    else:  # the suite runs from its clone: a relative --out means this recipe's directory
        i = passthrough.index("--out") + 1
        if i < len(passthrough):
            passthrough = passthrough[:i] + [str((ROOT / passthrough[i]).resolve())] + passthrough[i + 1:]
    return subprocess.run(args + passthrough, cwd=s.external_dir, env=_env()).returncode


def _env() -> dict[str, str]:
    """The suite downloads its datasets: python.org builds ship no CA certificates, macOS's own bundle serves."""
    env = dict(os.environ)
    if "SSL_CERT_FILE" not in env and os.path.isfile("/etc/ssl/cert.pem"):
        env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
    return env


def ruler(s: Settings, passthrough: list[str]) -> int:
    """python3 bench/ruler.py --url <ours> --model <ours> --concurrency 1 ...; --concurrency past 1 needs parallel > 1."""
    ensure_clone(s)
    args = [sys.executable, "bench/ruler.py", "--url", s.url, "--model", s.served_name]
    if "--concurrency" not in passthrough:
        args += ["--concurrency", "1"]
    return subprocess.run(args + passthrough, cwd=s.external_dir, env=_env()).returncode


def quality(s: Settings, label: str, humaneval: bool = False) -> int:
    """The recipe's standard measurement: every suite check and the decode ruler into evidence/<date>-<label>/.

    Long-context documents up to `context` nominal tokens (the suite sizes them nominally; GLM's tokenizer makes them
    ~20% shorter, so 128k runs in a 131k window); HumanEval only with its Docker sandbox; vision only with vision on.
    """
    out = ROOT / "evidence" / f"{time.strftime('%Y-%m-%d')}-{label}"
    if out.exists():
        log(f"{out} exists: pick another --label")
        return 2
    skip = [] if humaneval and shutil.which("docker") else ["humaneval"]
    if not s.vision:
        skip.append("vision")
    rc = suite(s, ["--long-max", str(s.context), "--out", str(out / "suite")] + (["--skip", *skip] if skip else []))
    (out / "ruler").mkdir(parents=True, exist_ok=True)
    rc2 = ruler(s, ["--runs", "3", "--max-tokens", "256", "--json", str(out / "ruler" / "ruler.json")])
    log(f"quality: {out / 'suite' / 'report.md'} and {out / 'ruler' / 'ruler.json'}")
    return rc or rc2
