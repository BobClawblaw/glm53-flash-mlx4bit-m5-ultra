"""The Spark recipe's quality suite and decode ruler, run from a pinned clone under .external/ (never copied here)."""

from __future__ import annotations

import json
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


def suite(s: Settings, passthrough: list[str], thinking_budget: int = 0) -> int:
    """python3 -m quality.suite --url <ours> --model <ours> ...; HumanEval needs its Docker sandbox, skipped unless asked.

    `thinking_budget` > 0: through tools/suite_thinking.py, every request with thinking on and that much reasoning room.
    """
    ensure_clone(s)
    if thinking_budget > 0:
        args = [sys.executable, str(ROOT / "tools" / "suite_thinking.py"), str(thinking_budget), "--url", s.url, "--model", s.served_name]
    else:
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


def quality(s: Settings, label: str, humaneval: bool = False, thinking_budget: int = 0, resume: bool = False) -> int:
    """The recipe's standard measurement: every suite check and the decode ruler into evidence/<date>-<label>/.

    Long-context documents up to `context` nominal tokens (the suite sizes them nominally; GLM's tokenizer makes them
    ~20% shorter, so 128k runs in a 131k window); HumanEval only with its Docker sandbox; vision only with vision on.
    `resume` continues the newest evidence/*-<label>/: checks with a saved result and a saved ruler are not rerun, and
    the report is rebuilt from every saved check.
    """
    if resume:
        found = sorted((ROOT / "evidence").glob(f"*-{label}"))
        if not found:
            log(f"no evidence/*-{label} to resume")
            return 2
        out = found[-1]
    else:
        out = ROOT / "evidence" / f"{time.strftime('%Y-%m-%d')}-{label}"
        if out.exists():
            log(f"{out} exists: pick another --label, or --resume it")
            return 2
    skip = [] if humaneval and shutil.which("docker") else ["humaneval"]
    if not s.vision:
        skip.append("vision")
    done = sorted(p.stem for p in (out / "suite").glob("*.json") if p.stem != "report") if resume else []
    if done:
        log(f"resuming {out.name}: keeping {', '.join(done)}")
    rc = suite(s, ["--long-max", str(s.context), "--out", str(out / "suite"), "--skip", *skip, *done], thinking_budget)
    if resume:
        _report(out / "suite")
    rc2 = 0
    if not (resume and (out / "ruler" / "ruler.json").is_file()):
        (out / "ruler").mkdir(parents=True, exist_ok=True)
        rc2 = ruler(s, ["--runs", "3", "--max-tokens", "256", "--json", str(out / "ruler" / "ruler.json")])
    log(f"quality: {out / 'suite' / 'report.md'} and {out / 'ruler' / 'ruler.json'}")
    return rc or rc2


def _report(d) -> None:
    """report.md and report.json from every check's own file, in the order they ran (as the suite writes them)."""
    results = []
    for p in sorted((p for p in d.glob("*.json") if p.stem != "report"), key=lambda p: p.stat().st_mtime):
        r = json.loads(p.read_text())
        results += r if isinstance(r, list) else [r]
    lines = ["| Check | Passed | Score | Time | Note |", "|---|---:|---:|---:|---|"]
    for r in results:
        note = r["notes"][0] if r["notes"] else ""
        lines.append(f"| {r['name']} | {r['passed']}/{r['cases']} | {100 * r['score']:.1f}% | {r['seconds']:.0f}s | {note} |")
    (d / "report.md").write_text("\n".join(lines) + "\n")
    old = json.loads((d / "report.json").read_text()) if (d / "report.json").is_file() else {}
    (d / "report.json").write_text(json.dumps({**{k: v for k, v in old.items() if k != "results"}, "results": results}, indent=1))
