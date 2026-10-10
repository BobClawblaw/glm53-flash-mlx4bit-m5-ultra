"""up / down / status / logs: one detached tensorfold-native process, a pid file, a memory watchdog, readiness on the API."""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import engine, memguard, pagecache, weights
from .config import Settings
from .log import log


def _get(url: str, timeout: float = 3) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, ConnectionError, OSError):
        return None


def pid(s: Settings) -> int | None:
    try:
        p = int(s.pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return p if memguard.alive(p) else None


def port_busy(s: Settings) -> bool:
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{s.port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
    return bool(out.strip())


def memory_check(s: Settings, info_text: str) -> None:
    m = re.search(r"memory floor:\s+([0-9.]+) GiB", info_text)
    floor = float(m.group(1)) if m else 0.0
    have = memguard.free_gib()
    if floor and have < floor + s.memory_margin_gib:
        raise RuntimeError(f"{have:.1f} GiB free, the engine's floor is {floor:.1f} GiB plus a {s.memory_margin_gib:g} GiB "
                           "margin (memory_margin_gib): close something or lower the margin")
    log(f"Memory: {have:.1f} GiB free, floor {floor:.1f} GiB, engine load limit {s.memory_limit_gb} GB")


def tail_memguard(s: Settings) -> str:
    try:
        return s.memguard_log.read_text().strip().splitlines()[-1]
    except (OSError, IndexError):
        return ""


def wait_ready(s: Settings, proc: subprocess.Popen, limit: int = 3600) -> None:
    log(f"Waiting for {s.url}/v1/models (log: {s.log_file})")
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:                 # reaped here: a killed child stays a zombie that kill(0) sees
            if "killing" in tail_memguard(s):
                raise RuntimeError(f"memguard killed the server: {tail_memguard(s)}")
            raise RuntimeError(f"the server exited; last lines of {s.log_file}:\n{tail(s, 30)}")
        models = _get(f"{s.url}/v1/models")
        if models and any(m.get("id") == s.served_name for m in models.get("data", [])):
            log(f"Ready in {time.time() - t0:.0f}s: {s.url}/v1 serves {s.served_name}")
            return
        time.sleep(2)
    raise RuntimeError(f"not ready after {limit}s; see {s.log_file}")


def command(s: Settings, download: bool = True) -> tuple[list[str], Path]:
    binary = engine.ensure_binary(s)
    snap = weights.ensure_snapshot(s, download=download)
    info_text = engine.info(s, snap)
    log("Engine: " + " ".join(line.strip() for line in info_text.strip().splitlines()))
    memory_check(s, info_text)
    return [str(binary)] + s.serve_args(snap), snap


def drop_cached_checkpoint(snap: Path) -> None:
    """The checkpoint's cached pages out of the file cache, so the load does not push the GPU copy into the compressor."""
    cached = pagecache.cached_bytes(snap)
    if cached >= 1 << 30:
        pagecache.drop(snap)
        log(f"Dropped {cached / 2**30:.1f} GiB of the checkpoint from the file cache "
            f"({pagecache.cached_bytes(snap) / 2**30:.1f} GiB left)")


def start_memguard(s: Settings, server_pid: int) -> None:
    if not s.memguard:
        log("memguard off (memguard = false)")
        return
    with open(s.memguard_log, "ab") as logf:
        subprocess.Popen([sys.executable, "-m", "mac.memguard", str(server_pid), str(s.memguard_min_free_gib),
                          str(s.memguard_max_swap_growth_mb), str(s.memguard_max_compressor_growth_gib)], cwd=Path(__file__).resolve().parents[1],
                         stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"memguard watching pid {server_pid}: kill under {s.memguard_min_free_gib:g} GiB free or "
        f"{s.memguard_max_swap_growth_mb} MB swap growth or {s.memguard_max_compressor_growth_gib:g} GiB compressor growth, or "
        f"{memguard.CRITICAL_SECONDS} s at critical pressure ({s.memguard_log})")


RECENT_KILL_S = 15 * 60


def recent_kill(s: Settings) -> str | None:
    """memguard's last kill if it was under 15 minutes ago: launchd's KeepAlive must not retry a load into the same wall."""
    line = tail_memguard(s)
    m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] memguard: .*killing", line)
    if not m:
        return None
    age = time.time() - time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
    return line if 0 <= age < RECENT_KILL_S else None


def up(s: Settings, *, download: bool = True, foreground: bool = False) -> None:
    killed = recent_kill(s)
    if killed and os.environ.get("MEMGUARD_RETRY") != "1":
        raise RuntimeError(f"memguard killed the last server under {RECENT_KILL_S // 60} minutes ago ({killed}); find what "
                           "holds memory first, or set MEMGUARD_RETRY=1 to try again now")
    if pid(s):
        raise RuntimeError(f"already running (pid {pid(s)}): python3 -m mac down first")
    if port_busy(s):
        raise RuntimeError(f"port {s.port} is in use by another process")
    cmd, snap = command(s, download=download)
    drop_cached_checkpoint(snap)
    s.tf_cache.mkdir(parents=True, exist_ok=True)
    s.state_dir.mkdir(exist_ok=True)
    env = s.server_env()
    log("Starting: " + " ".join(cmd))
    if foreground:                     # launchd: the server is the job; the guard watches this pid
        start_memguard(s, os.getpid())
        os.execve(cmd[0], cmd, env)
    with open(s.log_file, "ab") as logf:
        logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                             start_new_session=True)
    s.pid_file.write_text(str(p.pid))
    start_memguard(s, p.pid)
    try:
        wait_ready(s, p)
    except RuntimeError:
        if p.poll() is not None:
            s.pid_file.unlink(missing_ok=True)
        raise


def down(s: Settings, grace: float = 30) -> None:
    p = pid(s)
    if not p:
        log("not running")
        s.pid_file.unlink(missing_ok=True)
        return
    log(f"Stopping pid {p}")
    os.kill(p, signal.SIGTERM)
    t0 = time.time()
    while pid(s) and time.time() - t0 < grace:
        time.sleep(0.5)
    if pid(s):
        log(f"still up after {grace:.0f}s, SIGKILL")
        os.kill(p, signal.SIGKILL)
        time.sleep(1)
    s.pid_file.unlink(missing_ok=True)
    log("stopped")


def status(s: Settings) -> str:
    p = pid(s)
    lines = [f"pid: {p or 'not running'}", f"url: {s.url}/v1",
             f"memory: {memguard.free_gib():.1f} GiB free, {memguard.swap_used_mb():.0f} MB swap used"]
    health = _get(f"{s.url}/health")
    models = _get(f"{s.url}/v1/models")
    lines.append(f"health: {json.dumps(health) if health is not None else 'no answer'}")
    lines.append("models: " + (", ".join(m.get("id", "?") for m in models.get("data", [])) if models else "no answer"))
    if s.dashboard:
        stats = _get(f"{s.url}/stats")
        if stats:
            lines.append(f"stats: {json.dumps(stats)[:400]}")
    return "\n".join(lines)


def tail(s: Settings, n: int) -> str:
    try:
        lines = s.log_file.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return f"(no {s.log_file})"
    return "\n".join(lines[-n:])


def chat(s: Settings, prompt: str, max_tokens: int = 128, draft: bool | None = None,
         images: list[Path] | None = None, detail: str = "auto") -> str:
    """One greedy chat completion, the smoke test after `up`; draft=False asks for the plain reference. `images`: sent
    as data: URLs before the prompt (vision = true), as OpenAI image_url parts."""
    content: str | list = prompt
    if images:
        content = [{"type": "image_url", "image_url": {"url": data_url(p), "detail": detail}} for p in images]
        content.append({"type": "text", "text": prompt})
    body = {"model": s.served_name, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0}
    if draft is not None:
        body["draft"] = draft
    req = urllib.request.Request(f"{s.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if s.api_key:
        req.add_header("Authorization", f"Bearer {s.api_key}")
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')}") from None
    return json.dumps(out, indent=1)


def data_url(path: Path) -> str:
    kind = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif", ".webp": "webp", ".heic": "heic"}.get(
        path.suffix.lower(), "octet-stream")
    return f"data:image/{kind};base64," + base64.b64encode(path.read_bytes()).decode()
