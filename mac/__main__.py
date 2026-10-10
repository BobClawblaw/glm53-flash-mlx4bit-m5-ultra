"""python3 -m mac <command>: the recipe's one entry point (stdlib only, Python 3.11+)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import config, engine, external, memguard, serve, weights
from .log import fail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m mac", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="check the settings and print them; touches nothing")
    up = sub.add_parser("up", help="build the engine if needed, find or download the weights, start with the watchdog, wait for the API")
    up.add_argument("--no-download", action="store_true", help="refuse to download; serve what is cached")
    up.add_argument("--foreground", action="store_true", help="run the server as this process (launchd)")
    sub.add_parser("down", help="stop the server")
    sub.add_parser("status", help="pid, memory, /health, /v1/models")
    lg = sub.add_parser("logs", help="the server log's tail")
    lg.add_argument("--tail", type=int, default=80)
    b = sub.add_parser("build", help="worktree at the pinned sha, the patch, zig build native")
    b.add_argument("--rebuild", action="store_true")
    sub.add_parser("pull", help="download the checkpoint at the pinned revision unless a copy is cached")
    sub.add_parser("info", help="the engine's view of the checkpoint: family, format, memory floor")
    sub.add_parser("command", help="print the serve command line without starting anything")
    ch = sub.add_parser("chat", help="one greedy reply from the running server")
    ch.add_argument("prompt", nargs="?", default="Say hello in one sentence.")
    ch.add_argument("--max-tokens", type=int, default=128)
    ch.add_argument("--no-draft", action="store_true", help='send "draft": false (the plain reference)')
    see = sub.add_parser("see", help="one greedy reply about images from the running server (vision = true)")
    see.add_argument("images", nargs="+", type=Path, help="image files, then the question (default: describe them)")
    see.add_argument("--prompt", default="Describe this image in two sentences.")
    see.add_argument("--max-tokens", type=int, default=160)
    see.add_argument("--detail", choices=("auto", "low", "high"), default="auto")
    mg = sub.add_parser("memguard", help="(internal) watch a pid and kill it before the Mac swaps")
    mg.add_argument("pid", type=int)
    mg.add_argument("min_free_gib", type=float)
    mg.add_argument("max_swap_growth_mb", type=float)
    mg.add_argument("max_compressor_growth_gib", type=float)
    st = sub.add_parser("suite", help="the Spark recipe's quality suite against this server (extra args pass through)")
    st.add_argument("rest", nargs=argparse.REMAINDER)
    ru = sub.add_parser("ruler", help="the Spark recipe's decode ruler against this server (extra args pass through)")
    ru.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd == "memguard":
        return memguard.run(a.pid, a.min_free_gib, a.max_swap_growth_mb, a.max_compressor_growth_gib)
    try:
        s = config.load()
    except config.ConfigError as exc:
        print(f"settings: {exc}", file=sys.stderr)
        return 2
    try:
        if a.cmd == "validate":
            print(s.summary())
        elif a.cmd == "up":
            serve.up(s, download=not a.no_download, foreground=a.foreground)
        elif a.cmd == "down":
            serve.down(s)
        elif a.cmd == "status":
            print(serve.status(s))
        elif a.cmd == "logs":
            print(serve.tail(s, a.tail))
        elif a.cmd == "build":
            print(engine.ensure_binary(s, rebuild=a.rebuild))
        elif a.cmd == "pull":
            print(weights.ensure_snapshot(s))
        elif a.cmd == "info":
            print(engine.info(s, weights.ensure_snapshot(s, download=False)), end="")
        elif a.cmd == "command":
            cmd, _ = serve.command(s, download=False)
            print(" ".join(cmd))
        elif a.cmd == "chat":
            print(serve.chat(s, a.prompt, a.max_tokens, draft=False if a.no_draft else None))
        elif a.cmd == "see":
            missing = [str(p) for p in a.images if not p.is_file()]
            if missing:
                raise RuntimeError(f"no such image: {', '.join(missing)}")
            print(serve.chat(s, a.prompt, a.max_tokens, images=a.images, detail=a.detail))
        elif a.cmd == "suite":
            return external.suite(s, [w for w in a.rest if w != "--"])
        elif a.cmd == "ruler":
            return external.ruler(s, [w for w in a.rest if w != "--"])
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        fail(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
