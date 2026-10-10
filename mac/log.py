"""One timestamped line at a time, flushed, so launchd and terminals see progress."""

from __future__ import annotations

import sys
import time


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] failed: {msg}", file=sys.stderr, flush=True)
