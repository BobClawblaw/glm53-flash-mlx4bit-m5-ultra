"""The memory watchdog: kill the server before this Mac runs out. `python3 -m mac memguard PID MIN_FREE_GIB MAX_SWAP_GROWTH_MB`.

Unified memory has no GPU meter. Three signals kill after two readings in a row (4 s): free, inactive and speculative
pages under the floor; swap grown past a ceiling since the guard started; the compressor grown past a ceiling. The
model is wired (GLM_WIRE), so none of it can compress: compressor growth is other memory being squeezed, the path
that ended one unwired load in jetsam killing ~300 system daemons (145 GiB compressed, low swap). macOS's pressure
level at critical (kern.memorystatus_vm_pressure_level 4) kills only after CRITICAL_SECONDS, since a load shows it
for a reading or two while the kernel evicts file cache. Swap *free* is no signal: the swap file grows on demand.
SIGKILL, not SIGTERM (which would give the server time to page more in); the reason is logged, and the guard exits
when the server does.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time


def free_gib() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = 0
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)\.", out)
        pages += int(m.group(1)) if m else 0
    return pages * page / 2**30


def compressor_gib() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    m = re.search(r"Pages occupied by compressor:\s+(\d+)\.", out)
    return int(m.group(1)) * page / 2**30 if m else 0.0


def swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([0-9.]+)([MG])", out)
    if not m:
        return 0.0
    return float(m.group(1)) * (1024 if m.group(2) == "G" else 1)


def pressure_level() -> int:
    """macOS's memory pressure: 1 normal, 2 warning, 4 critical (0 if unreadable)."""
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True).stdout
    return int(out) if out.strip().isdigit() else 0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


CRITICAL_SECONDS = 30


def run(pid: int, min_free_gib: float, max_swap_growth_mb: float, max_compressor_growth_gib: float,
        interval: float = 2.0) -> int:
    base, base_compressed = swap_used_mb(), compressor_gib()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] memguard: watching pid {pid}, min free {min_free_gib:g} GiB, "
          f"max swap growth {max_swap_growth_mb:g} MB over {base:.0f} MB used", flush=True)
    low_readings, critical_readings, last_level = 0, 0, 1
    while alive(pid):
        free, grown, level = free_gib(), swap_used_mb() - base, pressure_level()
        compressed = compressor_gib() - base_compressed
        if level != last_level:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] memguard: pressure {last_level} -> {level}, free {free:.1f} GiB, "
                  f"swap grew {grown:.0f} MB", flush=True)
            last_level = level
        bad = free < min_free_gib or grown > max_swap_growth_mb or compressed > max_compressor_growth_gib
        low_readings = low_readings + 1 if bad else 0
        critical_readings = critical_readings + 1 if level >= 4 else 0
        if low_readings >= 2 or critical_readings * interval >= CRITICAL_SECONDS:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] memguard: free {free:.1f} GiB, pressure {level}, "
                  f"swap grew {grown:.0f} MB, compressor grew {compressed:.1f} GiB: killing pid {pid}", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return 3
        time.sleep(interval)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] memguard: pid {pid} gone, exiting", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])))
