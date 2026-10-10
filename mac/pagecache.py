"""Drop a checkpoint's pages from the file cache before a load, and measure how much of it is cached.

The engine reads its shards uncached (F_NOCACHE), but that does not evict pages an earlier download or read left in
the cache. With 100+ GiB of a 170 GiB checkpoint cached, macOS keeps those clean file pages and compresses the engine's
freshly filled GPU buffers instead (seen: 145 GiB in the compressor, critical pressure). msync(MS_INVALIDATE) over a
read-only shared mapping drops a file's cached pages without root, unlike `purge`.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
_PROT_READ, _MAP_SHARED, _MS_INVALIDATE = 1, 1, 2
_MAP_FAILED = ctypes.c_void_p(-1).value
PAGE = os.sysconf("SC_PAGE_SIZE")


def _shards(model_dir: Path) -> list[Path]:
    return sorted({p.resolve() for p in model_dir.glob("*.safetensors")})


def _map(path: Path, fn) -> int:
    n = path.stat().st_size
    if n == 0:
        return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, n, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr in (None, _MAP_FAILED):
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        try:
            return fn(addr, n)
        finally:
            _libc.munmap(addr, n)
    finally:
        os.close(fd)


def cached_bytes(model_dir: Path) -> int:
    def count(addr: int, n: int) -> int:
        vec = (ctypes.c_ubyte * ((n + PAGE - 1) // PAGE))()
        if _libc.mincore(addr, n, vec):
            return 0
        return sum(1 for v in vec if v & 1) * PAGE
    return sum(_map(p, count) for p in _shards(model_dir))


def drop(model_dir: Path) -> None:
    def invalidate(addr: int, n: int) -> int:
        if _libc.msync(addr, n, _MS_INVALIDATE):
            raise OSError(ctypes.get_errno(), "msync(MS_INVALIDATE)")
        return 0
    for p in _shards(model_dir):
        _map(p, invalidate)
