"""A small on-disk HTTP cache.

The analyzer makes a lot of repeat requests -- the same index pages across
runs, the same sdists across projects -- and distro package indexes are tens of
megabytes. Everything goes through here.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from platformdirs import user_cache_dir

DEFAULT_TTL = 24 * 3600
LONG_TTL = 7 * 24 * 3600


def default_cache_dir() -> Path:
    env = os.environ.get("WILL_IT_RISCV_CACHE")
    if env:
        return Path(env).expanduser()
    return Path(user_cache_dir("will-it-riscv", "tactcomplabs"))


class Cache:
    """Key/value blob store with per-entry TTL. Never raises on cache trouble."""

    def __init__(self, root: Optional[Path] = None, enabled: bool = True):
        self.root = Path(root) if root else default_cache_dir()
        self.enabled = enabled
        if self.enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / digest[:2] / digest[2:]

    def get(self, namespace: str, key: str, ttl: int = DEFAULT_TTL) -> Optional[bytes]:
        if not self.enabled:
            return None
        path = self._path(namespace, key)
        try:
            stat = path.stat()
        except OSError:
            return None
        if ttl >= 0 and time.time() - stat.st_mtime > ttl:
            return None
        try:
            return path.read_bytes()
        except OSError:
            return None

    def put(self, namespace: str, key: str, value: bytes) -> None:
        if not self.enabled:
            return
        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(value)
            os.replace(tmp, path)
        except OSError:
            pass

    def get_json(self, namespace: str, key: str, ttl: int = DEFAULT_TTL):
        raw = self.get(namespace, key, ttl)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def put_json(self, namespace: str, key: str, value) -> None:
        try:
            self.put(namespace, key, json.dumps(value).encode("utf-8"))
        except (TypeError, ValueError):
            pass

    def memo(
        self,
        namespace: str,
        key: str,
        producer: Callable[[], bytes],
        ttl: int = DEFAULT_TTL,
    ) -> bytes:
        """Return the cached blob, or call ``producer`` and cache its result."""
        hit = self.get(namespace, key, ttl)
        if hit is not None:
            return hit
        value = producer()
        self.put(namespace, key, value)
        return value

    def clear(self) -> int:
        """Delete every cached file. Returns how many were removed."""
        if not self.root.exists():
            return 0
        removed = 0
        for path in sorted(self.root.rglob("*"), reverse=True):
            try:
                if path.is_file():
                    path.unlink()
                    removed += 1
                elif path.is_dir():
                    path.rmdir()
            except OSError:
                pass
        return removed
