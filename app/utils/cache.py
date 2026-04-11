from __future__ import annotations

import time
from threading import Lock
from typing import Generic, TypeVar


T = TypeVar("T")


class SimpleTTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, T]] = {}
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            expires_at, value = entry
            if expires_at < now:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: T) -> None:
        expires_at = time.time() + self._ttl_seconds
        with self._lock:
            self._store[key] = (expires_at, value)
