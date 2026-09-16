"""Proactive client-side pacing to avoid provider 429s.

Spaces outbound requests to a configured requests-per-minute ceiling,
in-process, per ``(provider, model)``. See ``hermes_cli.rate_limits`` for
the config resolver (``providers.<id>.requests_per_minute`` /
``providers.<id>.models.<model>.requests_per_minute``; unset/<=0 disables
this, no overhead).

Complements the *reactive* breaker in ``agent.nous_rate_guard`` (which
waits for a real 429 before acting, and is cross-session): this module
tries to avoid tripping the limit in the first place, in-process only.

Design note (PR #17749 review): ``RateLimiter._lock`` guards only the O(1)
slot bookkeeping — never the sleep. A caller that reserves a far-future slot
returns immediately; it sleeps *after* releasing the lock, so it never
blocks another thread's scheduling.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

from hermes_cli.rate_limits import get_provider_requests_per_minute

logger = logging.getLogger(__name__)

_NOTICE_THRESHOLD_SECONDS = 5.0


class RateLimiter:
    """Spaces calls to at most ``requests_per_minute`` per minute.

    Single-slot scheduling (no burst credit banked for a caller that comes
    back late) — simple and enough to keep steady traffic under a ceiling.
    """

    def __init__(self, requests_per_minute: float) -> None:
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._interval = 60.0 / requests_per_minute

    def update_rate(self, requests_per_minute: float) -> None:
        with self._lock:
            self._interval = 60.0 / requests_per_minute

    def wait_time(self, now: Optional[float] = None) -> float:
        """Reserve this caller's slot and return how long it should wait.
        Never sleeps itself — the lock is held only for this bookkeeping."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            start = max(now, self._next_slot)
            self._next_slot = start + self._interval
        return start - now


_limiters: Dict[Tuple[str, str], RateLimiter] = {}
_registry_lock = threading.Lock()


def _get_limiter(provider: str, model: str, requests_per_minute: float) -> RateLimiter:
    key = (provider, model)
    with _registry_lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = _limiters[key] = RateLimiter(requests_per_minute)
        else:
            limiter.update_rate(requests_per_minute)
        return limiter


def throttle_before_request(agent: Any, provider: str, model: str) -> None:
    """Pace this outbound request against the configured RPM ceiling for
    ``(provider, model)``. No-op when unconfigured (default)."""
    requests_per_minute = get_provider_requests_per_minute(provider, model)
    if requests_per_minute is None:
        return
    delay = _get_limiter(provider, model, requests_per_minute).wait_time()
    logger.debug(
        "%s/%s: pacing to %.3g req/min -> %.3fs wait", provider, model, requests_per_minute, delay,
    )
    if delay <= 0:
        return
    if delay > _NOTICE_THRESHOLD_SECONDS and agent is not None:
        try:
            agent._vprint(
                f"{getattr(agent, 'log_prefix', '')}⏳ Pacing requests to "
                f"{provider}/{model} ({delay:.1f}s, limit={requests_per_minute:g}/min)...",
                force=True,
            )
        except Exception:
            pass
    time.sleep(delay)
