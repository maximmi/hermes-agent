"""Proactive client-side pacing to avoid provider 429s.

Two independent, in-process throttle dimensions per ``(provider, model)``:
requests-per-minute (``RateLimiter``) and input-tokens-per-minute
(``TokenBucketLimiter``). See ``hermes_cli.rate_limits`` for the config
resolvers (``providers.<id>.requests_per_minute`` /
``providers.<id>.input_tokens_per_minute``, each with a per-model override;
unset/<=0 disables that dimension, no overhead).

Complements the *reactive* breaker in ``agent.nous_rate_guard`` (which
waits for a real 429 before acting, and is cross-session): this module
tries to avoid tripping the limit in the first place, in-process only.

Design note (PR #17749 review): every limiter's lock guards only the O(1)
bookkeeping — never a sleep. A caller that computes a long wait returns
immediately; it sleeps *after* releasing the lock, so it never blocks
another thread's scheduling.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from hermes_cli.rate_limits import (
    get_provider_input_tokens_per_minute,
    get_provider_requests_per_minute,
)

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
        self._recent_calls: "deque[float]" = deque()

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
            self._recent_calls.append(now)
        return start - now

    def current_requests(self, now: Optional[float] = None) -> int:
        """Count of calls to ``wait_time`` in the trailing 60s window — a
        live usage figure for display, independent of the scheduler above."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            while self._recent_calls and now - self._recent_calls[0] > 60.0:
                self._recent_calls.popleft()
            return len(self._recent_calls)


class TokenBucketLimiter:
    """Leaky bucket over confirmed input-token usage: ``capacity`` tokens
    drain at ``capacity / 60`` tokens/sec. ``record_usage`` is fed actual
    (provider-confirmed) token counts after a response; ``wait_time`` gates
    the *next* request, optionally projecting a rough pre-send estimate on
    top of the confirmed backlog (without ever permanently reserving it —
    only real usage changes the bucket's level).

    The estimate's contribution is clamped to ``capacity``: a single
    request whose estimate alone exceeds the whole per-minute budget cannot
    be satisfied by waiting (the provider will count it against the window
    the instant it's sent, regardless), so letting an oversized estimate
    inflate the wait further would just be an unbounded, pointless delay.
    """

    def __init__(self, capacity: float) -> None:
        self._lock = threading.Lock()
        self._capacity = capacity
        self._rate = capacity / 60.0
        self._level = 0.0
        self._last_update: Optional[float] = None

    def update_capacity(self, capacity: float) -> None:
        with self._lock:
            self._capacity = capacity
            self._rate = capacity / 60.0

    def _drain_locked(self, now: float) -> None:
        if self._last_update is not None:
            elapsed = now - self._last_update
            if elapsed > 0:
                self._level = max(0.0, self._level - elapsed * self._rate)
        self._last_update = now

    def record_usage(self, tokens: float, now: Optional[float] = None) -> None:
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._drain_locked(now)
            self._level += max(0.0, tokens)

    def wait_time(self, estimated_tokens: float = 0, now: Optional[float] = None) -> float:
        """Never sleeps itself — the lock is held only for this bookkeeping."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._drain_locked(now)
            effective_estimate = min(max(0.0, estimated_tokens), self._capacity)
            projected = self._level + effective_estimate
            if projected <= self._capacity:
                return 0.0
            return (projected - self._capacity) / self._rate

    def current_level(self, now: Optional[float] = None) -> float:
        """Current bucket fill after draining — a live "tokens used this
        minute" figure for display."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._drain_locked(now)
            return self._level


_limiters: Dict[Tuple[str, str], RateLimiter] = {}
_registry_lock = threading.Lock()

_token_limiters: Dict[Tuple[str, str], TokenBucketLimiter] = {}
_token_registry_lock = threading.Lock()


def _get_limiter(provider: str, model: str, requests_per_minute: float) -> RateLimiter:
    key = (provider, model)
    with _registry_lock:
        limiter = _limiters.get(key)
        if limiter is None:
            limiter = _limiters[key] = RateLimiter(requests_per_minute)
        else:
            limiter.update_rate(requests_per_minute)
        return limiter


def _get_token_limiter(provider: str, model: str, capacity: float) -> TokenBucketLimiter:
    key = (provider, model)
    with _token_registry_lock:
        limiter = _token_limiters.get(key)
        if limiter is None:
            limiter = _token_limiters[key] = TokenBucketLimiter(capacity)
        else:
            limiter.update_capacity(capacity)
        return limiter


def _sum_string_lengths(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_sum_string_lengths(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_sum_string_lengths(v) for v in value)
    return 0


def _estimate_request_tokens(request: Any) -> int:
    """Best-effort, provider-agnostic size estimate: total characters across
    every string leaf in the request payload, divided by ~4 (a rough
    chars-per-token heuristic). Never raises; 0 on any failure or when
    ``request`` is falsy."""
    if not request:
        return 0
    try:
        return _sum_string_lengths(request) // 4
    except Exception:
        return 0


@dataclass(frozen=True)
class RateLimitStatus:
    """Live usage for one (provider, model), for status-bar display. Either
    dimension is ``None`` when that dimension isn't configured."""

    rpm_used: Optional[int]
    rpm_limit: Optional[float]
    tpm_used: Optional[float]
    tpm_limit: Optional[float]


def status_snapshot(provider: str, model: str) -> Optional[RateLimitStatus]:
    """Live RPM/TPM usage for ``(provider, model)``, or ``None`` when
    neither throttle dimension is configured for it (nothing to show).
    Never triggers a request — a configured-but-idle pair reports 0 used."""
    requests_per_minute = get_provider_requests_per_minute(provider, model)
    input_tokens_per_minute = get_provider_input_tokens_per_minute(provider, model)
    if requests_per_minute is None and input_tokens_per_minute is None:
        return None

    rpm_used: Optional[int] = None
    if requests_per_minute is not None:
        rpm_used = _get_limiter(provider, model, requests_per_minute).current_requests()

    tpm_used: Optional[float] = None
    if input_tokens_per_minute is not None:
        tpm_used = _get_token_limiter(provider, model, input_tokens_per_minute).current_level()

    return RateLimitStatus(
        rpm_used=rpm_used, rpm_limit=requests_per_minute,
        tpm_used=tpm_used, tpm_limit=input_tokens_per_minute,
    )


def record_input_tokens(provider: str, model: str, tokens: float) -> None:
    """Feed actual (provider-confirmed) input-token usage into the TPM
    bucket for ``(provider, model)``. No-op when
    ``input_tokens_per_minute`` is unconfigured for this pair."""
    capacity = get_provider_input_tokens_per_minute(provider, model)
    if capacity is None:
        return
    _get_token_limiter(provider, model, capacity).record_usage(tokens)


def throttle_before_request(agent: Any, provider: str, model: str, request: Any = None) -> None:
    """Pace this outbound request against the configured RPM and/or
    input-tokens-per-minute ceilings for ``(provider, model)``. No-op on
    whichever dimension is unconfigured (default: both, zero overhead)."""
    delay = 0.0

    requests_per_minute = get_provider_requests_per_minute(provider, model)
    if requests_per_minute is not None:
        rpm_delay = _get_limiter(provider, model, requests_per_minute).wait_time()
        logger.debug(
            "%s/%s: pacing to %.3g req/min -> %.3fs wait", provider, model, requests_per_minute, rpm_delay,
        )
        delay = max(delay, rpm_delay)

    input_tokens_per_minute = get_provider_input_tokens_per_minute(provider, model)
    if input_tokens_per_minute is not None:
        estimated = _estimate_request_tokens(request)
        if estimated > input_tokens_per_minute:
            logger.warning(
                "%s/%s: estimated input (~%d tokens) exceeds the configured "
                "input_tokens_per_minute ceiling (%g); sending anyway — the "
                "provider may still rate-limit this call.",
                provider, model, estimated, input_tokens_per_minute,
            )
            if agent is not None:
                try:
                    agent._vprint(
                        f"{getattr(agent, 'log_prefix', '')}⚠️ Estimated input (~{estimated:,} tokens) exceeds "
                        f"the configured input_tokens_per_minute limit ({input_tokens_per_minute:g}) for "
                        f"{provider}/{model} — sending anyway.",
                        force=True,
                    )
                except Exception:
                    pass
        tpm_delay = _get_token_limiter(provider, model, input_tokens_per_minute).wait_time(
            estimated_tokens=estimated,
        )
        logger.debug(
            "%s/%s: pacing to %.3g input tok/min (est=%d) -> %.3fs wait",
            provider, model, input_tokens_per_minute, estimated, tpm_delay,
        )
        delay = max(delay, tpm_delay)

    if delay <= 0:
        return
    if delay > _NOTICE_THRESHOLD_SECONDS and agent is not None:
        try:
            agent._vprint(
                f"{getattr(agent, 'log_prefix', '')}⏳ Pacing requests to "
                f"{provider}/{model} ({delay:.1f}s)...",
                force=True,
            )
        except Exception:
            pass
    time.sleep(delay)
