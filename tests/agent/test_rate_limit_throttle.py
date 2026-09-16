"""Tests for agent/rate_limit_throttle.py — proactive client-side RPM pacing.

Complements the reactive breaker in agent/nous_rate_guard.py: this module
spaces outbound requests *before* the provider ever sees them, so it never
needs to observe a 429 to act.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _reset_limiter_registry():
    """The module-level limiter registry is a process-wide singleton keyed
    by (provider, model); without resetting it, one test's schedule leaks
    into the next test that happens to reuse the same key."""
    import agent.rate_limit_throttle as mod

    mod._limiters.clear()
    yield
    mod._limiters.clear()


class TestRateLimiterScheduling:
    def test_first_call_never_waits(self):
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)  # 1s interval
        assert limiter.wait_time(now=100.0) == 0.0

    def test_second_call_waits_out_the_interval(self):
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)  # 1s interval
        assert limiter.wait_time(now=0.0) == 0.0
        assert limiter.wait_time(now=0.1) == pytest.approx(0.9)

    def test_a_slow_caller_is_not_penalized_or_allowed_to_burst(self):
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)  # 1s interval
        limiter.wait_time(now=0.0)
        limiter.wait_time(now=0.1)  # reserves slot at t=1.0
        # Caller only comes back long after its reserved slot: no wait, and
        # no credit is banked for a burst — the next slot starts from now.
        assert limiter.wait_time(now=5.0) == 0.0

    def test_update_rate_changes_interval_without_resetting_schedule(self):
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)  # 1s interval
        limiter.wait_time(now=0.0)  # next slot at t=1.0
        limiter.update_rate(120)  # 0.5s interval
        assert limiter.wait_time(now=0.0) == pytest.approx(1.0)

    def test_scheduling_never_sleeps_regardless_of_interval_size(self):
        """The lock must be held only for the O(1) bookkeeping, never across
        a sleep — otherwise one slow caller would block every other thread's
        scheduling for the full interval (see PR #17749 review comment)."""
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=1)  # 60s interval
        start = time.monotonic()
        delays = [limiter.wait_time() for _ in range(5)]
        elapsed = time.monotonic() - start

        assert elapsed < 1.0  # would be ~240s if wait_time() ever slept
        assert delays[0] == pytest.approx(0.0, abs=0.05)
        assert delays[4] == pytest.approx(240.0, abs=0.5)


class TestRateLimiterRegistry:
    def test_same_key_reuses_the_limiter_instance(self):
        from agent.rate_limit_throttle import _get_limiter

        a = _get_limiter("nim", "minimax", 40)
        b = _get_limiter("nim", "minimax", 40)
        assert a is b

    def test_different_keys_get_independent_limiters(self):
        from agent.rate_limit_throttle import _get_limiter

        a = _get_limiter("nim", "minimax", 40)
        b = _get_limiter("nim", "other-model", 40)
        assert a is not b


class TestThrottleBeforeRequest:
    def test_noop_when_unconfigured(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: None)
        sleeps = []
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        mod.throttle_before_request(agent=None, provider="openrouter", model="anything")

        assert sleeps == []

    def test_sleeps_for_the_scheduled_delay(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: 60.0)
        limiter = mod.RateLimiter(60.0)
        monkeypatch.setattr(mod, "_get_limiter", lambda *_a, **_kw: limiter)
        sleeps = []
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        mod.throttle_before_request(agent=None, provider="nim", model="minimax")  # delay=0 -> no sleep() call at all
        mod.throttle_before_request(agent=None, provider="nim", model="minimax")  # waits out the ~1s interval

        assert sleeps == [pytest.approx(1.0, abs=0.05)]

    def test_notifies_when_wait_exceeds_five_seconds(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: 1.0)  # 60s interval
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        notices = []
        agent = SimpleNamespace(log_prefix="", _vprint=lambda *a, **kw: notices.append(a[0]))

        mod.throttle_before_request(agent, "nim", "minimax")  # t~0, no wait -> no notice
        assert notices == []
        mod.throttle_before_request(agent, "nim", "minimax")  # scheduled ~60s out -> notice
        assert len(notices) == 1
        assert "nim/minimax" in notices[0]

    def test_silent_below_five_second_threshold(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: 60.0)  # 1s interval
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        notices = []
        agent = SimpleNamespace(log_prefix="", _vprint=lambda *a, **kw: notices.append(a[0]))

        mod.throttle_before_request(agent, "nim", "minimax")
        mod.throttle_before_request(agent, "nim", "minimax")  # ~1s wait, below the 5s threshold

        assert notices == []

    def test_every_call_logs_at_debug_level_for_verification(self, monkeypatch, caplog):
        """A configured-but-quiet throttle (small delay, below the user-visible
        notice threshold) must still be observable via --verbose/debug logs —
        otherwise there's no way to confirm it's active at realistic RPM values."""
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: 6000.0)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)

        with caplog.at_level("DEBUG", logger="agent.rate_limit_throttle"):
            mod.throttle_before_request(None, "nim", "minimax")

        assert any("nim" in r.message and "minimax" in r.message for r in caplog.records)


class TestConcurrentAcquisition:
    def test_concurrent_wait_time_calls_do_not_deadlock_or_violate_spacing(self):
        """N threads racing for the same key must each get a distinct,
        properly-spaced slot, and the scheduling itself must complete fast
        (proving the lock isn't held across anyone's sleep)."""
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=6000)  # 0.01s interval
        n = 20
        results: list[float] = []
        results_lock = threading.Lock()

        def worker():
            now = time.monotonic()
            delay = limiter.wait_time(now=now)
            with results_lock:
                results.append(now + delay)  # this thread's absolute reserved slot

        threads = [threading.Thread(target=worker) for _ in range(n)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        elapsed = time.monotonic() - start

        assert all(not t.is_alive() for t in threads)  # no deadlock
        assert elapsed < 1.0  # scheduling itself is O(1) per call, not serialized on sleep
        assert len(results) == n
        # The safety property under test: no two threads were ever handed
        # slots closer together than the configured interval, regardless of
        # real-world OS scheduling jitter in when each thread actually got
        # to call wait_time() (which shifts absolute delays but must never
        # shrink the *gap* between consecutive slots below the interval).
        sorted_slots = sorted(results)  # absolute monotonic slot per thread
        assert sorted_slots[0] - start == pytest.approx(0.0, abs=0.05)
        gaps = [b - a for a, b in zip(sorted_slots, sorted_slots[1:])]
        assert all(gap >= 0.01 - 1e-6 for gap in gaps)
