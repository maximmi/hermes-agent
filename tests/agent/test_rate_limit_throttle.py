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
    """The module-level limiter registries are process-wide singletons keyed
    by (provider, model); without resetting them, one test's schedule leaks
    into the next test that happens to reuse the same key."""
    import agent.rate_limit_throttle as mod

    mod._limiters.clear()
    mod._token_limiters.clear()
    yield
    mod._limiters.clear()
    mod._token_limiters.clear()


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
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: None)
        sleeps = []
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        mod.throttle_before_request(agent=None, provider="openrouter", model="anything")

        assert sleeps == []

    def test_sleeps_for_the_scheduled_delay(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: 60.0)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: None)
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
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: None)
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
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: None)
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
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: None)
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


class TestTokenBucketLimiter:
    def test_first_call_has_zero_wait(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1000)  # rate = 1000/60 tok/s
        assert limiter.wait_time(now=0.0) == 0.0

    def test_recorded_usage_pushes_the_bucket_over_capacity(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1000)
        limiter.record_usage(1500, now=0.0)  # 500 over capacity
        # rate = 1000/60 tok/s -> draining the 500 overage takes 30s
        assert limiter.wait_time(now=0.0) == pytest.approx(30.0)

    def test_bucket_drains_linearly_over_time(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1000)
        limiter.record_usage(1500, now=0.0)  # level=1500, 500 over
        # Halfway through the 30s drain: 250 tokens have leaked out.
        assert limiter.wait_time(now=15.0) == pytest.approx(15.0)
        assert limiter.wait_time(now=30.0) == pytest.approx(0.0)
        assert limiter.wait_time(now=60.0) == pytest.approx(0.0)  # never goes negative

    def test_estimated_tokens_included_in_the_projection(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1000)
        # Empty bucket, but this request alone would use 700 of the 1000 budget.
        assert limiter.wait_time(estimated_tokens=700, now=0.0) == 0.0
        # Two such requests back to back (bucket still empty since estimate
        # is never permanently reserved) both see zero wait...
        assert limiter.wait_time(estimated_tokens=700, now=0.0) == 0.0
        # ...but once actual usage lands, the next estimate is projected on top of it.
        limiter.record_usage(700, now=0.0)
        assert limiter.wait_time(estimated_tokens=700, now=0.0) == pytest.approx(24.0)  # (700+700-1000)/(1000/60)

    def test_estimate_beyond_capacity_is_clamped_not_unbounded(self):
        """A single request whose rough estimate alone exceeds the whole
        per-minute budget cannot be satisfied by waiting — clamp the
        estimate's contribution to `capacity` so the wait is bounded by the
        real (confirmed) backlog only, not by how wildly the estimate
        overshoots."""
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1000)
        limiter.record_usage(500, now=0.0)  # some real backlog
        wait = limiter.wait_time(estimated_tokens=500_000, now=0.0)
        # Clamped: effective estimate = capacity (1000) -> projected = 1500 -> wait = 500/(1000/60) = 30s
        assert wait == pytest.approx(30.0)
        assert wait < 60.0  # bounded regardless of how large the estimate is

    def test_update_capacity_changes_rate_without_resetting_level(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1000)
        limiter.record_usage(1500, now=0.0)  # 500 over capacity=1000
        limiter.update_capacity(2000)  # level (1500) unchanged, but now under the new capacity
        assert limiter.wait_time(now=0.0) == 0.0

    def test_wait_time_never_sleeps_regardless_of_backlog_size(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1)  # tiny capacity -> huge backlog possible
        limiter.record_usage(1_000_000, now=0.0)
        start = time.monotonic()
        wait = limiter.wait_time(now=0.0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # never actually sleeps inside wait_time()
        assert wait > 1000  # (a huge, but computed-not-slept, wait)


class TestTokenLimiterRegistry:
    def test_same_key_reuses_the_limiter_instance(self):
        from agent.rate_limit_throttle import _get_token_limiter

        a = _get_token_limiter("nim", "minimax", 1000)
        b = _get_token_limiter("nim", "minimax", 1000)
        assert a is b

    def test_different_keys_get_independent_limiters(self):
        from agent.rate_limit_throttle import _get_token_limiter

        a = _get_token_limiter("nim", "minimax", 1000)
        b = _get_token_limiter("nim", "other-model", 1000)
        assert a is not b


class TestEstimateRequestTokens:
    def test_sums_string_lengths_across_nested_structure(self):
        from agent.rate_limit_throttle import _estimate_request_tokens

        request = {
            "model": "x",  # 1 char
            "messages": [
                {"role": "user", "content": "a" * 40},
                {"role": "assistant", "content": "b" * 60},
            ],
        }
        # "x"(1) + "user"(4)+40 + "assistant"(9)+60 = 1+4+40+9+60 = 114 -> //4 = 28
        assert _estimate_request_tokens(request) == 28

    def test_ignores_non_string_leaves(self):
        from agent.rate_limit_throttle import _estimate_request_tokens

        assert _estimate_request_tokens({"n": 12345, "flag": True, "x": None}) == 0

    def test_empty_or_none_request_is_zero(self):
        from agent.rate_limit_throttle import _estimate_request_tokens

        assert _estimate_request_tokens(None) == 0
        assert _estimate_request_tokens({}) == 0

    def test_never_raises_on_unexpected_shapes(self):
        from agent.rate_limit_throttle import _estimate_request_tokens

        class Weird:
            def __iter__(self):
                raise RuntimeError("boom")

        assert _estimate_request_tokens(Weird()) == 0


class TestRecordInputTokens:
    def test_noop_when_unconfigured(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: None)
        mod.record_input_tokens("openrouter", "anything", 99999)  # must not raise, no limiter created

        assert mod._token_limiters == {}

    def test_feeds_the_bucket_for_the_configured_pair(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: 1000.0)
        mod.record_input_tokens("nim", "minimax", 1500)

        limiter = mod._token_limiters[("nim", "minimax")]
        assert limiter.wait_time(now=0.0) == pytest.approx(30.0)


class TestThrottleBeforeRequestTokens:
    def test_tpm_only_wait_when_rpm_unconfigured(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: None)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: 1000.0)
        limiter = mod.TokenBucketLimiter(1000.0)
        limiter.record_usage(1500)  # pre-loaded 500 over capacity (real clock: throttle_before_request reads it moments later)
        monkeypatch.setattr(mod, "_get_token_limiter", lambda *_a, **_kw: limiter)
        sleeps = []
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        mod.throttle_before_request(None, "nim", "minimax", request={})

        assert sleeps == [pytest.approx(30.0, abs=0.5)]

    def test_combined_delay_is_the_max_of_both_dimensions(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: 60.0)  # 1s interval
        rpm_limiter = mod.RateLimiter(60.0)
        rpm_limiter.wait_time()  # reserves a slot ~1s out; real clock, read moments later below
        monkeypatch.setattr(mod, "_get_limiter", lambda *_a, **_kw: rpm_limiter)

        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: 1000.0)
        token_limiter = mod.TokenBucketLimiter(1000.0)
        token_limiter.record_usage(6000)  # 5000 over -> (5000)/(1000/60) = 300s, way bigger than RPM's ~1s
        monkeypatch.setattr(mod, "_get_token_limiter", lambda *_a, **_kw: token_limiter)

        sleeps = []
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        mod.throttle_before_request(None, "nim", "minimax", request={})

        assert sleeps == [pytest.approx(300.0, abs=0.5)]  # TPM dominates

    def test_estimate_over_capacity_warns_and_still_sends(self, monkeypatch, caplog):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: None)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: 100.0)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        notices = []
        agent = SimpleNamespace(log_prefix="", _vprint=lambda *a, **kw: notices.append(a[0]))

        with caplog.at_level("WARNING", logger="agent.rate_limit_throttle"):
            # "x" * 4000 -> estimate ~1000 tokens, way over the 100/min ceiling.
            mod.throttle_before_request(agent, "nim", "minimax", request={"messages": [{"content": "x" * 4000}]})

        assert any("exceeds" in r.message for r in caplog.records if r.levelname == "WARNING")
        assert any("exceeds" in n for n in notices)

    def test_estimate_within_capacity_does_not_warn(self, monkeypatch, caplog):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: None)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: 100_000.0)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        notices = []
        agent = SimpleNamespace(log_prefix="", _vprint=lambda *a, **kw: notices.append(a[0]))

        with caplog.at_level("WARNING", logger="agent.rate_limit_throttle"):
            mod.throttle_before_request(agent, "nim", "minimax", request={"messages": [{"content": "hi"}]})

        assert not any(r.levelname == "WARNING" for r in caplog.records)
        assert notices == []

    def test_no_request_payload_means_zero_estimate(self, monkeypatch):
        """Callers that don't pass a request body (or pass None) get a 0
        estimate — TPM gating still applies to real recorded usage, just
        without a pre-send size check."""
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: None)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: 1000.0)
        limiter = mod.TokenBucketLimiter(1000.0)
        monkeypatch.setattr(mod, "_get_token_limiter", lambda *_a, **_kw: limiter)
        sleeps = []
        monkeypatch.setattr(mod.time, "sleep", lambda s: sleeps.append(s))

        mod.throttle_before_request(None, "nim", "minimax")  # no request kwarg at all

        assert sleeps == []


class TestRateLimiterCurrentRequests:
    """Rolling 60s request count, for live status-bar display — separate
    from the single-slot scheduler used for actual pacing."""

    def test_zero_before_any_call(self):
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.current_requests(now=0.0) == 0

    def test_counts_calls_within_the_last_60_seconds(self):
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=6000)  # fast enough not to gate
        limiter.wait_time(now=0.0)
        limiter.wait_time(now=10.0)
        limiter.wait_time(now=20.0)
        assert limiter.current_requests(now=20.0) == 3

    def test_calls_older_than_60_seconds_expire(self):
        from agent.rate_limit_throttle import RateLimiter

        limiter = RateLimiter(requests_per_minute=6000)
        limiter.wait_time(now=0.0)
        limiter.wait_time(now=10.0)
        assert limiter.current_requests(now=65.0) == 1  # only the t=10 call remains


class TestTokenBucketLimiterCurrentLevel:
    def test_zero_before_any_usage(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1000)
        assert limiter.current_level(now=0.0) == 0.0

    def test_reflects_recorded_usage(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=1000)
        limiter.record_usage(300, now=0.0)
        assert limiter.current_level(now=0.0) == pytest.approx(300.0)

    def test_drains_over_time_like_wait_time_does(self):
        from agent.rate_limit_throttle import TokenBucketLimiter

        limiter = TokenBucketLimiter(capacity=600)  # drains at 10/sec
        limiter.record_usage(300, now=0.0)
        assert limiter.current_level(now=10.0) == pytest.approx(200.0)


class TestStatusSnapshot:
    """Live RPM/TPM usage for the status bar: hidden entirely when neither
    dimension is configured for (provider, model); each configured
    dimension reports (used, limit) independently."""

    def test_none_when_neither_dimension_configured(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: None)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: None)

        assert mod.status_snapshot("nim", "minimax") is None

    def test_rpm_only(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: 40.0)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: None)
        mod.throttle_before_request(None, "nim", "minimax")  # one real request recorded

        snap = mod.status_snapshot("nim", "minimax")

        assert snap is not None
        assert snap.rpm_used == 1
        assert snap.rpm_limit == 40.0
        assert snap.tpm_used is None
        assert snap.tpm_limit is None

    def test_tpm_only(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: None)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: 1000.0)
        mod.record_input_tokens("nim", "minimax", 250)

        snap = mod.status_snapshot("nim", "minimax")

        assert snap is not None
        assert snap.rpm_used is None
        assert snap.rpm_limit is None
        assert snap.tpm_used == pytest.approx(250.0)
        assert snap.tpm_limit == 1000.0

    def test_both_dimensions_before_any_traffic_report_zero_used(self, monkeypatch):
        import agent.rate_limit_throttle as mod

        monkeypatch.setattr(mod, "get_provider_requests_per_minute", lambda *_a, **_kw: 40.0)
        monkeypatch.setattr(mod, "get_provider_input_tokens_per_minute", lambda *_a, **_kw: 1000.0)

        snap = mod.status_snapshot("nim", "minimax")

        assert snap == mod.RateLimitStatus(rpm_used=0, rpm_limit=40.0, tpm_used=0.0, tpm_limit=1000.0)
