"""Wiring tests: every real outbound LLM call funnels through exactly two
places (see agent/turn_api_call.py and agent/chat_completion_helpers.py
module docstrings) — the main turn loop's ``perform_api_call`` and the
iteration-summary path's ``_managed_summary_call``. Both must pace the
request via agent.rate_limit_throttle.throttle_before_request before doing
anything else, so a slow/misconfigured attempt never skips pacing.

A prior PR (#17749) wired this into AIAgent construction instead, which the
reviewer noted had already drifted from where requests actually go out. Pin
the two real call sites here so a future refactor can't silently drop this.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _ThrottleCalled(Exception):
    """Raised by a stubbed throttle_before_request to prove it ran, and ran first."""


def _stub_throttle(calls):
    def _throttle(agent, provider, model):
        calls.append((provider, model))
        raise _ThrottleCalled()
    return _throttle


class TestPerformApiCallThrottling:
    def test_throttles_before_dispatching_the_request(self, monkeypatch):
        import agent.turn_api_call as mod

        calls = []
        monkeypatch.setattr(mod, "throttle_before_request", _stub_throttle(calls))

        agent = SimpleNamespace(provider="nim", model="minimax")

        with pytest.raises(_ThrottleCalled):
            mod.perform_api_call(
                agent,
                api_kwargs={}, _original_api_kwargs={}, _llm_middleware_trace=[],
                _moa_prepared_request=None, _retry=None, thinking_spinner=None,
                retry_count=0, api_call_count=0, api_request_id="req-1",
                effective_task_id="t", turn_id="tid", interrupted=False,
            )

        assert calls == [("nim", "minimax")]


class TestManagedSummaryCallThrottling:
    def test_throttles_before_dispatching_the_summary_request(self, monkeypatch):
        import agent.chat_completion_helpers as mod

        calls = []
        monkeypatch.setattr(mod, "throttle_before_request", _stub_throttle(calls))

        agent = SimpleNamespace(provider="openrouter", model="some/model")

        with pytest.raises(_ThrottleCalled):
            mod._managed_summary_call(
                agent, "summary-req-1", {"messages": []}, lambda request: None, retry_count=0,
            )

        assert calls == [("openrouter", "some/model")]
