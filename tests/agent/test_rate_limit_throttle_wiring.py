"""Wiring tests: every real outbound LLM call funnels through exactly two
places (see agent/turn_api_call.py and agent/chat_completion_helpers.py
module docstrings) — the main turn loop's ``perform_api_call`` and the
iteration-summary path's ``_managed_summary_call``. Both must pace the
request via agent.rate_limit_throttle.throttle_before_request before doing
anything else, passing the request body along (needed for the
input-tokens-per-minute size estimate), so a slow/misconfigured attempt
never skips pacing.

Symmetrically, actual (provider-confirmed) input-token usage must be fed
back into the TPM bucket after a response: agent/turn_usage.py for the main
loop, and inside _managed_summary_call itself for the summary path (the
only place all three summary attempt builders funnel through).

A prior PR (#17749) wired the pre-request side into AIAgent construction
instead, which the reviewer noted had already drifted from where requests
actually go out. Pin the real call sites here so a future refactor can't
silently drop this.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _ThrottleCalled(Exception):
    """Raised by a stubbed throttle_before_request to prove it ran, and ran first."""


def _stub_throttle(calls):
    def _throttle(agent, provider, model, request=None):
        calls.append((provider, model, request))
        raise _ThrottleCalled()
    return _throttle


class TestPerformApiCallThrottling:
    def test_throttles_before_dispatching_the_request(self, monkeypatch):
        import agent.turn_api_call as mod

        calls = []
        monkeypatch.setattr(mod, "throttle_before_request", _stub_throttle(calls))

        agent = SimpleNamespace(provider="nim", model="minimax")
        api_kwargs = {"messages": [{"role": "user", "content": "hi"}]}

        with pytest.raises(_ThrottleCalled):
            mod.perform_api_call(
                agent,
                api_kwargs=api_kwargs, _original_api_kwargs={}, _llm_middleware_trace=[],
                _moa_prepared_request=None, _retry=None, thinking_spinner=None,
                retry_count=0, api_call_count=0, api_request_id="req-1",
                effective_task_id="t", turn_id="tid", interrupted=False,
            )

        assert calls == [("nim", "minimax", api_kwargs)]


class TestManagedSummaryCallThrottling:
    def test_throttles_before_dispatching_the_summary_request(self, monkeypatch):
        import agent.chat_completion_helpers as mod

        calls = []
        monkeypatch.setattr(mod, "throttle_before_request", _stub_throttle(calls))

        agent = SimpleNamespace(provider="openrouter", model="some/model")
        request = {"messages": []}

        with pytest.raises(_ThrottleCalled):
            mod._managed_summary_call(
                agent, "summary-req-1", request, lambda request: None, retry_count=0,
            )

        assert calls == [("openrouter", "some/model", request)]

    def test_feeds_actual_input_tokens_back_after_the_response(self, monkeypatch):
        """The summary path never runs record_response_usage (that's the main
        loop's accounting chokepoint) — it must feed the TPM bucket itself."""
        import agent.chat_completion_helpers as mod

        monkeypatch.setattr(mod, "throttle_before_request", lambda *a, **kw: None)

        recorded = []
        monkeypatch.setattr(mod, "record_input_tokens", lambda provider, model, tokens: recorded.append(
            (provider, model, tokens),
        ))

        agent = SimpleNamespace(provider="openrouter", model="some/model", api_mode="chat_completions")
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=123, completion_tokens=7, total_tokens=130))

        def relay_execute_current(request, callback, **_kw):
            return callback(request)

        monkeypatch.setattr("agent.relay_llm.execute_current", relay_execute_current)

        result = mod._managed_summary_call(
            agent, "summary-req-2", {"messages": []}, lambda request: response, retry_count=0,
        )

        assert result is response
        assert recorded == [("openrouter", "some/model", 123)]


class TestRecordResponseUsageFeedsTokenBucket:
    def test_main_loop_feeds_actual_prompt_tokens_after_the_response(self, tmp_path, monkeypatch):
        import agent.turn_usage as mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
            api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            enabled_toolsets=["file"],
        )
        recorded = []
        monkeypatch.setattr(mod, "record_input_tokens", lambda provider, model, tokens: recorded.append(
            (provider, model, tokens),
        ))
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=456, completion_tokens=7, total_tokens=463,
                prompt_tokens_details=None, completion_tokens_details=None,
            ),
        )
        try:
            mod.record_response_usage(
                agent, response, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
            )
        finally:
            agent.close()

        assert recorded == [("nous", "anthropic/claude-fable-5.1", 456)]
