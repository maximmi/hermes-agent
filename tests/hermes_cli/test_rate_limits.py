"""Tests for hermes_cli/rate_limits.py — proactive-throttle config resolver.

Mirrors the provider/per-model override pattern already proven in
tests/hermes_cli/test_timeouts.py.
"""

from __future__ import annotations

import textwrap


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def _reload_with_config(monkeypatch, tmp_path, body: str):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, body)
    import importlib
    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import rate_limits as rl_mod
    importlib.reload(rl_mod)
    return rl_mod


def test_disabled_when_unconfigured(monkeypatch, tmp_path):
    rl = _reload_with_config(monkeypatch, tmp_path, "")
    assert rl.get_provider_requests_per_minute("nim", "minimax") is None


def test_provider_level_setting_is_used(monkeypatch, tmp_path):
    rl = _reload_with_config(monkeypatch, tmp_path, """\
        providers:
          nim:
            requests_per_minute: 40
        """)
    assert rl.get_provider_requests_per_minute("nim", "minimax") == 40.0


def test_per_model_override_wins_over_provider_level(monkeypatch, tmp_path):
    rl = _reload_with_config(monkeypatch, tmp_path, """\
        providers:
          nim:
            requests_per_minute: 40
            models:
              minimax:
                requests_per_minute: 10
        """)
    assert rl.get_provider_requests_per_minute("nim", "minimax") == 10.0
    # A different model on the same provider falls back to the provider level.
    assert rl.get_provider_requests_per_minute("nim", "other-model") == 40.0


def test_zero_or_negative_means_disabled(monkeypatch, tmp_path):
    rl = _reload_with_config(monkeypatch, tmp_path, """\
        providers:
          nim:
            requests_per_minute: 0
            models:
              minimax:
                requests_per_minute: -5
        """)
    assert rl.get_provider_requests_per_minute("nim") is None
    assert rl.get_provider_requests_per_minute("nim", "minimax") is None


def test_unconfigured_provider_id_returns_none(monkeypatch, tmp_path):
    rl = _reload_with_config(monkeypatch, tmp_path, """\
        providers:
          nim:
            requests_per_minute: 40
        """)
    assert rl.get_provider_requests_per_minute("openrouter", "minimax") is None


class TestInputTokensPerMinute:
    """Same provider/per-model-override pattern, independent of requests_per_minute."""

    def test_disabled_when_unconfigured(self, monkeypatch, tmp_path):
        rl = _reload_with_config(monkeypatch, tmp_path, "")
        assert rl.get_provider_input_tokens_per_minute("nim", "minimax") is None

    def test_provider_level_setting_is_used(self, monkeypatch, tmp_path):
        rl = _reload_with_config(monkeypatch, tmp_path, """\
            providers:
              nim:
                input_tokens_per_minute: 100000
            """)
        assert rl.get_provider_input_tokens_per_minute("nim", "minimax") == 100000.0

    def test_per_model_override_wins_over_provider_level(self, monkeypatch, tmp_path):
        rl = _reload_with_config(monkeypatch, tmp_path, """\
            providers:
              nim:
                input_tokens_per_minute: 100000
                models:
                  minimax:
                    input_tokens_per_minute: 20000
            """)
        assert rl.get_provider_input_tokens_per_minute("nim", "minimax") == 20000.0
        assert rl.get_provider_input_tokens_per_minute("nim", "other-model") == 100000.0

    def test_zero_or_negative_means_disabled(self, monkeypatch, tmp_path):
        rl = _reload_with_config(monkeypatch, tmp_path, """\
            providers:
              nim:
                input_tokens_per_minute: 0
            """)
        assert rl.get_provider_input_tokens_per_minute("nim") is None

    def test_independent_of_requests_per_minute(self, monkeypatch, tmp_path):
        """Setting one throttle dimension must not implicitly enable the other."""
        rl = _reload_with_config(monkeypatch, tmp_path, """\
            providers:
              nim:
                requests_per_minute: 40
            """)
        assert rl.get_provider_input_tokens_per_minute("nim") is None
        assert rl.get_provider_requests_per_minute("nim") == 40.0
