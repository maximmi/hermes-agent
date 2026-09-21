"""Config resolver for proactive client-side request throttling.

Mirrors ``hermes_cli.timeouts``: a per-model
``providers.<id>.models.<model>.<key>`` wins over the provider-wide
``providers.<id>.<key>``. Absent or <= 0 means disabled (the default).

Two independent throttle dimensions, each its own on/off switch:
``requests_per_minute`` and ``input_tokens_per_minute``.
"""

from __future__ import annotations

from typing import Optional


def _coerce_positive(raw: object) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _configured_value(provider_id: str, model: Optional[str], key: str) -> Optional[float]:
    if not provider_id:
        return None
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = providers.get(provider_id, {}) if isinstance(providers, dict) else {}
    if not isinstance(provider_config, dict):
        return None
    if model:
        models = provider_config.get("models", {})
        model_config = models.get(model, {}) if isinstance(models, dict) else {}
        if isinstance(model_config, dict):
            value = _coerce_positive(model_config.get(key))
            if value is not None:
                return value
    return _coerce_positive(provider_config.get(key))


def get_provider_requests_per_minute(provider_id: str, model: Optional[str] = None) -> Optional[float]:
    """Return the configured requests-per-minute throttle ceiling, if any."""
    return _configured_value(provider_id, model, "requests_per_minute")


def get_provider_input_tokens_per_minute(provider_id: str, model: Optional[str] = None) -> Optional[float]:
    """Return the configured input-tokens-per-minute throttle ceiling, if any."""
    return _configured_value(provider_id, model, "input_tokens_per_minute")
