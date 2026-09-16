"""Config resolver for proactive client-side request throttling.

Mirrors ``hermes_cli.timeouts``: a per-model
``providers.<id>.models.<model>.requests_per_minute`` wins over the
provider-wide ``providers.<id>.requests_per_minute``. Absent or <= 0 means
disabled (the default).
"""

from __future__ import annotations

from typing import Optional


def _coerce_rpm(raw: object) -> Optional[float]:
    try:
        rpm = float(raw)
    except (TypeError, ValueError):
        return None
    return rpm if rpm > 0 else None


def get_provider_requests_per_minute(provider_id: str, model: Optional[str] = None) -> Optional[float]:
    """Return the configured requests-per-minute throttle ceiling, if any."""
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
            rpm = _coerce_rpm(model_config.get("requests_per_minute"))
            if rpm is not None:
                return rpm
    return _coerce_rpm(provider_config.get("requests_per_minute"))
