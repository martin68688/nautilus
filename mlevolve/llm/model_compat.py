"""Provider model-name compatibility and validation.

Keep provider migrations out of the generic retry loop: deprecated aliases are
translated before the request is sent, while unknown names on the official
endpoint fail locally instead of consuming the generation retry budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
DEEPSEEK_CURRENT_MODELS = frozenset({"deepseek-v4-flash", "deepseek-v4-pro"})

# The legacy aliases selected both a model and a thinking mode. Preserve both
# parts of that contract when translating them to the V4 API.
_DEEPSEEK_LEGACY_ALIASES: dict[str, tuple[str, bool]] = {
    "deepseek-chat": ("deepseek-v4-flash", False),
    "deepseek-reasoner": ("deepseek-v4-flash", True),
}


class UnsupportedDeepSeekModel(ValueError):
    """Raised before I/O for an invalid model on the official DeepSeek API."""


@dataclass(frozen=True)
class ModelResolution:
    requested_name: str
    effective_name: str
    forced_thinking: bool | None = None

    @property
    def migrated(self) -> bool:
        return self.requested_name != self.effective_name

    @property
    def is_deepseek_v4(self) -> bool:
        return self.effective_name in DEEPSEEK_CURRENT_MODELS


def is_official_deepseek_endpoint(base_url: str | None) -> bool:
    """Return whether ``base_url`` targets DeepSeek's official API host."""
    if not base_url:
        return False
    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    return (parsed.hostname or "").lower() == "api.deepseek.com"


def resolve_model_name(model_name: str, *, base_url: str | None = None) -> ModelResolution:
    """Resolve deprecated DeepSeek aliases and validate the official endpoint.

    Custom OpenAI-compatible endpoints remain open-ended because they may expose
    provider-specific model names. The two known legacy DeepSeek aliases are
    translated for every endpoint so pass-through gateways receive current names.
    """
    requested = (model_name or "").strip()
    normalized = requested.lower()

    if normalized in _DEEPSEEK_LEGACY_ALIASES:
        effective, thinking = _DEEPSEEK_LEGACY_ALIASES[normalized]
        return ModelResolution(requested, effective, thinking)

    if normalized in DEEPSEEK_CURRENT_MODELS:
        return ModelResolution(requested, normalized)

    if is_official_deepseek_endpoint(base_url):
        accepted = ", ".join(sorted(DEEPSEEK_CURRENT_MODELS))
        aliases = ", ".join(sorted(_DEEPSEEK_LEGACY_ALIASES))
        raise UnsupportedDeepSeekModel(
            f"Unsupported DeepSeek model {requested!r} for api.deepseek.com. "
            f"Use one of: {accepted}. Deprecated aliases accepted by this adapter: {aliases}."
        )

    return ModelResolution(requested, requested)


def deepseek_thinking_extra_body(
    resolution: ModelResolution,
    *,
    use_thinking: bool,
) -> dict:
    """Build the V4 OpenAI-format thinking toggle, preserving alias semantics."""
    if not resolution.is_deepseek_v4:
        return {}
    enabled = resolution.forced_thinking
    if enabled is None:
        enabled = use_thinking
    return {"thinking": {"type": "enabled" if enabled else "disabled"}}
