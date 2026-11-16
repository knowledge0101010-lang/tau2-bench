"""Enterprise profile YAML.

Model ids are LiteLLM strings. xAI Grok uses the ``xai/`` prefix
(api.x.ai). Groq is a different company and a different provider prefix.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_GROK_MARK = "grok"
_XAI_PREFIX = "xai/"
_GROQ_PREFIX = "groq/"


def validate_model_id(model: str) -> str:
    """Reject ids that mix xAI Grok with Groq.

    ``xai/grok-4.7`` is xAI. ``groq/...`` is Groq. A Grok model id must use
    the xAI prefix. Other providers (``openai/``, ``anthropic/``) stay valid.
    """
    text = model.strip()
    if not text:
        raise ValueError("model id is empty")
    lower = text.lower()
    names_grok = _GROK_MARK in lower
    if lower.startswith(_GROQ_PREFIX) and names_grok:
        raise ValueError(
            f"{text} routes Grok through Groq. Use xai/<model> for xAI "
            "(api.x.ai). Groq is a different provider."
        )
    if lower.startswith(_XAI_PREFIX) and "groq" in lower:
        raise ValueError(
            f"{text} is an xAI model id that names Groq. xAI Grok ids look "
            "like xai/grok-4.7."
        )
    if names_grok and not lower.startswith(_XAI_PREFIX):
        raise ValueError(
            f"{text} names Grok but does not use the xai/ prefix. "
            "xAI Grok is xai/<model> (api.x.ai), not Groq."
        )
    return text


class RoleModels(BaseModel):
    """Per-role model ids. Omitted roles keep the process defaults."""

    model_config = ConfigDict(extra="forbid")

    agent: Optional[str] = None
    user: Optional[str] = None
    nl_assertions: Optional[str] = None
    env_interface: Optional[str] = None
    review: Optional[str] = None
    tool_mentor: Optional[str] = None
    voice_decision: Optional[str] = None

    @field_validator("*")
    @classmethod
    def _check_model(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return validate_model_id(value)


class ConcurrencyLimit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_concurrency: int = Field(ge=1)


class BudgetLimit(BaseModel):
    """Hard caps for one run. Crossing either cap stops further model calls."""

    model_config = ConfigDict(extra="forbid")

    max_cost_usd: Optional[float] = Field(default=None, ge=0)
    max_tokens: Optional[int] = Field(default=None, ge=0)


class RetryPolicy(BaseModel):
    """Backoff for each model call. ``max_retries`` is extra attempts after the first."""

    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=3, ge=0)
    min_wait_seconds: float = Field(default=1.0, ge=0)
    max_wait_seconds: float = Field(default=10.0, ge=0)
    multiplier: float = Field(default=1.0, gt=0)


class EnterpriseProfile(BaseModel):
    """YAML document for ``--profile enterprise``."""

    model_config = ConfigDict(extra="forbid")

    models: RoleModels = Field(default_factory=RoleModels)
    concurrency: Optional[ConcurrencyLimit] = None
    budget: BudgetLimit = Field(default_factory=BudgetLimit)
    retry: Optional[RetryPolicy] = None
    seed: Optional[int] = None
    audit_log: Optional[str] = None

    def routed_models(self) -> dict[str, str]:
        return {
            name: value
            for name, value in self.models.model_dump().items()
            if value is not None
        }


def load_enterprise_profile(path: str | Path) -> EnterpriseProfile:
    """Load and validate an enterprise YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Enterprise config not found: {config_path}. "
            "Pass --enterprise-config path/to/profile.yaml."
        )
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Enterprise config {config_path} must be a YAML mapping.")
    return EnterpriseProfile.model_validate(raw)
