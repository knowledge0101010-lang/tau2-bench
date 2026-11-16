"""Runtime edition session: routing, budget hard-stop, and secret-free audit."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from tau2.edition.config import RetryPolicy
from tau2.utils.utils import get_commit_hash

# call_name values produced by generate() -> enterprise role
CALL_NAME_TO_ROLE = {
    "agent_response": "agent",
    "agent_gt_response": "agent",
    "agent_solo_response": "agent",
    "user_simulator_response": "user",
    "user_streaming_response": "user",
    "nl_assertions_eval": "nl_assertions",
    "interface_agent_response": "env_interface",
    "interruption_decision": "voice_decision",
    "backchannel_decision": "voice_decision",
    "llm_judge_review": "review",
    "llm_judge_streaming_review": "review",
    "llm_judge_user_only_review": "review",
    "llm_judge_user_only_streaming_review": "review",
    "llm_judge_hallucination_check": "review",
    "classify_authentication": "review",
}

_AUDIT_FIELDS = (
    "event",
    "timestamp",
    "run_id",
    "model",
    "call_role",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "latency_seconds",
    "ok",
    "error_type",
    "git_sha",
    "seed",
    "profile",
    "models",
    "max_concurrency",
    "budget",
    "retry",
    "started_at",
    "finished_at",
    "stopped_for_budget",
)


class BudgetExceeded(RuntimeError):
    """The enterprise cost or token cap was reached. The run must stop."""


_current: Optional["EditionSession"] = None
_global_lock = threading.Lock()


def current_session() -> Optional["EditionSession"]:
    return _current


def resolve_routed_model(model: str, call_name: Optional[str]) -> str:
    """Return the enterprise model for this call, or ``model`` when no profile is active."""
    session = _current
    if session is None or not call_name:
        return model
    role = CALL_NAME_TO_ROLE.get(call_name)
    if role is None:
        return model
    routed = session.models.get(role)
    return routed or model


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, Exception) and not isinstance(exc, BudgetExceeded)


@dataclass
class EditionSession:
    """One enterprise run. Inactive unless entered as a context manager."""

    seed: Optional[int]
    models: dict[str, str]
    max_concurrency: Optional[int]
    max_cost_usd: Optional[float]
    max_tokens: Optional[int]
    retry: Optional[RetryPolicy]
    audit_path: Path
    metadata_path: Path
    git_sha: str = field(default_factory=get_commit_hash)
    profile: str = "enterprise"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=_utc_now)
    cost_usd: float = 0.0
    tokens: int = 0
    stopped_for_budget: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __enter__(self) -> "EditionSession":
        global _current
        with _global_lock:
            if _current is not None:
                raise RuntimeError("An edition session is already active")
            _current = self
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        start = self._metadata_body()
        start["event"] = "run_start"
        self._append(start)
        self.write_metadata(final=False)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        global _current
        finished = _utc_now()
        self._append(
            {
                "event": "run_end",
                "timestamp": finished,
                "finished_at": finished,
                "cost_usd": self.cost_usd,
                "total_tokens": self.tokens,
                "stopped_for_budget": self.stopped_for_budget
                or (exc_type is BudgetExceeded),
                "error_type": None if exc_type is None else exc_type.__name__,
                "ok": exc_type is None,
            }
        )
        self.write_metadata(final=True)
        with _global_lock:
            if _current is self:
                _current = None
        return False

    def blocked_reason(self) -> Optional[str]:
        if self.stopped_for_budget:
            return "enterprise budget already exhausted"
        if (
            self.max_tokens is not None
            and self.tokens >= self.max_tokens
            and self.tokens > 0
        ):
            return f"token budget exhausted ({self.tokens} >= {self.max_tokens})"
        if (
            self.max_cost_usd is not None
            and self.cost_usd >= self.max_cost_usd
            and self.cost_usd > 0
        ):
            return (
                "cost budget exhausted "
                f"({self.cost_usd:.6f} >= {self.max_cost_usd:.6f} USD)"
            )
        return None

    def note_usage(self, added_tokens: int, added_cost: float) -> None:
        with self._lock:
            self.tokens += max(added_tokens, 0)
            self.cost_usd += max(added_cost, 0.0)
            reason = self._over_cap_reason()
            if reason:
                self.stopped_for_budget = True
                raise BudgetExceeded(reason)

    def _over_cap_reason(self) -> Optional[str]:
        if self.max_tokens is not None and self.tokens > self.max_tokens:
            return f"token budget exceeded ({self.tokens} > {self.max_tokens})"
        if self.max_cost_usd is not None and self.cost_usd > self.max_cost_usd:
            return (
                "cost budget exceeded "
                f"({self.cost_usd:.6f} > {self.max_cost_usd:.6f} USD)"
            )
        return None

    def public_metadata(self) -> dict[str, Any]:
        body = self._metadata_body()
        body["cost_usd"] = self.cost_usd
        body["total_tokens"] = self.tokens
        body["stopped_for_budget"] = self.stopped_for_budget
        return body

    def write_metadata(self, *, final: bool) -> None:
        body = self.public_metadata()
        if final:
            body["finished_at"] = _utc_now()
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(body, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def record_call(
        self,
        *,
        model: Optional[str],
        ok: bool,
        error_type: Optional[str],
        latency_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
    ) -> None:
        self._append(
            {
                "event": "llm_call",
                "timestamp": _utc_now(),
                "model": model,
                "ok": ok,
                "error_type": error_type,
                "latency_seconds": round(latency_seconds, 6),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
            }
        )

    def _metadata_body(self) -> dict[str, Any]:
        retry = None
        if self.retry is not None:
            retry = self.retry.model_dump()
        return {
            "event": "run_metadata",
            "profile": self.profile,
            "run_id": self.run_id,
            "seed": self.seed,
            "git_sha": self.git_sha,
            "models": dict(self.models),
            "max_concurrency": self.max_concurrency,
            "budget": {
                "max_cost_usd": self.max_cost_usd,
                "max_tokens": self.max_tokens,
            },
            "retry": retry,
            "started_at": self.started_at,
        }

    def _append(self, payload: dict[str, Any]) -> None:
        safe = {key: payload[key] for key in _AUDIT_FIELDS if key in payload}
        safe.setdefault("run_id", self.run_id)
        safe.setdefault("git_sha", self.git_sha)
        line = json.dumps(safe, sort_keys=True, default=str)
        with self._lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _usage_tokens(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0, 0
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = usage.get("total_tokens")
    else:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        total = getattr(usage, "total_tokens", None)
    if total is None:
        total = prompt + completion
    return prompt, completion, int(total)


def response_cost(response: Any) -> float:
    """Best-effort USD cost. Unknown prices count as zero and still track tokens."""
    try:
        from litellm import completion_cost

        cost = completion_cost(completion_response=response)
    except Exception:
        return 0.0
    if cost is None:
        return 0.0
    return float(cost)


def guard_llm_call(provider_call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call the provider. With no edition session this is a direct pass-through."""
    session = current_session()
    if session is None:
        return provider_call(*args, **kwargs)

    reason = session.blocked_reason()
    if reason:
        session.stopped_for_budget = True
        raise BudgetExceeded(reason)

    inner = dict(kwargs)
    if session.retry is not None:
        inner["num_retries"] = 0
    model = inner.get("model")
    start = time.perf_counter()

    def attempt() -> Any:
        return provider_call(*args, **inner)

    try:
        if session.retry is None or session.retry.max_retries <= 0:
            response = attempt()
        else:
            policy = session.retry
            response = Retrying(
                stop=stop_after_attempt(policy.max_retries + 1),
                wait=wait_exponential(
                    multiplier=policy.multiplier,
                    min=policy.min_wait_seconds,
                    max=policy.max_wait_seconds,
                ),
                retry=retry_if_exception(_retryable),
                reraise=True,
            )(attempt)
    except BudgetExceeded:
        raise
    except Exception as exc:
        session.record_call(
            model=model if isinstance(model, str) else None,
            ok=False,
            error_type=type(exc).__name__,
            latency_seconds=time.perf_counter() - start,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
        )
        raise

    prompt, completion_tokens, total = _usage_tokens(response)
    cost = response_cost(response)
    session.record_call(
        model=model if isinstance(model, str) else None,
        ok=True,
        error_type=None,
        latency_seconds=time.perf_counter() - start,
        prompt_tokens=prompt,
        completion_tokens=completion_tokens,
        total_tokens=total,
        cost_usd=cost,
    )
    session.note_usage(total, cost)
    return response
