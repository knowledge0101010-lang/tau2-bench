"""Apply ``--profile`` to a parsed ``tau2 run`` namespace.

No profile leaves the namespace unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from tau2.config import (
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ENV_INTERFACE,
    DEFAULT_LLM_EVAL_USER_SIMULATOR,
    DEFAULT_LLM_NL_ASSERTIONS,
    DEFAULT_LLM_TOOL_MENTOR,
    DEFAULT_LLM_USER,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_SEED,
    VOICE_USER_SIMULATOR_DECISION_MODEL,
)
from tau2.edition.config import EnterpriseProfile, load_enterprise_profile
from tau2.edition.session import EditionSession
from tau2.utils.utils import DATA_DIR, get_commit_hash

CONSUMER_DOMAIN = "mock"
CONSUMER_NUM_TASKS = 1
CONSUMER_NUM_TRIALS = 1
CONSUMER_MAX_CONCURRENCY = 1
CONSUMER_MAX_STEPS = 8
CONSUMER_MAX_ERRORS = 3

_ROLE_ARG = {
    "agent": "agent_llm",
    "user": "user_llm",
    "review": "review_model",
    "tool_mentor": "tool_mentor_model",
}
_ROLE_FLAG = {
    "agent": "--agent-llm",
    "user": "--user-llm",
    "review": "--review-model",
    "tool_mentor": "--tool-mentor-model",
}


@dataclass
class EditionPlan:
    """What ``run`` should do around the normal config build."""

    profile: Optional[str] = None
    session: Optional[EditionSession] = None
    consumer_banner: Optional[str] = None


def flag_on_cli(argv: list[str], *flags: str) -> bool:
    """True when any flag was passed explicitly (``--flag`` or ``--flag=value``)."""
    for arg in argv:
        for flag in flags:
            if arg == flag or arg.startswith(flag + "="):
                return True
    return False


def consumer_missing_key_message() -> str:
    """Bilingual stop message. xAI Grok is not Groq."""
    return (
        "XAI_API_KEY is not set.\n"
        "The consumer profile calls xAI Grok at api.x.ai. This is xAI, not Groq.\n"
        "Add XAI_API_KEY to .env (see .env.example), then rerun:\n"
        "  tau2 run --profile consumer\n"
        "\n"
        "XAI_API_KEY가 없습니다.\n"
        "consumer 프로필은 xAI Grok(api.x.ai)을 호출합니다. Groq가 아닙니다.\n"
        ".env에 XAI_API_KEY를 넣은 뒤 다시 실행하세요:\n"
        "  tau2 run --profile consumer"
    )


def consumer_banner(args) -> str:
    sha = get_commit_hash()
    short = sha[:12] if sha != "unknown" else sha
    return (
        "tau2 consumer smoke (xAI Grok, not Groq)\n"
        f"  domain: {args.domain}\n"
        f"  tasks: {args.num_tasks}  trials: {args.num_trials}  "
        f"max_steps: {args.max_steps}  concurrency: {args.max_concurrency}\n"
        f"  agent: {args.agent_llm}\n"
        f"  user: {args.user_llm}\n"
        f"  seed: {args.seed}\n"
        f"  git: {short}"
    )


def consumer_summary(result) -> str:
    count = len(getattr(result, "simulations", []) or [])
    return f"Consumer smoke finished. simulations: {count}"


def _apply_consumer(args, argv: list[str]) -> None:
    if getattr(args, "audio_native", False):
        raise SystemExit(
            "--profile consumer is a text mock smoke. "
            "Voice runs use --profile enterprise or no profile."
        )
    if not flag_on_cli(argv, "--domain", "-d"):
        args.domain = CONSUMER_DOMAIN
    if not flag_on_cli(argv, "--num-tasks"):
        args.num_tasks = CONSUMER_NUM_TASKS
    if not flag_on_cli(argv, "--num-trials"):
        args.num_trials = CONSUMER_NUM_TRIALS
    if not flag_on_cli(argv, "--max-concurrency"):
        args.max_concurrency = CONSUMER_MAX_CONCURRENCY
    if not flag_on_cli(argv, "--max-steps"):
        args.max_steps = CONSUMER_MAX_STEPS
    if not flag_on_cli(argv, "--max-errors"):
        args.max_errors = CONSUMER_MAX_ERRORS
    if not flag_on_cli(argv, "--save-to"):
        args.save_to = "consumer-smoke"


def _effective_models(args, profile: EnterpriseProfile) -> dict[str, str]:
    """Resolved role -> model id after CLI flags have overridden the YAML."""
    routed = profile.routed_models()
    return {
        "agent": args.agent_llm,
        "user": args.user_llm,
        "review": args.review_model,
        "tool_mentor": args.tool_mentor_model,
        "nl_assertions": routed.get("nl_assertions", DEFAULT_LLM_NL_ASSERTIONS),
        "env_interface": routed.get("env_interface", DEFAULT_LLM_ENV_INTERFACE),
        "voice_decision": routed.get(
            "voice_decision", VOICE_USER_SIMULATOR_DECISION_MODEL
        ),
    }


def _apply_enterprise(
    args, argv: list[str], profile: EnterpriseProfile
) -> EditionSession:
    for role, arg_name in _ROLE_ARG.items():
        routed = getattr(profile.models, role)
        if routed is None:
            continue
        if flag_on_cli(argv, _ROLE_FLAG[role]):
            continue
        setattr(args, arg_name, routed)
    if profile.concurrency is not None and not flag_on_cli(argv, "--max-concurrency"):
        args.max_concurrency = profile.concurrency.max_concurrency
    if profile.seed is not None and not flag_on_cli(argv, "--seed"):
        args.seed = profile.seed
    if not flag_on_cli(argv, "--save-to") and args.save_to is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.save_to = f"enterprise-{stamp}"

    models = _effective_models(args, profile)
    if profile.audit_log:
        audit_path = Path(profile.audit_log)
    else:
        audit_path = DATA_DIR / "simulations" / args.save_to / "edition-audit.jsonl"
    metadata_path = audit_path.with_name("edition-metadata.json")
    return EditionSession(
        seed=args.seed,
        models=models,
        max_concurrency=args.max_concurrency,
        max_cost_usd=profile.budget.max_cost_usd,
        max_tokens=profile.budget.max_tokens,
        retry=profile.retry,
        audit_path=audit_path,
        metadata_path=metadata_path,
    )


def prepare_edition(args, argv: list[str]) -> EditionPlan:
    """Mutate ``args`` for the selected profile. No profile is a no-op.

    Consumer exits before any model call when ``XAI_API_KEY`` is missing.
    """
    profile = getattr(args, "profile", None)
    config_path = getattr(args, "enterprise_config", None)
    if profile is None:
        if config_path:
            raise SystemExit(
                "--enterprise-config requires --profile enterprise. "
                "With no profile, standard defaults are unchanged."
            )
        return EditionPlan()

    if profile == "consumer":
        if config_path:
            raise SystemExit(
                "--enterprise-config is only used with --profile enterprise."
            )
        _apply_consumer(args, argv)
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        if not os.environ.get("XAI_API_KEY"):
            raise SystemExit(consumer_missing_key_message())
        return EditionPlan(profile="consumer", consumer_banner=consumer_banner(args))

    if profile == "enterprise":
        if not config_path:
            raise SystemExit(
                "--profile enterprise requires --enterprise-config profile.yaml."
            )
        try:
            loaded = load_enterprise_profile(config_path)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        session = _apply_enterprise(args, argv, loaded)
        return EditionPlan(profile="enterprise", session=session)

    raise SystemExit(f"Unknown profile: {profile}")


def default_models_unchanged() -> dict[str, str]:
    """The text defaults a no-profile run keeps. Used by tests."""
    return {
        "agent": DEFAULT_LLM_AGENT,
        "user": DEFAULT_LLM_USER,
        "nl_assertions": DEFAULT_LLM_NL_ASSERTIONS,
        "env_interface": DEFAULT_LLM_ENV_INTERFACE,
        "review": DEFAULT_LLM_EVAL_USER_SIMULATOR,
        "tool_mentor": DEFAULT_LLM_TOOL_MENTOR,
        "voice_decision": VOICE_USER_SIMULATOR_DECISION_MODEL,
        "max_concurrency": str(DEFAULT_MAX_CONCURRENCY),
        "max_steps": str(DEFAULT_MAX_STEPS),
        "max_errors": str(DEFAULT_MAX_ERRORS),
        "num_trials": str(DEFAULT_NUM_TRIALS),
        "seed": str(DEFAULT_SEED),
    }
