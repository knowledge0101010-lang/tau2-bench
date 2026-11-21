"""Edition profiles. No API keys and no live model calls."""

import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from tau2.cli import add_run_args
from tau2.config import (
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_USER,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
)
from tau2.edition.apply import (
    CONSUMER_MAX_STEPS,
    CONSUMER_NUM_TASKS,
    consumer_missing_key_message,
    default_models_unchanged,
    prepare_edition,
)
from tau2.edition.config import RetryPolicy, load_enterprise_profile, validate_model_id
from tau2.edition.session import (
    BudgetExceeded,
    EditionSession,
    guard_llm_call,
    resolve_routed_model,
)
from tau2.runner.progress import run_with_retry


def _parser():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    return parser


def _fake_response(total_tokens=5):
    usage = SimpleNamespace(
        prompt_tokens=2,
        completion_tokens=total_tokens - 2,
        total_tokens=total_tokens,
    )
    return SimpleNamespace(usage=usage)


def test_no_profile_keeps_standard_defaults():
    args = _parser().parse_args(["--domain", "mock"])
    plan = prepare_edition(args, ["--domain", "mock"])

    assert plan.profile is None
    assert plan.session is None
    assert args.profile is None
    assert args.agent_llm == DEFAULT_LLM_AGENT
    assert args.user_llm == DEFAULT_LLM_USER
    assert args.max_steps == DEFAULT_MAX_STEPS
    assert args.max_concurrency == DEFAULT_MAX_CONCURRENCY
    assert args.seed == DEFAULT_SEED
    assert default_models_unchanged()["agent"] == DEFAULT_LLM_AGENT


def test_consumer_smoke_preset_and_missing_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    args = _parser().parse_args(["--profile", "consumer"])
    with pytest.raises(SystemExit) as caught:
        prepare_edition(args, ["--profile", "consumer"])

    message = str(caught.value)
    assert "XAI_API_KEY" in message
    assert "not Groq" in message
    assert "Groq가 아닙니다" in message
    assert args.domain == "mock"
    assert args.num_tasks == CONSUMER_NUM_TASKS
    assert args.max_steps == CONSUMER_MAX_STEPS
    assert args.max_concurrency == 1
    assert args.agent_llm == "xai/grok-4.7"
    assert "groq/" not in args.agent_llm


def test_consumer_respects_explicit_flags(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    argv = ["--profile", "consumer", "--domain", "airline", "--max-steps", "20"]
    args = _parser().parse_args(argv)
    plan = prepare_edition(args, argv)

    assert plan.profile == "consumer"
    assert plan.consumer_banner is not None
    assert "xAI Grok, not Groq" in plan.consumer_banner
    assert args.domain == "airline"
    assert args.max_steps == 20
    assert args.num_tasks == 1
    assert "test-key" not in plan.consumer_banner


def test_consumer_message_names_xai_not_groq():
    text = consumer_missing_key_message()
    assert "api.x.ai" in text
    assert "xai/" not in text or "Groq" in text


def test_enterprise_routes_models_and_records_reproducibility(tmp_path):
    config_path = tmp_path / "profile.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  agent: openai/gpt-4.1",
                "  user: xai/grok-4.7",
                "  tool_mentor: xai/grok-4.20-0309-non-reasoning",
                "concurrency:",
                "  max_concurrency: 2",
                "budget:",
                "  max_cost_usd: 0.25",
                "  max_tokens: 100",
                "retry:",
                "  max_retries: 2",
                "  min_wait_seconds: 0",
                "  max_wait_seconds: 0",
                "  multiplier: 1",
                "seed: 7",
                f"audit_log: {tmp_path / 'edition-audit.jsonl'}",
            ]
        ),
        encoding="utf-8",
    )
    argv = ["--profile", "enterprise", "--enterprise-config", str(config_path)]
    args = _parser().parse_args(argv)
    plan = prepare_edition(args, argv)

    assert args.agent_llm == "openai/gpt-4.1"
    assert args.user_llm == "xai/grok-4.7"
    assert args.tool_mentor_model == "xai/grok-4.20-0309-non-reasoning"
    assert args.max_concurrency == 2
    assert args.seed == 7
    assert plan.session is not None
    with plan.session as session:
        assert session.git_sha
        assert session.models["agent"] == "openai/gpt-4.1"
        assert session.models["voice_decision"] == "xai/grok-4.7"
        assert resolve_routed_model("xai/grok-4.7", "agent_response") == (
            "openai/gpt-4.1"
        )
        assert resolve_routed_model("other", None) == "other"

    metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))
    assert metadata["seed"] == 7
    assert metadata["git_sha"] == session.git_sha
    assert metadata["models"]["user"] == "xai/grok-4.7"
    audit = session.audit_path.read_text(encoding="utf-8")
    assert "run_start" in audit
    assert "run_end" in audit
    assert "api_key" not in audit


def test_explicit_cli_model_wins_over_yaml(tmp_path):
    config_path = tmp_path / "profile.yaml"
    config_path.write_text(
        "models:\n  agent: xai/grok-4.7\n",
        encoding="utf-8",
    )
    argv = [
        "--profile",
        "enterprise",
        "--enterprise-config",
        str(config_path),
        "--agent-llm",
        "anthropic/claude-sonnet-4-5",
        "--domain",
        "mock",
    ]
    args = _parser().parse_args(argv)
    plan = prepare_edition(args, argv)
    assert args.agent_llm == "anthropic/claude-sonnet-4-5"
    assert plan.session.models["agent"] == "anthropic/claude-sonnet-4-5"


def test_groq_cannot_impersonate_grok():
    with pytest.raises(ValueError, match="Groq"):
        validate_model_id("groq/grok-4.7")
    with pytest.raises(ValueError, match="xai/"):
        validate_model_id("grok-4.7")
    assert validate_model_id("xai/grok-4.7") == "xai/grok-4.7"
    assert validate_model_id("groq/llama-3.3-70b") == "groq/llama-3.3-70b"


def test_example_enterprise_file_loads():
    profile = load_enterprise_profile("docs/examples/enterprise.profile.yaml")
    assert profile.models.agent == "xai/grok-4.7"
    assert profile.models.tool_mentor == "xai/grok-4.20-0309-non-reasoning"
    assert profile.budget.max_tokens == 20000
    assert profile.retry is not None
    assert profile.retry.max_retries == 3


def test_guard_is_passthrough_without_a_profile():
    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return _fake_response()

    result = guard_llm_call(provider, model="xai/grok-4.7", api_key="secret")
    assert result.usage.total_tokens == 5
    assert calls == [{"model": "xai/grok-4.7", "api_key": "secret"}]


def test_budget_hard_stop_and_audit_omit_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau2.edition.session.response_cost",
        lambda response: 0.0,
    )
    session = EditionSession(
        seed=3,
        models={"agent": "xai/grok-4.7"},
        max_concurrency=1,
        max_cost_usd=None,
        max_tokens=4,
        retry=None,
        audit_path=tmp_path / "edition-audit.jsonl",
        metadata_path=tmp_path / "edition-metadata.json",
        git_sha="abc123",
    )
    calls = {"n": 0}

    def provider(**kwargs):
        calls["n"] += 1
        return _fake_response(total_tokens=10)

    with session:
        with pytest.raises(BudgetExceeded):
            guard_llm_call(provider, model="xai/grok-4.7", api_key="SUPER_SECRET")
        with pytest.raises(BudgetExceeded):
            guard_llm_call(provider, model="xai/grok-4.7", api_key="SUPER_SECRET")

    assert calls["n"] == 1
    raw = (tmp_path / "edition-audit.jsonl").read_text(encoding="utf-8")
    assert "SUPER_SECRET" not in raw
    assert "api_key" not in raw
    assert "xai/grok-4.7" in raw
    lines = [json.loads(line) for line in raw.splitlines()]
    assert lines[1]["event"] == "llm_call"
    assert lines[1]["total_tokens"] == 10
    assert set(lines[1]) <= {
        "event",
        "timestamp",
        "run_id",
        "model",
        "ok",
        "error_type",
        "latency_seconds",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "git_sha",
    }


def _session_with_caps(path, **caps):
    path.mkdir(parents=True, exist_ok=True)
    return EditionSession(
        seed=None,
        models={},
        max_concurrency=None,
        max_cost_usd=caps.get("max_cost_usd"),
        max_tokens=caps.get("max_tokens"),
        retry=None,
        audit_path=path / "audit.jsonl",
        metadata_path=path / "meta.json",
        git_sha="deadbeef",
    )


@pytest.mark.parametrize("caps", [{"max_tokens": 0}, {"max_cost_usd": 0.0}])
def test_zero_cap_blocks_the_first_call(tmp_path, caps):
    """A cap of 0 is a hard stop, not one free call."""
    calls = {"n": 0}

    def provider(**kwargs):
        calls["n"] += 1
        return _fake_response(total_tokens=5)

    session = _session_with_caps(tmp_path / "zero", **caps)
    with session:
        with pytest.raises(BudgetExceeded):
            guard_llm_call(provider, model="xai/grok-4.7", api_key="k")
    assert calls["n"] == 0
    assert session.stopped_for_budget is True


def test_exact_cap_spends_nothing_more_and_is_not_a_failure(tmp_path, monkeypatch):
    """Landing exactly on the cap does not fail the run; the next call is refused."""
    monkeypatch.setattr("tau2.edition.session.response_cost", lambda response: 0.0)
    calls = {"n": 0}

    def provider(**kwargs):
        calls["n"] += 1
        return _fake_response(total_tokens=4)

    session = _session_with_caps(tmp_path, max_tokens=4)
    with session:
        guard_llm_call(provider, model="xai/grok-4.7")
        assert session.tokens == 4
        assert session.stopped_for_budget is False
        assert session.blocked_reason() == "token budget exhausted (4 >= 4)"
        with pytest.raises(BudgetExceeded):
            guard_llm_call(provider, model="xai/grok-4.7")
    assert calls["n"] == 1


def test_overspend_stops_at_the_call_that_crossed_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("tau2.edition.session.response_cost", lambda response: 0.0)

    def provider(**kwargs):
        return _fake_response(total_tokens=5)

    session = _session_with_caps(tmp_path, max_tokens=4)
    with session:
        with pytest.raises(BudgetExceeded, match="token budget exceeded \\(5 > 4\\)"):
            guard_llm_call(provider, model="xai/grok-4.7")
        assert session.stopped_for_budget is True
        assert session.blocked_reason() == "enterprise budget already exhausted"


def test_cost_cap_stops_the_next_call(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tau2.edition.session.response_cost",
        lambda response: 1.25,
    )
    session = EditionSession(
        seed=1,
        models={},
        max_concurrency=1,
        max_cost_usd=1.0,
        max_tokens=None,
        retry=None,
        audit_path=tmp_path / "audit.jsonl",
        metadata_path=tmp_path / "meta.json",
    )
    calls = {"n": 0}

    def provider(**kwargs):
        calls["n"] += 1
        return _fake_response()

    with session:
        with pytest.raises(BudgetExceeded):
            guard_llm_call(provider, model="xai/grok-4.7")
        with pytest.raises(BudgetExceeded):
            guard_llm_call(provider, model="xai/grok-4.7")
    assert calls["n"] == 1


def test_retry_backoff_retries_then_succeeds(tmp_path):
    session = EditionSession(
        seed=1,
        models={"agent": "xai/grok-4.7"},
        max_concurrency=1,
        max_cost_usd=None,
        max_tokens=None,
        retry=RetryPolicy(
            max_retries=2,
            min_wait_seconds=0,
            max_wait_seconds=0,
            multiplier=1,
        ),
        audit_path=tmp_path / "audit.jsonl",
        metadata_path=tmp_path / "meta.json",
    )
    seen = []

    def provider(**kwargs):
        seen.append(kwargs.get("num_retries"))
        if len(seen) < 3:
            raise RuntimeError("temporary")
        return _fake_response()

    with session:
        guard_llm_call(provider, model="xai/grok-4.7", num_retries=9, api_key="nope")

    assert seen == [0, 0, 0]
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "nope" not in raw
    assert raw.count('"event": "llm_call"') == 1


def test_budget_exceeded_is_not_retried_as_a_task():
    calls = {"n": 0}

    def run_fn():
        calls["n"] += 1
        raise BudgetExceeded("cap")

    with pytest.raises(BudgetExceeded):
        run_with_retry(
            run_fn,
            SimpleNamespace(id="task"),
            1,
            1,
            max_retries=4,
            retry_delay=0,
            console_display=False,
        )
    assert calls["n"] == 1


def test_cli_budget_stop_exits_2_and_audits_the_real_cause(tmp_path, monkeypatch):
    """The audit run_end names BudgetExceeded, not the SystemExit the CLI raises."""
    import tau2.cli as cli

    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "\n".join(
            [
                "budget:",
                "  max_tokens: 4",
                f"audit_log: {tmp_path / 'audit.jsonl'}",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_domain(config):
        raise BudgetExceeded("token budget exceeded (5 > 4)")

    monkeypatch.setattr(cli, "run_domain", fake_run_domain)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--profile",
            "enterprise",
            "--enterprise-config",
            str(profile),
            "--domain",
            "mock",
            "--num-tasks",
            "1",
            "--save-to",
            str(tmp_path / "run"),
        ],
    )

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 2
    lines = [
        json.loads(line)
        for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    run_end = [line for line in lines if line["event"] == "run_end"][0]
    assert run_end["error_type"] == "BudgetExceeded"
    assert run_end["stopped_for_budget"] is True
    assert run_end["ok"] is False


def test_enterprise_config_is_required():
    args = _parser().parse_args(["--profile", "enterprise", "--domain", "mock"])
    with pytest.raises(SystemExit, match="enterprise-config"):
        prepare_edition(args, ["--profile", "enterprise", "--domain", "mock"])
