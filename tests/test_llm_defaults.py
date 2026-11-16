"""Pins text-role LLM defaults to xAI Grok (api.x.ai), not Groq."""

import argparse

from tau2.cli import add_run_args
from tau2.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ENV_INTERFACE,
    DEFAULT_LLM_EVAL_USER_SIMULATOR,
    DEFAULT_LLM_NL_ASSERTIONS,
    DEFAULT_LLM_TOOL_MENTOR,
    DEFAULT_LLM_USER,
    DEFAULT_OPENAI_REALTIME_MODEL,
    DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
    DEFAULT_VOICE_SYNTHESIS_PROVIDER,
    DEFAULT_VOICE_TRANSCRIPTION_MODEL,
    DEFAULT_XAI_MODEL,
    VOICE_USER_SIMULATOR_DECISION_MODEL,
)
from tau2.data_model.simulation import TextRunConfig
from tau2.scripts.check_voice_run_readiness import build_arg_parser

GROK_CHAT = "xai/grok-4.7"
GROK_TOOL_MENTOR = "xai/grok-4.20-0309-non-reasoning"


def test_text_role_constants_use_grok_chat():
    assert DEFAULT_LLM_AGENT == GROK_CHAT
    assert DEFAULT_LLM_USER == GROK_CHAT
    assert DEFAULT_LLM_NL_ASSERTIONS == GROK_CHAT
    assert DEFAULT_LLM_ENV_INTERFACE == GROK_CHAT
    assert DEFAULT_LLM_EVAL_USER_SIMULATOR == GROK_CHAT
    assert VOICE_USER_SIMULATOR_DECISION_MODEL == GROK_CHAT
    assert DEFAULT_LLM_TOOL_MENTOR == GROK_TOOL_MENTOR


def test_unchanged_voice_and_speech_defaults():
    """Realtime voice, TTS, and STT stay on their previous providers."""
    assert DEFAULT_XAI_MODEL == "grok-voice-think-fast-1.0"
    assert DEFAULT_VOICE_SYNTHESIS_PROVIDER == "elevenlabs"
    assert DEFAULT_VOICE_TRANSCRIPTION_MODEL == "nova-3"
    assert DEFAULT_OPENAI_TRANSCRIPTION_MODEL == "gpt-4o-transcribe"
    assert DEFAULT_OPENAI_REALTIME_MODEL == "gpt-realtime-1.5"
    assert DEFAULT_GEMINI_MODEL == "gemini-3.1-flash-live-preview"


def test_run_parser_defaults_match_config_constants():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    args = parser.parse_args(["--domain", "mock"])

    assert args.agent_llm == DEFAULT_LLM_AGENT
    assert args.user_llm == DEFAULT_LLM_USER
    assert args.tool_mentor_model == DEFAULT_LLM_TOOL_MENTOR
    assert args.review_model == DEFAULT_LLM_EVAL_USER_SIMULATOR


def test_text_run_config_and_readiness_cli_follow_the_same_defaults():
    config = TextRunConfig(domain="mock")
    assert config.llm_agent == GROK_CHAT
    assert config.llm_user == GROK_CHAT

    readiness = build_arg_parser().parse_args([])
    assert readiness.tool_mentor_model == GROK_TOOL_MENTOR
