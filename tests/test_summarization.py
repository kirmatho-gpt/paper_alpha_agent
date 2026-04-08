from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import pytest

from paper_alpha_agent.config import AppSettings
from paper_alpha_agent.llm.client import MockLLMClient, OpenAILLMClient
from paper_alpha_agent.llm.prompts import PromptLibrary
from paper_alpha_agent.llm.schemas import PaperSummaryResponse
from paper_alpha_agent.models.paper import Paper


def _sample_paper() -> Paper:
    return Paper(
        paper_id="paper-1",
        title="Transformer Models for Daily Futures Return Forecasting",
        abstract=(
            "We use a transformer architecture for daily futures return forecasting "
            "with financial price and volume data."
        ),
        authors=["A. Researcher"],
        categories=["q-fin.ST", "cs.LG"],
        published=datetime(2024, 1, 1, tzinfo=timezone.utc),
        query_topic="futures",
    )


def test_mock_summarize_paper_returns_deterministic_summary():
    summary = MockLLMClient().summarize_paper(_sample_paper())

    assert "transformer" in summary.summary.lower()
    assert "future returns" in summary.summary.lower()
    assert summary.relevance_label == "directly_relevant"
    assert len(summary.why_relevant) > 0
    assert summary.model_family == "transformer"
    assert summary.prediction_target == "future returns"
    assert summary.forecast_horizon == "daily"
    assert len(summary.implementation_constraints) > 0


def test_openai_summarize_paper_uses_summary_prompt(monkeypatch):
    calls: list[tuple[type[object], list[dict[str, str]]]] = []

    def fake_init(self, settings):
        self._model = settings.llm.model_name
        self._prompts = PromptLibrary(settings)
        self._cache_dir = Path("/tmp/paper_alpha_agent_test_cache")

    def fake_parse_cached(self, operation, schema, input_messages, cache_identity):
        calls.append((schema, input_messages))
        return PaperSummaryResponse(
            summary="Structured summary from OpenAI",
            relevance_label="directly_relevant",
        )

    monkeypatch.setattr(OpenAILLMClient, "__init__", fake_init)
    monkeypatch.setattr(OpenAILLMClient, "_parse_cached", fake_parse_cached)

    settings = AppSettings.model_validate(
        {
            "llm": {"model_name": "gpt-4o-mini"},
            "api_keys": {"openai": "test-key"},
            "prompts": {"summarize_paper": "Summarize this paper clearly."},
        }
    )
    client = OpenAILLMClient(settings)

    summary = client.summarize_paper(_sample_paper())

    assert summary.summary == "Structured summary from OpenAI"
    assert summary.relevance_label == "directly_relevant"
    assert len(calls) == 1
    schema, messages = calls[0]
    assert schema is PaperSummaryResponse
    assert messages[0]["content"] == "Summarize this paper clearly."
    assert "Abstract:" in messages[1]["content"]


@pytest.mark.parametrize(
    ("abstract", "expected_fragment"),
    [
        ("This paper studies daily price forecasting with LSTM models.", "daily horizon"),
        ("This paper studies price forecasting with transformer models.", "future prices"),
    ],
)
def test_mock_summarize_paper_uses_abstract_signals(abstract: str, expected_fragment: str):
    paper = _sample_paper().model_copy(update={"abstract": abstract})

    summary = MockLLMClient().summarize_paper(paper)

    assert expected_fragment in summary.summary.lower() or expected_fragment == summary.prediction_target


@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_LIVE_TEST") != "1",
    reason="Set RUN_OPENAI_LIVE_TEST=1 to run the live OpenAI summarization test.",
)
def test_openai_live_summarize_paper():
    api_key = os.getenv("PAPER_ALPHA_AGENT__API_KEYS__OPENAI")
    if not api_key:
        pytest.skip("PAPER_ALPHA_AGENT__API_KEYS__OPENAI is not set.")

    settings = AppSettings.model_validate(
        {
            "llm": {"model_name": "gpt-4o-mini"},
            "api_keys": {"openai": api_key},
            "prompts": {
                "summarize_paper": (
                    "Summarize the paper in 2-4 sentences. Focus on model family, "
                    "prediction target, forecast horizon, and implementation constraints."
                )
            },
        }
    )
    client = OpenAILLMClient(settings)

    summary = client.summarize_paper(_sample_paper())

    assert isinstance(summary, PaperSummaryResponse)
    assert len(summary.summary.strip()) > 20
    print("\nLive OpenAI summary:\n")
    print(summary.model_dump_json(indent=2))
