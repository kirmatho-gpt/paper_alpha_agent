from __future__ import annotations

from paper_alpha_agent.config import get_settings


def test_settings_load_with_env_override(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("PAPER_ALPHA_AGENT__LLM__MODEL_NAME", "test-model")
    settings = get_settings()

    assert settings.llm.model_name == "test-model"
    assert settings.pipeline.max_papers > 0
    assert len(settings.arxiv_query_topics) > 0
    assert "rank_paper_relevance" in settings.prompts

    get_settings.cache_clear()
