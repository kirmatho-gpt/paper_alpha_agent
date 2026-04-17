from __future__ import annotations

from datetime import datetime, timezone

from paper_alpha_agent.llm.client import MockLLMClient
from paper_alpha_agent.models.paper import Paper
from paper_alpha_agent.research.ranking import rank_papers


def test_rank_papers_orders_by_composite_score():
    papers = [
        Paper(
            paper_id="a",
            title="Financial Forecasting",
            abstract="Forecast return and price moves in financial markets.",
            authors=["A"],
            categories=["q-fin.ST"],
            published=datetime.now(timezone.utc),
        ),
        Paper(
            paper_id="b",
            title="Generic Deep Learning",
            abstract="Representation learning for images.",
            authors=["B"],
            categories=["cs.LG"],
            published=datetime.now(timezone.utc),
        ),
    ]

    ranked = rank_papers(papers, llm_client=MockLLMClient(allow_mock=True), relevance_threshold=0.0)

    assert ranked[0].paper_id == "a"
    assert ranked[0].composite_score >= ranked[1].composite_score
