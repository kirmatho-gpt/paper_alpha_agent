from __future__ import annotations

from datetime import datetime, timezone

from paper_alpha_agent.models.paper import Paper
from paper_alpha_agent.research.ranking import heuristic_relevance_score, shortlist_papers_for_ranking


def _paper(paper_id: str, title: str, abstract: str, categories: list[str]) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=["A"],
        categories=categories,
        published=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )


def test_heuristic_relevance_score_prefers_forecasting_finance_paper():
    finance_paper = _paper(
        "1",
        "Transformer futures return forecasting",
        "Daily futures return forecasting with transformer models and market data.",
        ["q-fin.ST", "cs.LG"],
    )
    generic_paper = _paper(
        "2",
        "Enterprise resource allocation benchmark",
        "Large language agents for enterprise resource allocation and CFO decisions.",
        ["cs.AI"],
    )

    assert heuristic_relevance_score(finance_paper) > heuristic_relevance_score(generic_paper)


def test_shortlist_papers_for_ranking_filters_obvious_mismatch():
    papers = [
        _paper("1", "Futures forecasting", "Daily futures return forecasting with neural networks.", ["q-fin.ST"]),
        _paper("2", "EnterpriseArena", "LLM benchmark for enterprise CFO resource allocation.", ["cs.AI"]),
        _paper("3", "Bond volatility forecast", "Forecast bond volatility with machine learning.", ["q-fin.ST"]),
    ]

    shortlisted = shortlist_papers_for_ranking(papers, shortlist_size=2)

    assert len(shortlisted) == 2
    assert [paper.paper_id for paper in shortlisted] == ["1", "3"]
