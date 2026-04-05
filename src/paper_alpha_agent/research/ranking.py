from __future__ import annotations

from paper_alpha_agent.llm.client import LLMClient
from paper_alpha_agent.models.paper import Paper, RankedPaper


def rank_papers(papers: list[Paper], llm_client: LLMClient, relevance_threshold: float = 0.0) -> list[RankedPaper]:
    ranked: list[RankedPaper] = []
    for paper in papers:
        response = llm_client.rank_paper_relevance(paper)
        item = RankedPaper(
            **paper.model_dump(),
            **response.model_dump(),
        )
        if item.relevance_score >= relevance_threshold:
            ranked.append(item)
    ranked.sort(key=lambda paper: paper.composite_score, reverse=True)
    return ranked
