from __future__ import annotations

from paper_alpha_agent.llm.client import LLMClient
from paper_alpha_agent.models.paper import RankedPaper, RelatedWorkItem
from paper_alpha_agent.tools.semantic_scholar_client import SemanticScholarClient


def enrich_with_prior_art(
    papers: list[RankedPaper],
    llm_client: LLMClient,
    semantic_scholar_client: SemanticScholarClient,
) -> dict[str, list[RelatedWorkItem]]:
    result: dict[str, list[RelatedWorkItem]] = {}
    for paper in papers:
        llm_items = llm_client.assess_prior_art(paper).items
        semantic_items = [
            RelatedWorkItem(
                title=item.title,
                url=item.url,
                overlap_summary=item.abstract or "Stubbed related work item.",
                similarity_score=0.5,
            )
            for item in semantic_scholar_client.find_similar_papers(paper.title).data
        ]
        result[paper.paper_id] = llm_items + semantic_items
    return result
