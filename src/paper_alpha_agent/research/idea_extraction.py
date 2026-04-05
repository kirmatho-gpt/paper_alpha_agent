from __future__ import annotations

from paper_alpha_agent.llm.client import LLMClient
from paper_alpha_agent.models.idea import ResearchIdea
from paper_alpha_agent.models.paper import RankedPaper


def extract_ideas(papers: list[RankedPaper], llm_client: LLMClient) -> list[ResearchIdea]:
    return [llm_client.extract_research_idea(paper).idea for paper in papers]
