from __future__ import annotations

from paper_alpha_agent.config import AppSettings
from paper_alpha_agent.models.backtest import BacktestResult
from paper_alpha_agent.models.idea import ResearchIdea
from paper_alpha_agent.models.paper import Paper, RankedPaper


class PromptLibrary:
    def __init__(self, settings: AppSettings) -> None:
        self._prompts = settings.prompts

    def get(self, name: str) -> str:
        return self._prompts.get(name, "")

    def ranking_messages(self, paper: Paper) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.get("rank_paper_relevance")},
            {
                "role": "user",
                "content": (
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(paper.authors)}\n"
                    f"Published: {paper.published.isoformat()}\n"
                    f"Categories: {', '.join(paper.categories)}\n"
                    f"Topic: {paper.query_topic}\n\n"
                    f"Abstract:\n{paper.abstract}"
                ),
            },
        ]

    def summary_messages(self, paper: Paper) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.get("summarize_paper")},
            {"role": "user", "content": f"Title: {paper.title}\n\nAbstract:\n{paper.abstract}"},
        ]

    def prior_art_messages(self, paper: RankedPaper) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.get("assess_prior_art")},
            {"role": "user", "content": f"Title: {paper.title}\n\nSummary:\n{paper.summary}"},
        ]

    def idea_messages(self, paper: RankedPaper) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.get("extract_research_idea")},
            {"role": "user", "content": f"Title: {paper.title}\n\nAbstract:\n{paper.abstract}"},
        ]

    def prototype_messages(self, idea: ResearchIdea) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.get("build_prototype_spec")},
            {"role": "user", "content": idea.model_dump_json(indent=2)},
        ]

    def critique_messages(self, result: BacktestResult) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.get("critique_backtest")},
            {"role": "user", "content": result.model_dump_json(indent=2)},
        ]
