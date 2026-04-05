from __future__ import annotations

from abc import ABC, abstractmethod

from paper_alpha_agent.llm.schemas import (
    BacktestCritiqueResponse,
    IdeaExtractionResponse,
    PaperRankingResponse,
    PriorArtAssessmentResponse,
    PrototypeSpecResponse,
)
from paper_alpha_agent.models.backtest import BacktestResult
from paper_alpha_agent.models.idea import PrototypeSpec, ResearchIdea
from paper_alpha_agent.models.paper import Paper, RankedPaper, RelatedWorkItem


class LLMClient(ABC):
    @abstractmethod
    def rank_paper_relevance(self, paper: Paper) -> PaperRankingResponse:
        raise NotImplementedError

    @abstractmethod
    def summarize_paper(self, paper: Paper) -> str:
        raise NotImplementedError

    @abstractmethod
    def assess_prior_art(self, paper: RankedPaper) -> PriorArtAssessmentResponse:
        raise NotImplementedError

    @abstractmethod
    def extract_research_idea(self, paper: RankedPaper) -> IdeaExtractionResponse:
        raise NotImplementedError

    @abstractmethod
    def build_prototype_spec(self, idea: ResearchIdea) -> PrototypeSpecResponse:
        raise NotImplementedError

    @abstractmethod
    def critique_backtest(self, result: BacktestResult) -> BacktestCritiqueResponse:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    def rank_paper_relevance(self, paper: Paper) -> PaperRankingResponse:
        abstract = paper.abstract.lower()
        finance_terms = ["forecast", "return", "asset", "financial", "price", "market", "trading"]
        hits = sum(term in abstract for term in finance_terms)
        score = min(1.0, 0.35 + hits * 0.08)
        return PaperRankingResponse(
            relevance_score=round(score, 2),
            implementability_score=0.7,
            novelty_score=0.62,
            summary=self.summarize_paper(paper),
            horizon="1d to 5d" if "daily" in abstract or "day" in abstract else "unspecified",
            frequency="daily",
            asset_classes=["equities", "futures"] if "equity" in abstract else ["multi-asset"],
            rationale=[
                "Paper mentions forecast-oriented modeling.",
                "Method appears simple enough to prototype with public data.",
            ],
        )

    def summarize_paper(self, paper: Paper) -> str:
        return (
            f"{paper.title} studies ML-based financial prediction with emphasis on "
            f"implementable signals, data assumptions, and practical evaluation."
        )

    def assess_prior_art(self, paper: RankedPaper) -> PriorArtAssessmentResponse:
        items = [
            RelatedWorkItem(
                title=f"Related sequence-model forecasting work for {paper.title[:40]}",
                overlap_summary="Similar modeling family but unclear relative value framing.",
                similarity_score=0.67,
                url=str(paper.entry_url) if paper.entry_url else None,
            )
        ]
        return PriorArtAssessmentResponse(items=items)

    def extract_research_idea(self, paper: RankedPaper) -> IdeaExtractionResponse:
        idea = ResearchIdea(
            idea_id=f"idea-{paper.paper_id}",
            paper_id=paper.paper_id,
            title=f"Cross-sectional residual forecast from {paper.title[:60]}",
            hypothesis="Model latent residual mispricing and trade mean reversion in the cross section.",
            signal_definition="Rank assets by predicted residual return and form long-short deciles.",
            target_universe=["liquid equities", "index futures"],
            forecast_horizon=paper.horizon or "1d",
            frequency=paper.frequency or "daily",
            required_data=["OHLCV", "sector labels", "benchmark returns"],
            implementation_steps=[
                "Create lagged returns and volatility features.",
                "Fit a simple forecasting model on residualized returns.",
                "Construct a cross-sectional rank signal.",
            ],
            caveats=["Toy extraction only", "Needs transaction cost modeling"],
        )
        return IdeaExtractionResponse(idea=idea)

    def build_prototype_spec(self, idea: ResearchIdea) -> PrototypeSpecResponse:
        prototype = PrototypeSpec(
            prototype_id=f"prototype-{idea.idea_id}",
            idea_id=idea.idea_id,
            title=f"Prototype for {idea.title}",
            objective=idea.hypothesis,
            feature_set=["lagged returns", "rolling volatility", "cross-sectional z-scores"],
            labels=[f"forward_return_{idea.forecast_horizon}"],
            data_requirements=idea.required_data,
            modeling_approach="Begin with linear baseline, then swap in tree or transformer models later.",
            signal_logic=idea.signal_definition,
            evaluation_plan=[
                "Train on rolling windows.",
                "Rank by predicted return.",
                "Evaluate long-short spread and hit rate.",
            ],
            risk_controls=["Sector neutralization", "Position caps", "Volatility scaling"],
        )
        return PrototypeSpecResponse(prototype=prototype)

    def critique_backtest(self, result: BacktestResult) -> BacktestCritiqueResponse:
        return BacktestCritiqueResponse(
            critique=(
                "This toy backtest ignores realistic costs, slippage, borrow constraints, and "
                "regime instability. Treat the output as plumbing validation only."
            )
        )
