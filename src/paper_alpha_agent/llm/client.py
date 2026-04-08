from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from paper_alpha_agent.config import AppSettings
from paper_alpha_agent.llm.prompts import PromptLibrary
from paper_alpha_agent.llm.schemas import (
    BacktestCritiqueResponse,
    IdeaExtractionResponse,
    PaperRankingResponse,
    PaperSummaryResponse,
    PriorArtAssessmentResponse,
    PrototypeSpecResponse,
)
from paper_alpha_agent.models.backtest import BacktestResult
from paper_alpha_agent.models.idea import PrototypeSpec, ResearchIdea
from paper_alpha_agent.models.paper import Paper, RankedPaper, RelatedWorkItem
from paper_alpha_agent.tools.storage import read_json, write_json


LOGGER = logging.getLogger(__name__)
DEFAULT_OPENAI_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "openai"


class LLMClient(ABC):
    @abstractmethod
    def rank_paper_relevance(self, paper: Paper) -> PaperRankingResponse:
        raise NotImplementedError

    @abstractmethod
    def summarize_paper(self, paper: Paper) -> PaperSummaryResponse:
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
        paper_summary = self.summarize_paper(paper)
        return PaperRankingResponse(
            relevance_score=round(score, 2),
            implementability_score=0.7,
            novelty_score=0.62,
            summary=paper_summary.summary,
            horizon="1d to 5d" if "daily" in abstract or "day" in abstract else "unspecified",
            frequency="daily",
            asset_classes=["equities", "futures"] if "equity" in abstract else ["multi-asset"],
            rationale=[
                "Paper mentions forecast-oriented modeling.",
                "Method appears simple enough to prototype with public data.",
            ],
        )

    def summarize_paper(self, paper: Paper) -> PaperSummaryResponse:
        abstract = paper.abstract.lower()
        if "transformer" in abstract:
            model_family = "transformer"
        elif "lstm" in abstract or "rnn" in abstract:
            model_family = "recurrent neural network"
        else:
            model_family = "machine learning model"

        if any(term in abstract for term in {"forecast", "prediction", "predict", "return", "price"}):
            relevance_label = "directly_relevant"
            why_relevant = ["Paper appears to address predictive modeling for financial variables."]
        elif any(term in abstract for term in {"volatility", "risk", "liquidity"}):
            relevance_label = "adjacent"
            why_relevant = ["Paper is financially relevant but may be more about risk or market structure than alpha generation."]
        else:
            relevance_label = "out_of_scope"
            why_relevant = ["Paper does not clearly describe a forecasting or relative value signal problem."]

        if "return" in abstract:
            prediction_target = "future returns"
        elif "price" in abstract:
            prediction_target = "future prices"
        elif "volatility" in abstract:
            prediction_target = "future volatility"
        else:
            prediction_target = None

        horizon = "daily" if "daily" in abstract or "day" in abstract else "unspecified"
        if "futures" in abstract or "futures" in paper.title.lower():
            asset_class = "futures"
        elif "equity" in abstract or "equities" in abstract:
            asset_class = "equities"
        elif "bond" in abstract or "fixed income" in abstract:
            asset_class = "fixed income"
        elif "crypto" in abstract or "cryptocurrency" in abstract:
            asset_class = "crypto"
        else:
            asset_class = None
        data_context = "financial price and volume data" if "volume" in abstract or "price" in abstract else "financial data"
        takeaways = ["Start with a narrow prototype and explicit target definition.", "Validate against simple baselines."]
        missing_information = []
        if horizon == "unspecified":
            missing_information.append("Exact forecast horizon is unclear from the abstract.")
        if prediction_target is None:
            missing_information.append("Prediction target is not clearly specified in the abstract.")
        constraints = ["Needs real dataset mapping", "Needs realistic transaction cost assumptions"]
        return PaperSummaryResponse(
            summary=(
                f"{paper.title} describes a {model_family} approach"
                f"{f' for {prediction_target}' if prediction_target else ''} with a "
                f"{horizon} horizon, using financial data assumptions that appear simple enough for an initial prototype."
            ),
            relevance_label=relevance_label,
            why_relevant=why_relevant,
            model_family=model_family,
            prediction_target=prediction_target,
            forecast_horizon=horizon,
            asset_class=asset_class,
            data_context=data_context,
            implementation_takeaways=takeaways,
            missing_information=missing_information,
            caveats=["Abstract-only summary; important implementation details may be omitted."],
            implementation_constraints=constraints,
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


class OpenAILLMClient(LLMClient):
    """OpenAI-backed implementation using typed structured outputs.

    This uses the official Python SDK's `responses.parse(...)` helper with
    Pydantic schemas, following the Structured Outputs guidance in the official
    OpenAI docs.
    """

    def __init__(self, settings: AppSettings) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The `openai` package is required for OpenAILLMClient.") from exc

        if not settings.api_keys.openai:
            raise ValueError("OpenAI API key is required for OpenAILLMClient.")

        self._sdk = OpenAI(api_key=settings.api_keys.openai)
        self._model = settings.llm.model_name
        self._prompts = PromptLibrary(settings)
        self._cache_dir = DEFAULT_OPENAI_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def rank_paper_relevance(self, paper: Paper) -> PaperRankingResponse:
        return self._parse_cached(
            operation="rank_paper_relevance",
            schema=PaperRankingResponse,
            input_messages=self._prompts.ranking_messages(paper),
            cache_identity={"paper_id": paper.paper_id, "updated": paper.updated.isoformat() if paper.updated else None},
        )

    def summarize_paper(self, paper: Paper) -> PaperSummaryResponse:
        return self._parse_cached(
            operation="summarize_paper",
            schema=PaperSummaryResponse,
            input_messages=self._prompts.summary_messages(paper),
            cache_identity={"paper_id": paper.paper_id, "updated": paper.updated.isoformat() if paper.updated else None},
        )

    def assess_prior_art(self, paper: RankedPaper) -> PriorArtAssessmentResponse:
        return self._parse(PriorArtAssessmentResponse, self._prompts.prior_art_messages(paper))

    def extract_research_idea(self, paper: RankedPaper) -> IdeaExtractionResponse:
        return self._parse(IdeaExtractionResponse, self._prompts.idea_messages(paper))

    def build_prototype_spec(self, idea: ResearchIdea) -> PrototypeSpecResponse:
        return self._parse(PrototypeSpecResponse, self._prompts.prototype_messages(idea))

    def critique_backtest(self, result: BacktestResult) -> BacktestCritiqueResponse:
        return self._parse(BacktestCritiqueResponse, self._prompts.critique_messages(result))

    def _parse(self, schema: type[Any], input_messages: list[dict[str, str]]) -> Any:
        response = self._sdk.responses.parse(
            model=self._model,
            input=input_messages,
            text_format=schema,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("OpenAI response did not contain a parsed structured output.")
        return parsed

    def _parse_cached(
        self,
        operation: str,
        schema: type[Any],
        input_messages: list[dict[str, str]],
        cache_identity: dict[str, Any],
    ) -> Any:
        cache_path = self._cache_path(
            operation=operation,
            schema_name=schema.__name__,
            input_messages=input_messages,
            cache_identity=cache_identity,
        )
        if cache_path.exists():
            LOGGER.info("Using cached OpenAI response for operation=%s from %s", operation, cache_path)
            payload = read_json(cache_path)
            return schema.model_validate(payload["parsed"])

        parsed = self._parse(schema, input_messages)
        write_json(
            cache_path,
            {
                "operation": operation,
                "model": self._model,
                "schema": schema.__name__,
                "cache_identity": cache_identity,
                "input_messages": input_messages,
                "parsed": parsed.model_dump(mode="json"),
            },
        )
        return parsed

    def _cache_path(
        self,
        operation: str,
        schema_name: str,
        input_messages: list[dict[str, str]],
        cache_identity: dict[str, Any],
    ) -> Path:
        material = {
            "operation": operation,
            "model": self._model,
            "schema": schema_name,
            "cache_identity": cache_identity,
            "input_messages": input_messages,
        }
        digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        paper_id = str(cache_identity.get("paper_id", "object"))
        safe_prefix = "".join(char.lower() if char.isalnum() else "_" for char in paper_id)[:60]
        return self._cache_dir / f"{safe_prefix}_{operation}_{digest}.json"
