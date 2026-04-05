from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from paper_alpha_agent.config import AppSettings, get_settings
from paper_alpha_agent.llm.client import LLMClient, MockLLMClient
from paper_alpha_agent.models.paper import Paper, RankedPaper
from paper_alpha_agent.models.report import ResearchReport
from paper_alpha_agent.research.discovery import default_date_window, discover_papers
from paper_alpha_agent.research.evaluation import evaluate_prototypes
from paper_alpha_agent.research.idea_extraction import extract_ideas
from paper_alpha_agent.research.prior_art import enrich_with_prior_art
from paper_alpha_agent.research.prototype_builder import build_prototypes
from paper_alpha_agent.research.ranking import rank_papers
from paper_alpha_agent.tools.arxiv_client import ArxivClient
from paper_alpha_agent.tools.backtest_runner import BacktestRunner, SimpleBacktestRunner
from paper_alpha_agent.tools.market_data_client import DummyMarketDataClient, MarketDataClient
from paper_alpha_agent.tools.report_writer import MarkdownReportWriter
from paper_alpha_agent.tools.semantic_scholar_client import SemanticScholarClient


LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineDependencies:
    settings: AppSettings
    arxiv_client: ArxivClient
    llm_client: LLMClient
    semantic_scholar_client: SemanticScholarClient
    market_data_client: MarketDataClient
    backtest_runner: BacktestRunner
    report_writer: MarkdownReportWriter


def build_default_dependencies(settings: AppSettings | None = None) -> PipelineDependencies:
    resolved = settings or get_settings()
    return PipelineDependencies(
        settings=resolved,
        arxiv_client=ArxivClient(),
        llm_client=MockLLMClient(),
        semantic_scholar_client=SemanticScholarClient(api_key=resolved.api_keys.semantic_scholar),
        market_data_client=DummyMarketDataClient(),
        backtest_runner=SimpleBacktestRunner(),
        report_writer=MarkdownReportWriter(output_dir=resolved.report_output_path),
    )


class ResearchPipeline:
    def __init__(self, dependencies: PipelineDependencies) -> None:
        self.dependencies = dependencies

    def discover(self, start_date: str | None = None, end_date: str | None = None) -> list[Paper]:
        settings = self.dependencies.settings
        if not start_date or not end_date:
            default_start, default_end = default_date_window(settings.pipeline.date_window_days)
            start_date = start_date or default_start
            end_date = end_date or default_end
        return discover_papers(
            arxiv_client=self.dependencies.arxiv_client,
            topics=settings.arxiv_query_topics,
            max_papers=settings.pipeline.max_papers,
            start_date=start_date,
            end_date=end_date,
            default_window_days=settings.pipeline.date_window_days,
        )

    def rank(self, papers: list[Paper]) -> list[RankedPaper]:
        return rank_papers(
            papers,
            llm_client=self.dependencies.llm_client,
            relevance_threshold=self.dependencies.settings.pipeline.relevance_threshold,
        )

    def prior_art(self, papers: list[RankedPaper]) -> dict[str, list]:
        return enrich_with_prior_art(
            papers,
            llm_client=self.dependencies.llm_client,
            semantic_scholar_client=self.dependencies.semantic_scholar_client,
        )

    def select(self, papers: list[RankedPaper]) -> list[RankedPaper]:
        return papers[: self.dependencies.settings.pipeline.top_k]

    def extract_ideas(self, papers: list[RankedPaper]):
        return extract_ideas(papers, llm_client=self.dependencies.llm_client)

    def build_prototypes(self, ideas):
        return build_prototypes(ideas, llm_client=self.dependencies.llm_client)

    def backtest(self, prototypes, start_date: str, end_date: str):
        return evaluate_prototypes(
            prototypes,
            market_data_client=self.dependencies.market_data_client,
            backtest_runner=self.dependencies.backtest_runner,
            llm_client=self.dependencies.llm_client,
            start_date=start_date,
            end_date=end_date,
        )

    def report(
        self,
        discovered_count: int,
        ranked_papers,
        related_work,
        selected_papers,
        ideas,
        prototypes,
        backtests,
        start_date: str,
        end_date: str,
    ) -> ResearchReport:
        report = ResearchReport(
            created_at=datetime.utcnow(),
            universe="financial forecasting and relative value research",
            start_date=start_date,
            end_date=end_date,
            discovered_count=discovered_count,
            ranked_papers=ranked_papers,
            related_work=related_work,
            selected_paper_ids=[paper.paper_id for paper in selected_papers],
            ideas=ideas,
            prototypes=prototypes,
            backtests=backtests,
            findings=[
                "Recent ML-finance papers often emphasize forecasting but under-specify executable portfolio construction.",
                "The current skeleton favors ideas that can be prototyped with standard OHLCV-style data.",
            ],
            caveats=[
                "LLM outputs are mocked.",
                "Backtests are placeholders and not investment research.",
            ],
            next_steps=[
                "Replace the mock LLM adapter with schema-constrained real model calls.",
                "Add a production market data adapter and realistic transaction cost modeling.",
            ],
        )
        path = self.dependencies.report_writer.write(report)
        return report.model_copy(update={"report_path": str(path)})


def run_research_pipeline(start_date: str | None = None, end_date: str | None = None) -> ResearchReport:
    dependencies = build_default_dependencies()
    pipeline = ResearchPipeline(dependencies)
    if not start_date or not end_date:
        start_date, end_date = default_date_window(dependencies.settings.pipeline.date_window_days)
    LOGGER.info("Running bounded research pipeline for %s to %s", start_date, end_date)
    discovered = pipeline.discover(start_date=start_date, end_date=end_date)
    ranked = pipeline.rank(discovered)
    selected = pipeline.select(ranked)
    related = pipeline.prior_art(selected)
    ideas = pipeline.extract_ideas(selected)
    prototypes = pipeline.build_prototypes(ideas)
    backtests = pipeline.backtest(prototypes, start_date=start_date, end_date=end_date)
    return pipeline.report(
        discovered_count=len(discovered),
        ranked_papers=ranked,
        related_work=related,
        selected_papers=selected,
        ideas=ideas,
        prototypes=prototypes,
        backtests=backtests,
        start_date=start_date,
        end_date=end_date,
    )
