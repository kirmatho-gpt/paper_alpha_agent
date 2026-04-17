from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from paper_alpha_agent.config import AppSettings, get_settings
from paper_alpha_agent.llm.client import LLMClient, OpenAILLMClient
from paper_alpha_agent.models.paper import (
    ALPHA_PRIORITY,
    FullPaperSummary,
    Paper,
    RankedPaper,
    SummarizedPaper,
    TopicSummaryBatch,
    TopicSummaryStageResult,
)
from paper_alpha_agent.models.report import ResearchReport
from paper_alpha_agent.research.discovery import default_date_window, discover_papers
from paper_alpha_agent.research.evaluation import evaluate_prototypes
from paper_alpha_agent.research.idea_extraction import extract_ideas
from paper_alpha_agent.research.prior_art import enrich_with_prior_art
from paper_alpha_agent.research.prototype_builder import build_prototypes
from paper_alpha_agent.research.ranking import heuristic_relevance_score, rank_papers, shortlist_papers_for_ranking
from paper_alpha_agent.tools.arxiv_client import ArxivClient
from paper_alpha_agent.tools.backtest_runner import BacktestRunner, SimpleBacktestRunner
from paper_alpha_agent.tools.market_data_client import DummyMarketDataClient, MarketDataClient
from paper_alpha_agent.tools.report_writer import MarkdownReportWriter
from paper_alpha_agent.tools.semantic_scholar_client import SemanticScholarClient


LOGGER = logging.getLogger(__name__)

DEFAULT_TOPIC_SUMMARIZATION_TOPICS = ["markets", "bonds", "forecasting", "commodities", "stocks"]
DEFAULT_TOPIC_FETCH_LIMIT = 30
DEFAULT_TOPIC_SUMMARY_LIMIT = 7
DEFAULT_TOPIC_FULL_PAPER_LIMIT = 10


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
    if not resolved.api_keys.openai:
        raise RuntimeError(
            "OpenAI API key is required to build default pipeline dependencies. "
            "Set PAPER_ALPHA_AGENT__API_KEYS__OPENAI or inject an explicit llm_client."
        )

    llm_client: LLMClient = OpenAILLMClient(resolved)
    return PipelineDependencies(
        settings=resolved,
        arxiv_client=ArxivClient(),
        llm_client=llm_client,
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

    def summarize_topics(
        self,
        topics: list[str] | None = None,
        fetch_limit: int = DEFAULT_TOPIC_FETCH_LIMIT,
        summary_limit: int = DEFAULT_TOPIC_SUMMARY_LIMIT,
        full_paper_limit: int = DEFAULT_TOPIC_FULL_PAPER_LIMIT,
        log_heuristic_decisions: bool = False,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> TopicSummaryStageResult:
        settings = self.dependencies.settings
        if not start_date or not end_date:
            default_start, default_end = default_date_window(settings.pipeline.date_window_days)
            start_date = start_date or default_start
            end_date = end_date or default_end

        resolved_topics = topics or DEFAULT_TOPIC_SUMMARIZATION_TOPICS
        batches: list[TopicSummaryBatch] = []
        directly_relevant_papers: list[SummarizedPaper] = []

        for topic in resolved_topics:
            fetched_papers = self.dependencies.arxiv_client.search(
                query=topic,
                max_results=fetch_limit,
                start_date=start_date,
                end_date=end_date,
            )
            shortlisted_papers = shortlist_papers_for_ranking(
                fetched_papers,
                shortlist_size=min(summary_limit, len(fetched_papers)),
                log_decisions=log_heuristic_decisions,
            )
            summarized_papers = [
                self._summarize_paper(paper, summary_rank=index)
                for index, paper in enumerate(shortlisted_papers, start=1)
            ]
            directly_relevant_for_topic = [
                paper for paper in summarized_papers if paper.relevance_label == "directly_relevant"
            ]
            directly_relevant_papers.extend(directly_relevant_for_topic)
            batches.append(
                TopicSummaryBatch(
                    topic=topic,
                    fetched_count=len(fetched_papers),
                    summarized_papers=summarized_papers,
                    directly_relevant_papers=directly_relevant_for_topic,
                )
            )

        filtered_papers = self._filter_topic_summary_candidates(directly_relevant_papers)
        full_paper_summaries = [self._summarize_full_paper(paper) for paper in filtered_papers[:full_paper_limit]]

        return TopicSummaryStageResult(
            topics=resolved_topics,
            fetch_limit=fetch_limit,
            summary_limit=summary_limit,
            full_paper_limit=full_paper_limit,
            batches=batches,
            directly_relevant_papers=directly_relevant_papers,
            filtered_papers=filtered_papers,
            full_paper_summaries=full_paper_summaries,
        )

    def rank(self, papers: list[Paper]) -> list[RankedPaper]:
        return rank_papers(
            papers,
            llm_client=self.dependencies.llm_client,
            relevance_threshold=self.dependencies.settings.pipeline.relevance_threshold,
        )

    def _summarize_paper(self, paper: Paper, summary_rank: int | None = None) -> SummarizedPaper:
        summary = self.dependencies.llm_client.summarize_paper(paper)
        return SummarizedPaper.model_validate(
            {
                **paper.model_dump(mode="json"),
                "summary_rank": summary_rank,
                **summary.model_dump(mode="json"),
            }
        )

    def _summarize_full_paper(self, paper: SummarizedPaper) -> FullPaperSummary:
        full_text = self.dependencies.arxiv_client.fetch_full_text(paper)
        response = self.dependencies.llm_client.summarize_full_paper(paper, full_text)
        return FullPaperSummary.model_validate(
            {
                **paper.model_dump(mode="json"),
                "full_text_summary": response.summary,
                "alpha_thesis": response.alpha_thesis,
                "implementation_complexity": response.implementation_complexity,
                "strategy_quality": response.strategy_quality,
                "sharpe_ratio": response.sharpe_ratio,
                "sharpe_ratio_context": response.sharpe_ratio_context,
                "evidence": response.evidence,
                "implementation_requirements": response.implementation_requirements,
                "key_risks": response.key_risks,
                "full_text_char_count": len(full_text),
            }
        )

    @staticmethod
    def _filter_topic_summary_candidates(papers: list[SummarizedPaper]) -> list[SummarizedPaper]:
        filtered = [
            paper
            for paper in papers
            if paper.relevance_label == "directly_relevant"
            and paper.implementable_alpha_label in {"yes", "likely"}
        ]
        deduped: dict[str, SummarizedPaper] = {}
        for paper in filtered:
            incumbent = deduped.get(paper.paper_id)
            if incumbent is None:
                deduped[paper.paper_id] = paper
                continue
            if ResearchPipeline._topic_summary_sort_key(paper) < ResearchPipeline._topic_summary_sort_key(incumbent):
                deduped[paper.paper_id] = paper

        reranked = sorted(
            deduped.values(),
            key=lambda paper: (
                ALPHA_PRIORITY.get(paper.implementable_alpha_label, 99),
                -heuristic_relevance_score(paper),
                paper.summary_rank if paper.summary_rank is not None else 999,
            ),
        )
        return [
            paper.model_copy(
                update={
                    "global_rank": index,
                    "global_relevance_score": heuristic_relevance_score(paper),
                }
            )
            for index, paper in enumerate(reranked, start=1)
        ]

    @staticmethod
    def _topic_summary_sort_key(paper: SummarizedPaper) -> tuple[int, float, int]:
        return (
            ALPHA_PRIORITY.get(paper.implementable_alpha_label, 99),
            -heuristic_relevance_score(paper),
            paper.summary_rank if paper.summary_rank is not None else 999,
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
