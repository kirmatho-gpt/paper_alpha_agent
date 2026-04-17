from __future__ import annotations

from datetime import datetime, timezone

from paper_alpha_agent.config import AppSettings
from paper_alpha_agent.llm.client import LLMClient, MockLLMClient
from paper_alpha_agent.llm.schemas import (
    BacktestCritiqueResponse,
    FullPaperSummaryResponse,
    IdeaExtractionResponse,
    PaperSummaryResponse,
    PriorArtAssessmentResponse,
    PrototypeSpecResponse,
)
from paper_alpha_agent.models.idea import PrototypeSpec, ResearchIdea
from paper_alpha_agent.models.paper import Paper, RankedPaper
from paper_alpha_agent.orchestration.pipeline import (
    DEFAULT_TOPIC_FETCH_LIMIT,
    DEFAULT_TOPIC_SUMMARY_LIMIT,
    PipelineDependencies,
    ResearchPipeline,
)
from paper_alpha_agent.tools.backtest_runner import SimpleBacktestRunner
from paper_alpha_agent.tools.market_data_client import DummyMarketDataClient
from paper_alpha_agent.tools.report_writer import MarkdownReportWriter
from paper_alpha_agent.tools.semantic_scholar_client import SemanticScholarClient


class StubArxivClient:
    def search(
        self,
        query: str,
        max_results: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        return [
            Paper(
                paper_id=f"{query}-{index}",
                title=f"{query} paper {index}",
                abstract="Daily financial return forecasting with cross-sectional signals.",
                authors=["Researcher"],
                categories=["q-fin.ST"],
                published=datetime(2024, 1, 10, tzinfo=timezone.utc),
                query_topic=query,
            )
            for index in range(1, max_results + 1)
        ]

    @staticmethod
    def deduplicate(papers):
        unique = {paper.paper_id: paper for paper in papers}
        return list(unique.values())

    def fetch_full_text(self, paper: Paper, max_chars: int = 120_000) -> str:
        return f"Full paper text for {paper.paper_id} with signal construction, long-short portfolio, and Sharpe ratio discussion."[:max_chars]


class StubTopicSummaryLLMClient(LLMClient):
    def rank_paper_relevance(self, paper: Paper):
        raise NotImplementedError

    def summarize_paper(self, paper: Paper) -> PaperSummaryResponse:
        mapping = {
            "shared-1": ("directly_relevant", "likely"),
            "markets-1": ("directly_relevant", "yes"),
            "markets-2": ("directly_relevant", "likely"),
            "markets-3": ("adjacent", "yes"),
            "bonds-1": ("directly_relevant", "yes"),
            "bonds-2": ("directly_relevant", "likely"),
            "bonds-3": ("directly_relevant", "no"),
        }
        relevance_label, alpha_label = mapping[paper.paper_id]
        return PaperSummaryResponse(
            summary=f"Summary for {paper.paper_id}",
            relevance_label=relevance_label,
            implementable_alpha_label=alpha_label,
            why_relevant=["stub"],
        )

    def summarize_full_paper(self, paper: Paper, full_text: str) -> FullPaperSummaryResponse:
        return FullPaperSummaryResponse(
            summary=f"Full summary for {paper.paper_id}",
            implementable_alpha_label="yes" if paper.paper_id.endswith("-1") else "likely",
            alpha_thesis=f"Alpha thesis for {paper.paper_id}",
            implementation_complexity="medium",
            strategy_quality="promising",
            sharpe_ratio=1.2,
            sharpe_ratio_context="Reported in the paper.",
            evidence=["evidence"],
            implementation_requirements=["data", "backtest"],
            key_risks=["risk"],
        )

    def assess_prior_art(self, paper: RankedPaper) -> PriorArtAssessmentResponse:
        return PriorArtAssessmentResponse()

    def extract_research_idea(self, paper: RankedPaper) -> IdeaExtractionResponse:
        return IdeaExtractionResponse(
            idea=ResearchIdea(
                idea_id="idea",
                paper_id=paper.paper_id,
                title="idea",
                hypothesis="hypothesis",
                signal_definition="signal",
                target_universe=["equities"],
                forecast_horizon="1d",
                frequency="daily",
                required_data=["ohlcv"],
            )
        )

    def build_prototype_spec(self, idea: ResearchIdea) -> PrototypeSpecResponse:
        return PrototypeSpecResponse(
            prototype=PrototypeSpec(
                prototype_id="prototype",
                idea_id=idea.idea_id,
                title="prototype",
                objective="objective",
                feature_set=["x"],
                labels=["y"],
                data_requirements=["ohlcv"],
                modeling_approach="linear",
                signal_logic="signal",
                evaluation_plan=["eval"],
            )
        )

    def critique_backtest(self, result) -> BacktestCritiqueResponse:
        return BacktestCritiqueResponse(critique="critique")


def test_pipeline_runs_end_to_end(tmp_path):
    settings = AppSettings.model_validate(
        {
            "arxiv_query_topics": ["topic-a", "topic-b"],
            "prompts": {},
            "pipeline": {
                "date_window_days": 30,
                "max_papers": 10,
                "top_k": 2,
                "relevance_threshold": 0.0,
            },
            "reporting": {"output_dir": str(tmp_path)},
        }
    )
    dependencies = PipelineDependencies(
        settings=settings,
        arxiv_client=StubArxivClient(),
        llm_client=MockLLMClient(allow_mock=True),
        semantic_scholar_client=SemanticScholarClient(),
        market_data_client=DummyMarketDataClient(),
        backtest_runner=SimpleBacktestRunner(),
        report_writer=MarkdownReportWriter(tmp_path),
    )
    pipeline = ResearchPipeline(dependencies)

    discovered = pipeline.discover(start_date="2024-01-01", end_date="2024-12-31")
    ranked = pipeline.rank(discovered)
    selected = pipeline.select(ranked)
    related = pipeline.prior_art(selected)
    ideas = pipeline.extract_ideas(selected)
    prototypes = pipeline.build_prototypes(ideas)
    backtests = pipeline.backtest(prototypes, start_date="2024-01-01", end_date="2024-12-31")
    report = pipeline.report(
        discovered_count=len(discovered),
        ranked_papers=ranked,
        related_work=related,
        selected_papers=selected,
        ideas=ideas,
        prototypes=prototypes,
        backtests=backtests,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert len(discovered) == 10
    assert len(selected) == 2
    assert len(ideas) == 2
    assert len(backtests) == 2
    assert report.report_path is not None


def test_summarize_topics_merges_directly_relevant_papers(tmp_path):
    settings = AppSettings.model_validate(
        {
            "prompts": {},
            "pipeline": {
                "date_window_days": 30,
                "max_papers": 10,
                "top_k": 2,
                "relevance_threshold": 0.0,
            },
            "reporting": {"output_dir": str(tmp_path)},
        }
    )
    dependencies = PipelineDependencies(
        settings=settings,
        arxiv_client=StubArxivClient(),
        llm_client=MockLLMClient(allow_mock=True),
        semantic_scholar_client=SemanticScholarClient(),
        market_data_client=DummyMarketDataClient(),
        backtest_runner=SimpleBacktestRunner(),
        report_writer=MarkdownReportWriter(tmp_path),
    )
    pipeline = ResearchPipeline(dependencies)

    result = pipeline.summarize_topics(
        topics=["markets", "bonds"],
        fetch_limit=DEFAULT_TOPIC_FETCH_LIMIT,
        summary_limit=DEFAULT_TOPIC_SUMMARY_LIMIT,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert result.topics == ["markets", "bonds"]
    assert result.fetch_limit == DEFAULT_TOPIC_FETCH_LIMIT
    assert result.summary_limit == DEFAULT_TOPIC_SUMMARY_LIMIT
    assert len(result.batches) == 2
    assert all(batch.fetched_count == DEFAULT_TOPIC_FETCH_LIMIT for batch in result.batches)
    assert all(len(batch.summarized_papers) == DEFAULT_TOPIC_SUMMARY_LIMIT for batch in result.batches)
    assert len(result.directly_relevant_papers) == 2 * DEFAULT_TOPIC_SUMMARY_LIMIT
    assert all(paper.relevance_label == "directly_relevant" for paper in result.directly_relevant_papers)


class OrderedStubArxivClient(StubArxivClient):
    def search(
        self,
        query: str,
        max_results: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        papers_by_topic = {
            "markets": [
                Paper(
                    paper_id="shared-1",
                    title="shared alpha return trading",
                    abstract="return forecasting alpha long-short trading daily equities",
                    authors=["Researcher"],
                    categories=["q-fin.ST"],
                    published=datetime(2024, 1, 11, tzinfo=timezone.utc),
                    query_topic=query,
                    pdf_url="https://example.com/shared-1.pdf",
                ),
                Paper(
                    paper_id="markets-1",
                    title="markets alpha return trading",
                    abstract="return forecasting alpha long-short trading daily equities",
                    authors=["Researcher"],
                    categories=["q-fin.ST"],
                    published=datetime(2024, 1, 10, tzinfo=timezone.utc),
                    query_topic=query,
                    pdf_url="https://example.com/markets-1.pdf",
                ),
                Paper(
                    paper_id="markets-2",
                    title="markets return forecasting",
                    abstract="return forecasting daily equities",
                    authors=["Researcher"],
                    categories=["q-fin.ST"],
                    published=datetime(2024, 1, 9, tzinfo=timezone.utc),
                    query_topic=query,
                    pdf_url="https://example.com/markets-2.pdf",
                ),
                Paper(
                    paper_id="markets-3",
                    title="markets risk model",
                    abstract="volatility risk forecasting equities",
                    authors=["Researcher"],
                    categories=["q-fin.RM"],
                    published=datetime(2024, 1, 8, tzinfo=timezone.utc),
                    query_topic=query,
                    pdf_url="https://example.com/markets-3.pdf",
                ),
            ],
            "bonds": [
                Paper(
                    paper_id="shared-1",
                    title="shared alpha return trading",
                    abstract="return forecasting alpha long-short trading daily equities",
                    authors=["Researcher"],
                    categories=["q-fin.ST"],
                    published=datetime(2024, 1, 11, tzinfo=timezone.utc),
                    query_topic=query,
                    pdf_url="https://example.com/shared-1.pdf",
                ),
                Paper(
                    paper_id="bonds-1",
                    title="bonds alpha return trading",
                    abstract="return forecasting alpha long-short trading bonds",
                    authors=["Researcher"],
                    categories=["q-fin.ST"],
                    published=datetime(2024, 1, 10, tzinfo=timezone.utc),
                    query_topic=query,
                    pdf_url="https://example.com/bonds-1.pdf",
                ),
                Paper(
                    paper_id="bonds-2",
                    title="bonds return forecasting",
                    abstract="return forecasting bonds",
                    authors=["Researcher"],
                    categories=["q-fin.ST"],
                    published=datetime(2024, 1, 9, tzinfo=timezone.utc),
                    query_topic=query,
                    pdf_url="https://example.com/bonds-2.pdf",
                ),
                Paper(
                    paper_id="bonds-3",
                    title="bonds return analysis",
                    abstract="return prediction bonds",
                    authors=["Researcher"],
                    categories=["q-fin.ST"],
                    published=datetime(2024, 1, 8, tzinfo=timezone.utc),
                    query_topic=query,
                    pdf_url="https://example.com/bonds-3.pdf",
                ),
            ],
        }
        return papers_by_topic[query][:max_results]


def test_summarize_topics_filters_orders_and_summarizes_full_papers(tmp_path):
    settings = AppSettings.model_validate(
        {
            "prompts": {},
            "pipeline": {
                "date_window_days": 30,
                "max_papers": 10,
                "top_k": 2,
                "relevance_threshold": 0.0,
            },
            "reporting": {"output_dir": str(tmp_path)},
        }
    )
    dependencies = PipelineDependencies(
        settings=settings,
        arxiv_client=OrderedStubArxivClient(),
        llm_client=StubTopicSummaryLLMClient(),
        semantic_scholar_client=SemanticScholarClient(),
        market_data_client=DummyMarketDataClient(),
        backtest_runner=SimpleBacktestRunner(),
        report_writer=MarkdownReportWriter(tmp_path),
    )
    pipeline = ResearchPipeline(dependencies)

    result = pipeline.summarize_topics(
        topics=["markets", "bonds"],
        fetch_limit=3,
        summary_limit=3,
        full_paper_limit=2,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    assert [paper.paper_id for paper in result.filtered_papers] == [
        "markets-1",
        "bonds-1",
        "shared-1",
        "markets-2",
        "bonds-2",
    ]
    assert [paper.implementable_alpha_label for paper in result.filtered_papers] == ["yes", "yes", "likely", "likely", "likely"]
    assert [paper.global_rank for paper in result.filtered_papers] == [1, 2, 3, 4, 5]
    assert len([paper for paper in result.filtered_papers if paper.paper_id == "shared-1"]) == 1
    assert [paper.paper_id for paper in result.full_paper_summaries] == ["markets-1", "bonds-1"]
    assert all(paper.full_text_summary.startswith("Full summary") for paper in result.full_paper_summaries)
