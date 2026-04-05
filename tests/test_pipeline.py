from __future__ import annotations

from datetime import datetime, timezone

from paper_alpha_agent.config import AppSettings
from paper_alpha_agent.llm.client import MockLLMClient
from paper_alpha_agent.models.paper import Paper
from paper_alpha_agent.orchestration.pipeline import PipelineDependencies, ResearchPipeline
from paper_alpha_agent.tools.backtest_runner import SimpleBacktestRunner
from paper_alpha_agent.tools.market_data_client import DummyMarketDataClient
from paper_alpha_agent.tools.report_writer import MarkdownReportWriter
from paper_alpha_agent.tools.semantic_scholar_client import SemanticScholarClient


class StubArxivClient:
    def search(self, query: str, max_results: int = 10):
        return [
            Paper(
                paper_id=f"{query}-1",
                title=f"{query} paper",
                abstract="Daily financial return forecasting with cross-sectional signals.",
                authors=["Researcher"],
                categories=["q-fin.ST"],
                published=datetime(2024, 1, 10, tzinfo=timezone.utc),
                query_topic=query,
            )
        ]

    @staticmethod
    def deduplicate(papers):
        unique = {paper.paper_id: paper for paper in papers}
        return list(unique.values())


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
        llm_client=MockLLMClient(),
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

    assert len(discovered) == 2
    assert len(selected) == 2
    assert len(ideas) == 2
    assert len(backtests) == 2
    assert report.report_path is not None
