from __future__ import annotations

from datetime import datetime, timezone

from paper_alpha_agent.models.paper import FullPaperSummary, SummarizedPaper, TopicSummaryStageResult
from paper_alpha_agent.models.report import ResearchReport
from paper_alpha_agent.tools.report_writer import MarkdownReportWriter
from paper_alpha_agent.tools.topic_summary_email_writer import TopicSummaryEmailHtmlWriter


def test_markdown_report_writer_creates_file(tmp_path):
    writer = MarkdownReportWriter(tmp_path)
    report = ResearchReport(
        created_at=datetime(2024, 1, 1),
        universe="test",
        start_date="2024-01-01",
        end_date="2024-01-31",
        discovered_count=1,
        findings=["A finding"],
        caveats=["A caveat"],
        next_steps=["A next step"],
    )

    path = writer.write(report)

    assert path.exists()
    contents = path.read_text(encoding="utf-8")
    assert "# Research Report" in contents
    assert "A finding" in contents


def test_topic_summary_email_html_writer_creates_mobile_friendly_html(tmp_path):
    filtered = SummarizedPaper.model_validate(
        {
            "paper_id": "2604.13260v1",
            "title": "Which Voices Move Markets?",
            "abstract": "Post-earnings return research.",
            "authors": ["Researcher"],
            "categories": ["q-fin.ST"],
            "published": datetime(2026, 4, 12, tzinfo=timezone.utc),
            "entry_url": "http://arxiv.org/abs/2604.13260v1",
            "query_topic": "stocks",
            "summary_rank": 2,
            "global_rank": 1,
            "summary": "Paper-level summary",
            "relevance_label": "directly_relevant",
            "implementable_alpha_label": "yes",
        }
    )
    full_summary = FullPaperSummary.model_validate(
        {
            **filtered.model_dump(mode="json"),
            "full_text_summary": (
                "The paper presents evidence that speaker identity can explain "
                "cross-sectional post-earnings returns."
            ),
            "alpha_thesis": "Speaker identity signal predicts post-earnings returns.",
            "implementation_complexity": "Moderate",
            "strategy_quality": "Strong",
            "sharpe_ratio": 1.3,
            "sharpe_ratio_context": "Reported in out-of-sample tests.",
            "evidence": ["Significant spread between top and bottom speaker buckets."],
            "implementation_requirements": ["Earnings call transcripts", "Speaker metadata"],
            "key_risks": ["Regime shifts in earnings communication style"],
        }
    )
    result = TopicSummaryStageResult(
        topics=["stocks"],
        fetch_limit=30,
        summary_limit=7,
        full_paper_limit=10,
        filtered_papers=[filtered],
        full_paper_summaries=[full_summary],
    )

    writer = TopicSummaryEmailHtmlWriter(tmp_path)
    output_path = writer.write(result=result, start_date="2026-04-01", end_date="2026-04-17")

    assert output_path.exists()
    assert "end_2026-04-17" in output_path.name
    assert "span_17d" in output_path.name
    html = output_path.read_text(encoding="utf-8")
    assert "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"" in html
    assert "Final Filtered Candidates" in html
    assert "Full-Paper Summaries" in html
    assert "http://arxiv.org/abs/2604.13260v1" in html
    assert "Open paper on arXiv" in html
    assert "PaperSummary #1" in html
    assert "Speaker identity signal predicts post-earnings returns." in html
