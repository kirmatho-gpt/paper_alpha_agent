from __future__ import annotations

from datetime import datetime

from paper_alpha_agent.models.report import ResearchReport
from paper_alpha_agent.tools.report_writer import MarkdownReportWriter


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
