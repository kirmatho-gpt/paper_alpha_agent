from __future__ import annotations

from pathlib import Path

from paper_alpha_agent.models.report import ResearchReport


class MarkdownReportWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, report: ResearchReport) -> Path:
        path = self.output_dir / f"research_report_{report.created_at.strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text(self._render(report), encoding="utf-8")
        return path

    def _render(self, report: ResearchReport) -> str:
        paper_lines = "\n".join(
            f"- {paper.title} (`{paper.paper_id}`) score={paper.composite_score:.2f}"
            for paper in report.ranked_papers
        )
        idea_lines = "\n".join(f"- {idea.title}: {idea.hypothesis}" for idea in report.ideas)
        backtest_lines = "\n".join(
            f"- {result.prototype_id}: total_return={result.total_return:.2%}, sharpe={result.sharpe_ratio:.2f}"
            for result in report.backtests
        )
        findings = "\n".join(f"- {item}" for item in report.findings)
        caveats = "\n".join(f"- {item}" for item in report.caveats)
        next_steps = "\n".join(f"- {item}" for item in report.next_steps)

        return f"""# Research Report

- Created at: {report.created_at.isoformat()}
- Universe: {report.universe}
- Discovery window: {report.start_date} to {report.end_date}
- Discovered papers: {report.discovered_count}

## Ranked Papers
{paper_lines or "- None"}

## Extracted Ideas
{idea_lines or "- None"}

## Backtest Results
{backtest_lines or "- None"}

## Findings
{findings or "- None"}

## Caveats
{caveats or "- None"}

## Next Steps
{next_steps or "- None"}
"""
