from __future__ import annotations

from datetime import datetime

import typer

from paper_alpha_agent.config import get_settings
from paper_alpha_agent.logging_config import configure_logging
from paper_alpha_agent.orchestration.pipeline import (
    ResearchPipeline,
    build_default_dependencies,
    run_research_pipeline,
)
from paper_alpha_agent.models.report import ResearchReport
from paper_alpha_agent.tools.report_writer import MarkdownReportWriter


app = typer.Typer(help="Bounded research pipeline for finance-focused paper discovery and prototyping.")


@app.callback()
def main() -> None:
    configure_logging()


@app.command()
def run(
    start_date: str | None = typer.Option(None, help="ISO start date"),
    end_date: str | None = typer.Option(None, help="ISO end date"),
) -> None:
    report = run_research_pipeline(start_date=start_date, end_date=end_date)
    typer.echo(f"Report written to {report.report_path}")
    typer.echo(f"Discovered {report.discovered_count} papers, selected {len(report.selected_paper_ids)}")


@app.command()
def discover(
    start_date: str | None = typer.Option(None, help="ISO start date"),
    end_date: str | None = typer.Option(None, help="ISO end date"),
) -> None:
    settings = get_settings()
    dependencies = build_default_dependencies(settings)
    papers = ResearchPipeline(dependencies).discover(
        start_date=start_date,
        end_date=end_date,
    )
    typer.echo("date | paper_id | title | categories")
    for paper in papers:
        typer.echo(
            f"{paper.published.date()} | {paper.paper_id} | {paper.title} | {', '.join(paper.categories)}"
        )


@app.command()
def report() -> None:
    settings = get_settings()
    writer = MarkdownReportWriter(settings.report_output_path)
    sample = ResearchReport(
        created_at=datetime.utcnow(),
        universe="sample offline report",
        start_date="2024-01-01",
        end_date="2024-12-31",
        discovered_count=0,
        findings=["This is an offline sample report for local smoke testing."],
        caveats=["No external APIs were called."],
        next_steps=["Run `paper-alpha-agent run` to execute the bounded pipeline."],
    )
    path = writer.write(sample)
    typer.echo(f"Sample report written to {path}")
