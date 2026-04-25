from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from paper_alpha_agent.config import get_settings
from paper_alpha_agent.logging_config import configure_logging
from paper_alpha_agent.orchestration.pipeline import (
    DEFAULT_TOPIC_FETCH_LIMIT,
    DEFAULT_TOPIC_FULL_PAPER_LIMIT,
    DEFAULT_TOPIC_SUMMARY_LIMIT,
    DEFAULT_TOPIC_SUMMARIZATION_TOPICS,
    ResearchPipeline,
    build_default_dependencies,
    run_research_pipeline,
)
from paper_alpha_agent.models.report import ResearchReport
from paper_alpha_agent.research.discovery import default_date_window
from paper_alpha_agent.research.ranking import shortlist_papers_for_ranking, rank_papers
from paper_alpha_agent.tools.report_writer import MarkdownReportWriter
from paper_alpha_agent.tools.topic_summary_email_writer import TopicSummaryEmailHtmlWriter


app = typer.Typer(help="Bounded research pipeline for finance-focused paper discovery and prototyping.")


def _stringify(value: object) -> str:
    return "None" if value is None else str(value)


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return f"{value[: width - 3]}..."


def _format_table_row(cells: list[str], widths: list[int], right_aligned_indexes: set[int] | None = None) -> str:
    right_aligned_indexes = right_aligned_indexes or set()
    formatted_cells: list[str] = []
    for index, (cell, width) in enumerate(zip(cells, widths, strict=True)):
        clipped_cell = _truncate(cell, width)
        if index in right_aligned_indexes:
            formatted_cells.append(clipped_cell.rjust(width))
        else:
            formatted_cells.append(clipped_cell.ljust(width))
    return " | ".join(formatted_cells)


def _build_filtered_candidates_table_lines(filtered_papers: list, selected_paper_ids: set[str]) -> list[str]:
    headers = ["selected", "alpha", "g_rank", "t_rank", "topic", "link", "title"]
    link_values = [str(paper.entry_url or paper.pdf_url or "") for paper in filtered_papers]
    link_width = max([len("link"), *(len(link) for link in link_values)])
    widths = [8, 6, 6, 6, 12, link_width, 38]
    right_aligned_indexes = {2, 3}

    lines = [
        _format_table_row(headers, widths, right_aligned_indexes=right_aligned_indexes),
        "-+-".join("-" * width for width in widths),
    ]

    for paper in filtered_papers:
        link = str(paper.entry_url or paper.pdf_url or "")
        selected_for_full_text = "yes" if paper.paper_id in selected_paper_ids else "no"
        lines.append(
            _format_table_row(
                [
                    selected_for_full_text,
                    _stringify(paper.implementable_alpha_label),
                    _stringify(paper.global_rank),
                    _stringify(paper.summary_rank),
                    _stringify(paper.query_topic),
                    link,
                    _stringify(paper.title),
                ],
                widths,
                right_aligned_indexes=right_aligned_indexes,
            )
        )

    return lines


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
def doctor() -> None:
    settings = get_settings()
    dependencies = build_default_dependencies(settings)

    openai_key_present = bool(settings.api_keys.openai)
    semantic_scholar_key_present = bool(settings.api_keys.semantic_scholar)
    llm_client_name = type(dependencies.llm_client).__name__

    typer.echo("paper_alpha_agent environment check")
    typer.echo(f"OpenAI API key present: {'yes' if openai_key_present else 'no'}")
    typer.echo(f"Semantic Scholar API key present: {'yes' if semantic_scholar_key_present else 'no'}")
    typer.echo(f"Configured model: {settings.llm.model_name}")
    typer.echo(f"Resolved LLM client: {llm_client_name}")
    typer.echo(f"Report output dir: {settings.report_output_path}")

    if openai_key_present:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.api_keys.openai)
            model = client.models.retrieve(settings.llm.model_name)
            typer.echo(f"OpenAI API ping: ok (model='{model.id}')")
        except Exception as exc:
            typer.echo(f"OpenAI API ping: failed ({type(exc).__name__}: {exc})", err=True)
    else:
        typer.echo("OpenAI API ping: skipped (no key configured)")


@app.command()
def summarize(
    topic: str = typer.Option(..., "--topic", help="Topic query to send to arXiv."),
    limit: int = typer.Option(5, "--limit", min=1, help="Number of shortlisted papers to summarize."),
    fetch_limit: int | None = typer.Option(
        None,
        "--fetch-limit",
        min=1,
        help="Number of papers to fetch from arXiv before heuristic filtering. Defaults to max(limit * 6, 15).",
    ),
    shortlist_limit: int | None = typer.Option(None, "--shortlist-limit", min=1, help="Number of papers to keep after heuristic shortlisting before final summary selection."),
    start_date: str | None = typer.Option(None, help="Inclusive ISO start date."),
    end_date: str | None = typer.Option(None, help="Inclusive ISO end date."),
    json_output: bool = typer.Option(False, "--json", help="Print structured summary JSON."),
    show_heuristic_logs: bool = typer.Option(
        False,
        "--show-heuristic-logs",
        help="Suppress prefilter and shortlist heuristic logs. Logging is suppressed by default.",
    ),
) -> None:
    settings = get_settings()
    dependencies = build_default_dependencies(settings)

    if not start_date or not end_date:
        start_date, end_date = default_date_window(settings.pipeline.date_window_days)

    resolved_fetch_limit = fetch_limit or max(limit * 6, 15)
    resolved_shortlist_limit = shortlist_limit or max(limit * 3, 10)

    fetched_papers = dependencies.arxiv_client.search(
        query=topic,
        max_results=resolved_fetch_limit,
        start_date=start_date,
        end_date=end_date,
    )
    shortlisted_papers = shortlist_papers_for_ranking(
        fetched_papers,
        shortlist_size=min(resolved_shortlist_limit, len(fetched_papers)),
        log_decisions=show_heuristic_logs,
    )
    selected_papers = shortlisted_papers[:limit]

    typer.echo(f"topic={topic}")
    typer.echo(f"date_window={start_date}..{end_date}")
    typer.echo(f"llm_client={type(dependencies.llm_client).__name__}")
    typer.echo(f"fetched_papers={len(fetched_papers)}")
    typer.echo(f"shortlisted_papers={len(shortlisted_papers)}")
    typer.echo(f"selected_papers={len(selected_papers)}")

    if not selected_papers:
        typer.echo("No papers survived heuristic filtering for this topic/date window.", err=True)
        raise typer.Exit(code=1)

    for index, paper in enumerate(selected_papers, start=1):
        summary = dependencies.llm_client.summarize_paper(paper)
        typer.echo(f"\n--- Paper {index} ---")
        typer.echo(f"title: {paper.title}")
        typer.echo(f"paper_id: {paper.paper_id}")
        typer.echo(f"published: {paper.published.date()}")
        typer.echo(f"categories: {', '.join(paper.categories)}")
        if json_output:
            typer.echo(summary.model_dump_json(indent=2))
        else:
            typer.echo(f"summary: {summary.summary}")
            typer.echo(f"relevance_label: {summary.relevance_label}")
            typer.echo(f"implementable_alpha_label: {summary.implementable_alpha_label}")
            typer.echo(f"why_relevant: {', '.join(summary.why_relevant)}")
            typer.echo(f"model_family: {summary.model_family}")
            typer.echo(f"prediction_target: {summary.prediction_target}")
            typer.echo(f"forecast_horizon: {summary.forecast_horizon}")
            typer.echo(f"asset_class: {summary.asset_class}")
            typer.echo(f"data_context: {summary.data_context}")
            typer.echo(f"implementation_takeaways: {', '.join(summary.implementation_takeaways)}")
            typer.echo(f"missing_information: {', '.join(summary.missing_information)}")
            typer.echo(f"caveats: {', '.join(summary.caveats)}")


@app.command()
def summarize_topics(
    json_output: bool = typer.Option(False, "--json", help="Print structured stage JSON."),
    full_paper_limit: int = typer.Option(
        DEFAULT_TOPIC_FULL_PAPER_LIMIT,
        "--full-paper-limit",
        min=1,
        help="Number of filtered papers to download and summarize from full text.",
    ),
    show_heuristic_logs: bool = typer.Option(
        False,
        "--show-heuristic-logs",
        help="Enable prefilter and shortlist heuristic logs during topic summarization.",
    ),
    start_date: str | None = typer.Option(None, help="Inclusive ISO start date."),
    end_date: str | None = typer.Option(None, help="Inclusive ISO end date."),
) -> None:
    settings = get_settings()
    dependencies = build_default_dependencies(settings)
    result = ResearchPipeline(dependencies).summarize_topics(
        topics=DEFAULT_TOPIC_SUMMARIZATION_TOPICS,
        fetch_limit=DEFAULT_TOPIC_FETCH_LIMIT,
        summary_limit=DEFAULT_TOPIC_SUMMARY_LIMIT,
        full_paper_limit=full_paper_limit,
        log_heuristic_decisions=show_heuristic_logs,
        start_date=start_date,
        end_date=end_date,
    )

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(f"topics={result.topics}")
    typer.echo(f"fetch_limit={result.fetch_limit}")
    typer.echo(f"summary_limit={result.summary_limit}")
    typer.echo(f"full_paper_limit={result.full_paper_limit}")
    typer.echo(f"merged_directly_relevant={len(result.directly_relevant_papers)}")
    typer.echo(f"filtered_candidates={len(result.filtered_papers)}")
    typer.echo(f"full_paper_summaries={len(result.full_paper_summaries)}")

    for batch in result.batches:
        typer.echo(
            f"\n[{batch.topic}] fetched={batch.fetched_count} "
            f"summarized={len(batch.summarized_papers)} "
            f"directly_relevant={len(batch.directly_relevant_papers)}"
        )

    if not result.filtered_papers:
        typer.echo("\nNo directly relevant papers with implementable alpha in {yes, likely}.", err=True)
        raise typer.Exit(code=1)

    typer.echo("\nFinal filtered candidates before full-paper truncation:")
    selected_paper_ids = {paper.paper_id for paper in result.full_paper_summaries}
    for line in _build_filtered_candidates_table_lines(result.filtered_papers, selected_paper_ids):
        typer.echo(line)

    for index, paper in enumerate(result.full_paper_summaries, start=1):
        typer.echo(
            f"\n[PaperSummary #{index}] "
            f"global_rank={_stringify(paper.global_rank)} "
            f"topic_rank={_stringify(paper.summary_rank)} "
            f"topic={_stringify(paper.query_topic)} "
            f"alpha={_stringify(paper.implementable_alpha_label)}"
        )
        typer.echo(f"Title      : {_stringify(paper.title)}")
        typer.echo(f"Complexity : {_stringify(paper.implementation_complexity)}")
        typer.echo(f"Quality    : {_stringify(paper.strategy_quality)}")
        typer.echo(f"Sharpe     : {_stringify(paper.sharpe_ratio)}")
        typer.echo(f"SharpeNote : {_stringify(paper.sharpe_ratio_context)}")
        typer.echo("Summary:")
        typer.echo(_stringify(paper.full_text_summary))


@app.command()
def rank(
    topic: str = typer.Option(..., "--topic", help="Topic query to send to arXiv."),
    limit: int = typer.Option(5, "--limit", min=1, help="Number of ranked papers to print."),
    fetch_limit: int | None = typer.Option(
        None,
        "--fetch-limit",
        min=1,
        help="Number of papers to fetch from arXiv before heuristic filtering. Defaults to max(limit * 6, 15).",
    ),
    shortlist_limit: int | None = typer.Option(
        None,
        "--shortlist-limit",
        min=1,
        help="Number of papers to send to the LLM ranking stage before selecting final ranked output.",
    ),
    start_date: str | None = typer.Option(None, help="Inclusive ISO start date."),
    end_date: str | None = typer.Option(None, help="Inclusive ISO end date."),
) -> None:
    settings = get_settings()
    dependencies = build_default_dependencies(settings)

    if not start_date or not end_date:
        start_date, end_date = default_date_window(settings.pipeline.date_window_days)

    resolved_fetch_limit = fetch_limit or max(limit * 6, 15)
    resolved_shortlist_limit = shortlist_limit or max(limit * 3, 10)

    fetched_papers = dependencies.arxiv_client.search(
        query=topic,
        max_results=resolved_fetch_limit,
        start_date=start_date,
        end_date=end_date,
    )
    shortlisted_papers = shortlist_papers_for_ranking(
        fetched_papers,
        shortlist_size=min(resolved_shortlist_limit, len(fetched_papers)),
    )
    ranked_papers = rank_papers(
        shortlisted_papers,
        llm_client=dependencies.llm_client,
        relevance_threshold=settings.pipeline.relevance_threshold,
    )
    selected_papers = ranked_papers[:limit]

    typer.echo(f"topic={topic}")
    typer.echo(f"date_window={start_date}..{end_date}")
    typer.echo(f"llm_client={type(dependencies.llm_client).__name__}")
    typer.echo(f"fetched_papers={len(fetched_papers)}")
    typer.echo(f"shortlisted_papers={len(shortlisted_papers)}")
    typer.echo(f"ranked_papers={len(ranked_papers)}")
    typer.echo(f"selected_papers={len(selected_papers)}")
    typer.echo(
        "date | paper_id | title | categories | relevance | implementability | novelty | composite"
    )

    if not selected_papers:
        typer.echo("No papers survived heuristic filtering and LLM ranking for this topic/date window.", err=True)
        raise typer.Exit(code=1)

    for paper in selected_papers:
        typer.echo(
            f"{paper.published.date()} | "
            f"{paper.paper_id} | "
            f"{paper.title} | "
            f"{', '.join(paper.categories)} | "
            f"{paper.relevance_score:.2f} | "
            f"{paper.implementability_score:.2f} | "
            f"{paper.novelty_score:.2f} | "
            f"{paper.composite_score:.2f}"
        )


@app.command()
def summarize_topics_email(
    full_paper_limit: int = typer.Option(
        DEFAULT_TOPIC_FULL_PAPER_LIMIT,
        "--full-paper-limit",
        min=1,
        help="Number of filtered papers to download and summarize from full text.",
    ),
    show_heuristic_logs: bool = typer.Option(
        False,
        "--show-heuristic-logs",
        help="Enable prefilter and shortlist heuristic logs during topic summarization.",
    ),
    start_date: str | None = typer.Option(None, help="Inclusive ISO start date."),
    end_date: str | None = typer.Option(None, help="Inclusive ISO end date."),
    output_file: str | None = typer.Option(
        None,
        "--output-file",
        help="Optional output path for the generated HTML file.",
    ),
) -> None:
    settings = get_settings()
    dependencies = build_default_dependencies(settings)

    if not start_date or not end_date:
        start_date, end_date = default_date_window(settings.pipeline.date_window_days)

    result = ResearchPipeline(dependencies).summarize_topics(
        topics=DEFAULT_TOPIC_SUMMARIZATION_TOPICS,
        fetch_limit=DEFAULT_TOPIC_FETCH_LIMIT,
        summary_limit=DEFAULT_TOPIC_SUMMARY_LIMIT,
        full_paper_limit=full_paper_limit,
        log_heuristic_decisions=show_heuristic_logs,
        start_date=start_date,
        end_date=end_date,
    )
    writer = TopicSummaryEmailHtmlWriter(settings.report_output_path)
    output_path = Path(output_file).expanduser() if output_file else None
    path = writer.write(
        result=result,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
    )

    typer.echo(f"HTML topic summary email written to {path}")
    typer.echo(f"filtered_candidates={len(result.filtered_papers)}")
    typer.echo(f"full_paper_summaries={len(result.full_paper_summaries)}")


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
