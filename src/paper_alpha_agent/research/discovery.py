"""Paper discovery stage built on top of the arXiv client.

This module is intentionally split into:

- pure discovery helpers that are easy to unit test
- a small module entry point for manual execution

Run it directly as a module:

```bash
python -m paper_alpha_agent.research.discovery
python -m paper_alpha_agent.research.discovery --topic commodities --topic "relative value"
python -m paper_alpha_agent.research.discovery --start-date 2024-01-01 --end-date 2024-12-31 --max-papers 15
```

By default the module loads topic queries from `config/topics.yaml`. You can
override them by passing one or more `--topic` arguments.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Sequence

from paper_alpha_agent.config import AppSettings, get_settings
from paper_alpha_agent.logging_config import configure_logging
from paper_alpha_agent.models.paper import Paper
from paper_alpha_agent.tools.arxiv_client import ArxivClient


LOGGER = logging.getLogger(__name__)


def discover_papers(
    arxiv_client: ArxivClient,
    topics: Sequence[str],
    max_papers: int,
    start_date: str | None = None,
    end_date: str | None = None,
    default_window_days: int | None = None,
) -> list[Paper]:
    """Run the bounded discovery stage across multiple topics.

    Responsibilities of this function:

    - run multiple topic queries
    - merge all results into one list
    - deduplicate papers by `paper_id`
    - optionally filter the merged set by publication date
    - cap the final output at `max_papers`

    This function is the pure stage-level API and is the one to call from tests
    or orchestration code:

    ```python
    from paper_alpha_agent.research.discovery import discover_papers
    from paper_alpha_agent.tools.arxiv_client import ArxivClient

    client = ArxivClient(request_pause_seconds=4.0)
    papers = discover_papers(
        arxiv_client=client,
        topics=["commodities", "relative value"],
        max_papers=10,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    print(len(papers))
    ```

    Args:
        arxiv_client: Preconfigured `ArxivClient` instance.
        topics: Iterable of topic strings. Each topic is queried independently.
        max_papers: Maximum number of papers returned after merge, deduplication,
            date filtering, and sorting.
        start_date: Optional inclusive ISO date filter, such as `2024-01-01`.
        end_date: Optional inclusive ISO date filter, such as `2024-12-31`.
        default_window_days: Optional fallback lookback window. If either date
            bound is omitted and this value is provided, the function resolves a
            default `(start_date, end_date)` window before filtering.

    Returns:
        A list of deduplicated `Paper` models sorted by `published` descending.
    """
    if (not start_date or not end_date) and default_window_days is not None:
        resolved_start, resolved_end = default_date_window(default_window_days)
        start_date = start_date or resolved_start
        end_date = end_date or resolved_end

    discovered: list[Paper] = []
    for topic in topics:
        discovered.extend(
            arxiv_client.search(
                topic,
                max_results=max_papers,
                start_date=start_date,
                end_date=end_date,
            )
        )

    deduped = ArxivClient.deduplicate(discovered)
    filtered = filter_papers_by_date(deduped, start_date=start_date, end_date=end_date)
    filtered.sort(key=lambda paper: paper.published, reverse=True)
    capped = filtered[:max_papers]

    LOGGER.info(
        "Discovery finished: %s raw, %s deduplicated, %s returned",
        len(discovered),
        len(deduped),
        len(capped),
    )
    return capped


def default_date_window(days: int) -> tuple[str, str]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return str(start), str(end)


def filter_papers_by_date(
    papers: Sequence[Paper],
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Paper]:
    """Filter papers by inclusive publication date bounds.

    This helper is deterministic and easy to test in isolation:

    ```python
    filtered = filter_papers_by_date(
        papers,
        start_date="2024-01-01",
        end_date="2024-12-31",
    )
    ```

    Args:
        papers: Input papers.
        start_date: Optional inclusive lower bound in ISO format.
        end_date: Optional inclusive upper bound in ISO format.

    Returns:
        Filtered papers preserving original relative order.
    """
    if not start_date and not end_date:
        return list(papers)

    start = _parse_iso_date(start_date) if start_date else None
    end = _parse_iso_date(end_date) if end_date else None
    kept: list[Paper] = []

    for paper in papers:
        published = paper.published.date()
        if start and published < start:
            continue
        if end and published > end:
            continue
        kept.append(paper)
    return kept


def resolve_topics(settings: AppSettings, cli_topics: Sequence[str] | None = None) -> list[str]:
    if cli_topics:
        return [topic for topic in cli_topics if topic.strip()]
    return settings.arxiv_query_topics


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the module CLI parser used by `python -m ...`.

    Returns:
        Configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description="Run the paper discovery stage against arXiv.")
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="Topic to query. Pass multiple times to run multiple queries. Defaults to config/topics.yaml.",
    )
    parser.add_argument("--start-date", help="Inclusive ISO start date, e.g. 2024-01-01.")
    parser.add_argument("--end-date", help="Inclusive ISO end date, e.g. 2024-12-31.")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to return.")
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=4.0,
        help="Optional delay inserted before each arXiv request. Defaults to 4.0 seconds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the discovery stage as a standalone module.

    Example:

    ```bash
    python -m paper_alpha_agent.research.discovery --topic commodities --max-papers 5
    ```

    Args:
        argv: Optional explicit argument sequence for testing.

    Returns:
        Process-style integer exit code.
    """
    configure_logging()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    topics = resolve_topics(settings, args.topics)
    if not topics:
        parser.error("No topics available. Add them in config/topics.yaml or pass --topic.")

    max_papers = args.max_papers or settings.pipeline.max_papers
    start_date = args.start_date
    end_date = args.end_date
    if not start_date and not end_date:
        start_date, end_date = default_date_window(settings.pipeline.date_window_days)

    client = ArxivClient(request_pause_seconds=args.request_pause_seconds)
    papers = discover_papers(
        arxiv_client=client,
        topics=topics,
        max_papers=max_papers,
        start_date=start_date,
        end_date=end_date,
        default_window_days=settings.pipeline.date_window_days,
    )

    print(f"topics={topics}")
    print(f"date_window={start_date}..{end_date}")
    print(f"returned={len(papers)}")
    for index, paper in enumerate(papers, start=1):
        print(f"\n--- Paper {index} ---")
        print(f"paper_id:    {paper.paper_id}")
        print(f"title:       {paper.title}")
        print(f"published:   {paper.published.isoformat()}")
        print(f"authors:     {', '.join(paper.authors)}")
        print(f"categories:  {', '.join(paper.categories)}")
        print(f"pdf_url:     {paper.pdf_url}")
        print(f"entry_url:   {paper.entry_url}")
        print(f"query_topic: {paper.query_topic}")
        print(f"abstract:    {paper.abstract}")
    return 0


def _parse_iso_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


if __name__ == "__main__":
    raise SystemExit(main())
