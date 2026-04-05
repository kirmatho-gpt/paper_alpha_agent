from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx

from paper_alpha_agent.models.paper import Paper
from paper_alpha_agent.tools.storage import read_json, write_json


LOGGER = logging.getLogger(__name__)
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "arxiv"


class ArxivClient:
    def __init__(
        self,
        http_client: httpx.Client | None = None,
        base_url: str = ARXIV_API_URL,
        request_pause_seconds: float = 4.0,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ) -> None:
        """Create a minimal arXiv export API client.

        This class can be used directly from a Python shell without any other
        project components:

        ```python
        from paper_alpha_agent.tools.arxiv_client import ArxivClient

        client = ArxivClient(request_pause_seconds=4.0)
        papers = client.search("deep learning financial forecasting", max_results=5)
        print(papers[0].title)
        ```

        Args:
            http_client: Optional preconfigured `httpx.Client`. Pass one when you
                want custom headers, retries, or test doubles.
            base_url: arXiv export API endpoint. The default is the public export
                API URL.
            request_pause_seconds: Optional delay inserted before each request.
                This is useful when batching multiple searches and wanting to be
                polite with arXiv's infrastructure. Defaults to `4.0`.
            cache_dir: Directory for on-disk query caching. Defaults to
                `data/cache/arxiv` under the repository root. Repeated calls
                with the same query and date bounds are served locally from
                here.
        """
        self.http_client = http_client or httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            follow_redirects=True,
            headers={
                "User-Agent": "paper_alpha_agent/0.1 (research pipeline)"
            },
        )
        self.base_url = base_url
        self.request_pause_seconds = request_pause_seconds
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Paper]:
        """Search arXiv and return parsed `Paper` models.

        This is the main standalone entry point for the client. Example:

        ```python
        from paper_alpha_agent.tools.arxiv_client import ArxivClient

        client = ArxivClient()
        papers = client.search(
            query="transformers asset pricing",
            max_results=3,
            sort_by="submittedDate",
            sort_order="descending",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        for paper in papers:
            print(paper.paper_id, paper.title)
        ```

        Args:
            query: Raw search string inserted into the export API query.
            max_results: Maximum number of entries requested from arXiv.
            sort_by: arXiv sort key, for example `submittedDate` or `relevance`.
            sort_order: arXiv sort direction, typically `ascending` or
                `descending`.
            start_date: Optional inclusive ISO start date used to constrain the
                arXiv query itself.
            end_date: Optional inclusive ISO end date used to constrain the
                arXiv query itself.

        Returns:
            A list of parsed `Paper` objects. If a cache entry already exists for
            the same query, dates, and search parameters, the cached result is
            returned without calling arXiv again.

        Raises:
            httpx.HTTPError: If the underlying HTTP request fails.
            xml.etree.ElementTree.ParseError: If arXiv returns malformed XML.
        """
        search_query = (
            f"(cat:q-fin.* AND abs:{query}) OR "
            f"((cat:cs.LG OR cat:cs.ai OR cat:stat.ML) AND (abs:finance OR abs:financial) AND abs:{query})"
        )
        if start_date and end_date:
            search_query = (
                f"({search_query}) AND submittedDate:"
                f"[{_format_arxiv_date(start_date, end_of_day=False)} TO {_format_arxiv_date(end_date, end_of_day=True)}]"
            )

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        cache_path = self._cache_path(
            query=query,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        if cache_path.exists():
            LOGGER.info("Using cached arXiv response for query '%s' from %s", query, cache_path)
            return self._read_cached_papers(cache_path)

        # arXiv asks clients to be polite with request pacing; keep this hook for callers.
        if self.request_pause_seconds > 0:
            time.sleep(self.request_pause_seconds)
        LOGGER.info("arXiv search query: %s", search_query)
        response = self.http_client.get(self.base_url, params=params)
        response.raise_for_status()
        papers = self.parse_feed(response.text, query_topic=query)
        self._write_cached_papers(
            cache_path=cache_path,
            query=query,
            search_query=search_query,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            papers=papers,
        )
        LOGGER.info("Fetched %s papers from arXiv for query '%s'", len(papers), query)
        return papers

    @staticmethod
    def parse_feed(feed_xml: str, query_topic: str | None = None) -> list[Paper]:
        """Parse a raw arXiv Atom feed string into `Paper` models.

        This is useful when testing or when you already have XML content and do
        not want to hit the network:

        ```python
        from pathlib import Path
        from paper_alpha_agent.tools.arxiv_client import ArxivClient

        feed_xml = Path("sample_arxiv_feed.xml").read_text()
        papers = ArxivClient.parse_feed(feed_xml, query_topic="financial forecasting")
        print(len(papers))
        ```

        Args:
            feed_xml: Raw Atom XML returned by the arXiv export API.
            query_topic: Optional topic label copied into each parsed `Paper`.

        Returns:
            A list of parsed `Paper` objects.
        """
        root = ET.fromstring(feed_xml)
        return [ArxivClient._parse_entry(entry, query_topic) for entry in root.findall("atom:entry", ATOM_NS)]

    @staticmethod
    def _parse_entry(entry: ET.Element, query_topic: str | None) -> Paper:
        paper_id = ArxivClient._safe_text(entry, "atom:id").rsplit("/", maxsplit=1)[-1]
        pdf_url = None
        entry_url = ArxivClient._safe_text(entry, "atom:id")
        for link in entry.findall("atom:link", ATOM_NS):
            href = link.attrib.get("href")
            title = link.attrib.get("title", "")
            if title == "pdf" or (href and href.endswith(".pdf")):
                pdf_url = href
        authors = [node.text or "" for node in entry.findall("atom:author/atom:name", ATOM_NS)]
        categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", ATOM_NS)]
        return Paper(
            paper_id=paper_id,
            title=ArxivClient._normalize_whitespace(ArxivClient._safe_text(entry, "atom:title")),
            abstract=ArxivClient._normalize_whitespace(ArxivClient._safe_text(entry, "atom:summary")),
            authors=authors,
            categories=[item for item in categories if item],
            published=datetime.fromisoformat(ArxivClient._safe_text(entry, "atom:published").replace("Z", "+00:00")),
            updated=datetime.fromisoformat(ArxivClient._safe_text(entry, "atom:updated").replace("Z", "+00:00")),
            pdf_url=pdf_url,
            entry_url=entry_url,
            query_topic=query_topic,
        )

    @staticmethod
    def deduplicate(papers: Iterable[Paper]) -> list[Paper]:
        unique: dict[str, Paper] = {}
        for paper in papers:
            unique.setdefault(paper.paper_id, paper)
        return list(unique.values())

    @staticmethod
    def _safe_text(entry: ET.Element, tag: str) -> str:
        node = entry.find(tag, ATOM_NS)
        return (node.text or "").strip() if node is not None else ""

    @staticmethod
    def _normalize_whitespace(value: str) -> str:
        return " ".join(value.split())

    def _cache_path(
        self,
        query: str,
        start_date: str | None,
        end_date: str | None,
        max_results: int,
        sort_by: str,
        sort_order: str,
    ) -> Path:
        cache_key = {
            "query": query,
            "start_date": start_date,
            "end_date": end_date,
            "max_results": max_results,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "base_url": self.base_url,
        }
        digest = hashlib.sha256(repr(sorted(cache_key.items())).encode("utf-8")).hexdigest()[:16]
        safe_topic = _slugify(query)
        return self.cache_dir / f"{safe_topic}_{digest}.json"

    @staticmethod
    def _read_cached_papers(cache_path: Path) -> list[Paper]:
        payload = read_json(cache_path)
        return [Paper.model_validate(item) for item in payload.get("papers", [])]

    @staticmethod
    def _write_cached_papers(
        cache_path: Path,
        query: str,
        search_query: str,
        start_date: str | None,
        end_date: str | None,
        max_results: int,
        sort_by: str,
        sort_order: str,
        papers: list[Paper],
    ) -> None:
        write_json(
            cache_path,
            {
                "query": query,
                "search_query": search_query,
                "start_date": start_date,
                "end_date": end_date,
                "max_results": max_results,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "cached_at": datetime.utcnow().isoformat(),
                "papers": [paper.model_dump(mode="json") for paper in papers],
            },
        )


def _format_arxiv_date(value: str, end_of_day: bool = False) -> str:
    """Convert `YYYY-MM-DD` into the compact datetime syntax used by arXiv."""
    suffix = "2359" if end_of_day else "0000"
    return f"{datetime.fromisoformat(value).strftime('%Y%m%d')}{suffix}"


def _slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    compact = "_".join(part for part in normalized.split("_") if part)
    return compact[:60] or "query"
