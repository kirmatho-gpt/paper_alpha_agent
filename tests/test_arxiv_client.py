from __future__ import annotations

import os

import pytest

from paper_alpha_agent.tools.arxiv_client import ArxivClient


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-03T00:00:00Z</updated>
    <published>2024-01-02T00:00:00Z</published>
    <title>Transformer Signals for Asset Return Forecasting</title>
    <summary>We study daily equity return forecasting with deep learning.</summary>
    <author><name>Alice Quant</name></author>
    <author><name>Bob Research</name></author>
    <link href="http://arxiv.org/abs/2401.00001v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1.pdf" rel="related" type="application/pdf"/>
    <category term="q-fin.ST"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""


def test_parse_feed_extracts_paper_fields():
    papers = ArxivClient.parse_feed(SAMPLE_FEED, query_topic="finance ml")

    assert len(papers) == 1
    paper = papers[0]
    assert paper.paper_id == "2401.00001v1"
    assert paper.title == "Transformer Signals for Asset Return Forecasting"
    assert "daily equity return forecasting" in paper.abstract
    assert paper.authors == ["Alice Quant", "Bob Research"]
    assert "q-fin.ST" in paper.categories
    assert str(paper.pdf_url).endswith(".pdf")


@pytest.mark.skipif(
    os.getenv("RUN_ARXIV_LIVE_TEST") != "1",
    reason="Set RUN_ARXIV_LIVE_TEST=1 to run the live arXiv integration test.",
)
def test_live_arxiv_query_prints_results():
    client = ArxivClient(request_pause_seconds=4.0)
    papers = client.search("commodities", max_results=5)

    assert len(papers) > 0
    for index, paper in enumerate(papers, start=1):
        print(f"\n--- Paper {index} ---")
        print(f"paper_id:    {paper.paper_id}")
        print(f"title:       {paper.title}")
        print(f"published:   {paper.published.isoformat()}")
        print(f"updated:     {paper.updated.isoformat() if paper.updated else None}")
        print(f"authors:     {', '.join(paper.authors)}")
        print(f"categories:  {', '.join(paper.categories)}")
        print(f"pdf_url:     {paper.pdf_url}")
        print(f"entry_url:   {paper.entry_url}")
        print(f"source:      {paper.source}")
        print(f"query_topic: {paper.query_topic}")
        print(f"abstract:    {paper.abstract}")
