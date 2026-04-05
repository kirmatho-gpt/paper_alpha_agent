from __future__ import annotations

from pydantic import BaseModel, Field


class SemanticScholarPaper(BaseModel):
    title: str
    abstract: str | None = None
    year: int | None = None
    citation_count: int = 0
    url: str | None = None


class SemanticScholarSearchResponse(BaseModel):
    total: int = 0
    data: list[SemanticScholarPaper] = Field(default_factory=list)


class SemanticScholarClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def find_similar_papers(self, title: str, limit: int = 3) -> SemanticScholarSearchResponse:
        stubbed = [
            SemanticScholarPaper(
                title=f"Adjacent work related to {title[:50]}",
                abstract="Stubbed Semantic Scholar result used until a real API integration is added.",
                year=2024,
                citation_count=12,
                url="https://www.semanticscholar.org/",
            )
        ]
        return SemanticScholarSearchResponse(total=len(stubbed), data=stubbed[:limit])
