from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: datetime
    updated: datetime | None = None
    pdf_url: HttpUrl | None = None
    entry_url: HttpUrl | None = None
    source: str = "arxiv"
    query_topic: str | None = None


class RankedPaper(Paper):
    relevance_score: float = Field(ge=0.0, le=1.0)
    implementability_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    summary: str
    horizon: str | None = None
    frequency: str | None = None
    asset_classes: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)

    @property
    def composite_score(self) -> float:
        return round(
            0.5 * self.relevance_score
            + 0.3 * self.implementability_score
            + 0.2 * self.novelty_score,
            4,
        )


class RelatedWorkItem(BaseModel):
    title: str
    source: str = "semantic_scholar"
    url: str | None = None
    overlap_summary: str
    similarity_score: float = Field(ge=0.0, le=1.0)
