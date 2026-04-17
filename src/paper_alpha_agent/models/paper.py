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


class SummarizedPaper(Paper):
    summary_rank: int | None = None
    global_rank: int | None = None
    global_relevance_score: float | None = None
    summary: str
    relevance_label: str
    implementable_alpha_label: str
    why_relevant: list[str] = Field(default_factory=list)
    model_family: str | None = None
    prediction_target: str | None = None
    forecast_horizon: str | None = None
    asset_class: str | None = None
    data_context: str | None = None
    implementation_takeaways: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    implementation_constraints: list[str] = Field(default_factory=list)


class FullPaperSummary(SummarizedPaper):
    full_text_summary: str
    alpha_thesis: str | None = None
    implementation_complexity: str | None = None
    strategy_quality: str | None = None
    sharpe_ratio: float | None = None
    sharpe_ratio_context: str | None = None
    evidence: list[str] = Field(default_factory=list)
    implementation_requirements: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    full_text_char_count: int = 0


class TopicSummaryBatch(BaseModel):
    topic: str
    fetched_count: int
    summarized_papers: list[SummarizedPaper] = Field(default_factory=list)
    directly_relevant_papers: list[SummarizedPaper] = Field(default_factory=list)


class TopicSummaryStageResult(BaseModel):
    topics: list[str] = Field(default_factory=list)
    fetch_limit: int
    summary_limit: int
    full_paper_limit: int
    batches: list[TopicSummaryBatch] = Field(default_factory=list)
    directly_relevant_papers: list[SummarizedPaper] = Field(default_factory=list)
    filtered_papers: list[SummarizedPaper] = Field(default_factory=list)
    full_paper_summaries: list[FullPaperSummary] = Field(default_factory=list)


ALPHA_PRIORITY = {"yes": 0, "likely": 1, "unlikely": 2, "no": 3}


class RelatedWorkItem(BaseModel):
    title: str
    source: str = "semantic_scholar"
    url: str | None = None
    overlap_summary: str
    similarity_score: float = Field(ge=0.0, le=1.0)
