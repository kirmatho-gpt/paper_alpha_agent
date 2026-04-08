from __future__ import annotations

from pydantic import BaseModel, Field

from paper_alpha_agent.models.idea import PrototypeSpec, ResearchIdea
from paper_alpha_agent.models.paper import RelatedWorkItem


class PaperSummaryResponse(BaseModel):
    summary: str
    relevance_label: str
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


class PaperRankingResponse(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    implementability_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    summary: str
    horizon: str | None = None
    frequency: str | None = None
    asset_classes: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class PriorArtAssessmentResponse(BaseModel):
    items: list[RelatedWorkItem] = Field(default_factory=list)


class IdeaExtractionResponse(BaseModel):
    idea: ResearchIdea


class PrototypeSpecResponse(BaseModel):
    prototype: PrototypeSpec


class BacktestCritiqueResponse(BaseModel):
    critique: str
