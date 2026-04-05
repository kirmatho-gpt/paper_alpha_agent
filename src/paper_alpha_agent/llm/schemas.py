from __future__ import annotations

from pydantic import BaseModel, Field

from paper_alpha_agent.models.idea import PrototypeSpec, ResearchIdea
from paper_alpha_agent.models.paper import RelatedWorkItem


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
