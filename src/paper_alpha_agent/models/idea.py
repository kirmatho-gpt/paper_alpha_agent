from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchIdea(BaseModel):
    idea_id: str
    paper_id: str
    title: str
    hypothesis: str
    signal_definition: str
    target_universe: list[str] = Field(default_factory=list)
    forecast_horizon: str
    frequency: str
    required_data: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class PrototypeSpec(BaseModel):
    prototype_id: str
    idea_id: str
    title: str
    objective: str
    feature_set: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    modeling_approach: str
    signal_logic: str
    evaluation_plan: list[str] = Field(default_factory=list)
    risk_controls: list[str] = Field(default_factory=list)
