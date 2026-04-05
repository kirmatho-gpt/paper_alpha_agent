from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from paper_alpha_agent.models.backtest import BacktestResult
from paper_alpha_agent.models.idea import PrototypeSpec, ResearchIdea
from paper_alpha_agent.models.paper import RankedPaper, RelatedWorkItem


class ResearchReport(BaseModel):
    created_at: datetime
    universe: str
    start_date: str | None = None
    end_date: str | None = None
    discovered_count: int
    ranked_papers: list[RankedPaper] = Field(default_factory=list)
    related_work: dict[str, list[RelatedWorkItem]] = Field(default_factory=dict)
    selected_paper_ids: list[str] = Field(default_factory=list)
    ideas: list[ResearchIdea] = Field(default_factory=list)
    prototypes: list[PrototypeSpec] = Field(default_factory=list)
    backtests: list[BacktestResult] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    report_path: str | None = None
