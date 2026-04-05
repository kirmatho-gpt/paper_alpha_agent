from __future__ import annotations

from pydantic import BaseModel, Field


class BacktestResult(BaseModel):
    prototype_id: str
    instrument: str
    start_date: str
    end_date: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float
    observations: int
    assumptions: list[str] = Field(default_factory=list)
    critique: str | None = None
