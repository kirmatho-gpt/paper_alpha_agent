from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from paper_alpha_agent.models.backtest import BacktestResult
from paper_alpha_agent.models.idea import PrototypeSpec


class BacktestRunner(ABC):
    @abstractmethod
    def run(self, prototype: PrototypeSpec, market_data: pd.DataFrame) -> BacktestResult:
        raise NotImplementedError


class SimpleBacktestRunner(BacktestRunner):
    def run(self, prototype: PrototypeSpec, market_data: pd.DataFrame) -> BacktestResult:
        frame = market_data.copy()
        frame["returns"] = frame["close"].pct_change().fillna(0.0)
        frame["signal"] = np.sign(frame["returns"].rolling(5).mean().fillna(0.0))
        frame["strategy_returns"] = frame["signal"].shift(1).fillna(0.0) * frame["returns"]
        equity_curve = (1 + frame["strategy_returns"]).cumprod()

        total_return = float(equity_curve.iloc[-1] - 1)
        ann_return = float(frame["strategy_returns"].mean() * 252)
        ann_vol = float(frame["strategy_returns"].std(ddof=0) * np.sqrt(252))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
        drawdown = equity_curve / equity_curve.cummax() - 1

        return BacktestResult(
            prototype_id=prototype.prototype_id,
            instrument=str(market_data["symbol"].iloc[0]),
            start_date=str(frame.index.min().date()),
            end_date=str(frame.index.max().date()),
            total_return=round(total_return, 4),
            annualized_return=round(ann_return, 4),
            annualized_volatility=round(ann_vol, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(float(drawdown.min()), 4),
            turnover=round(float(frame["signal"].diff().abs().fillna(0.0).mean()), 4),
            observations=len(frame),
            assumptions=[
                "Toy strategy uses rolling mean sign as a placeholder signal.",
                "No commissions, slippage, financing, or borrow costs.",
            ],
        )
