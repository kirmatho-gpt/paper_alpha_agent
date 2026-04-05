from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import pandas as pd


class MarketDataClient(ABC):
    @abstractmethod
    def get_history(self, symbol: str, start_date: str, end_date: str, frequency: str = "1D") -> pd.DataFrame:
        raise NotImplementedError


class DummyMarketDataClient(MarketDataClient):
    def get_history(self, symbol: str, start_date: str, end_date: str, frequency: str = "1D") -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        if len(dates) == 0:
            dates = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
        rng = np.random.default_rng(seed=7)
        returns = rng.normal(loc=0.0004, scale=0.01, size=len(dates))
        close = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
        frame = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": rng.integers(100_000, 500_000, size=len(dates)),
            },
            index=dates,
        )
        frame.index.name = "timestamp"
        frame["symbol"] = symbol
        return frame
