from __future__ import annotations

from paper_alpha_agent.llm.client import LLMClient
from paper_alpha_agent.models.backtest import BacktestResult
from paper_alpha_agent.models.idea import PrototypeSpec
from paper_alpha_agent.tools.backtest_runner import BacktestRunner
from paper_alpha_agent.tools.market_data_client import MarketDataClient


def evaluate_prototypes(
    prototypes: list[PrototypeSpec],
    market_data_client: MarketDataClient,
    backtest_runner: BacktestRunner,
    llm_client: LLMClient,
    symbol: str = "SPY",
    start_date: str = "2020-01-01",
    end_date: str = "2020-12-31",
) -> list[BacktestResult]:
    results: list[BacktestResult] = []
    for prototype in prototypes:
        data = market_data_client.get_history(symbol=symbol, start_date=start_date, end_date=end_date)
        result = backtest_runner.run(prototype, data)
        critique = llm_client.critique_backtest(result)
        results.append(result.model_copy(update={"critique": critique.critique}))
    return results
