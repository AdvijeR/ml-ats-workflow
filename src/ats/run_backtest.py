from __future__ import annotations

from typing import Dict, Any, Type
import backtrader as bt
import pandas as pd
import numpy as np

from utils.feeds import PredictionData
from strategies.baseline import PortfolioThresholdLongFlatStrategy


def compute_sharpe(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    dr = daily_returns.dropna()
    if len(dr) < 2:
        return np.nan

    std = dr.std()
    if std == 0 or np.isnan(std):
        return np.nan

    return (dr.mean() / std) * np.sqrt(periods_per_year)


def compute_max_drawdown(daily_returns: pd.Series) -> float:
    equity = (1.0 + daily_returns.fillna(0.0)).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    return drawdown.min()


def run_portfolio_backtest(
    data_dict: Dict[str, pd.DataFrame],
    strategy_cls: Type[bt.Strategy] = PortfolioThresholdLongFlatStrategy,
    threshold: float = 0.0,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
) -> Dict[str, Any]:
    """
    Run a multi-stock portfolio backtest.

    Each dataframe must have:
    - datetime index
    - open, high, low, close, volume, pred_return
    """

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)

    for ticker, df in data_dict.items():
        df = df.copy().sort_index()

        required_cols = ["open", "high", "low", "close", "volume", "pred_return"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker} missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{ticker} dataframe index must be DatetimeIndex")

        data = PredictionData(dataname=df)
        cerebro.adddata(data, name=ticker)

    cerebro.addstrategy(strategy_cls, threshold=threshold)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat = results[0]

    daily_returns_dict = strat.analyzers.time_return.get_analysis()
    daily_returns = pd.Series(daily_returns_dict, dtype=float)
    daily_returns.index = pd.to_datetime(daily_returns.index)
    daily_returns = daily_returns.sort_index()

    final_value = cerebro.broker.getvalue()

    summary = {
        "final_value": final_value,
        "cumulative_return": (final_value / initial_cash) - 1.0,
        "sharpe": compute_sharpe(daily_returns),
        "max_drawdown": compute_max_drawdown(daily_returns),
        "trade_count": getattr(strat, "trade_count", None),
        "threshold": threshold,
    }

    return {
        "summary": summary,
        "daily_returns": daily_returns,
        "strategy": strat,
        "cerebro": cerebro,
    }