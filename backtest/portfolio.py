"""Monthly long-only backtest from factor ranks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr


def _max_drawdown(cumulative_wealth: pd.Series) -> float:
    peak = cumulative_wealth.cummax()
    drawdown = cumulative_wealth / peak - 1.0
    return float(drawdown.min())


def _annualized_return(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if r.empty:
        return float("nan")
    gross = (1.0 + r).prod()
    years = len(r) / 12.0
    return float(gross ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def _monthly_rf(annual_rf: float = 0.04) -> float:
    return (1.0 + annual_rf) ** (1.0 / 12.0) - 1.0


def _annualized_sharpe(monthly_returns: pd.Series, annual_rf: float = 0.04) -> float:
    rf_m = _monthly_rf(annual_rf)
    x = monthly_returns.dropna() - rf_m
    if x.std(ddof=1) == 0 or x.empty:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12.0))


def _spy_monthly_returns(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    end_dl = end + pd.Timedelta(days=7)
    raw = yf.download(
        "SPY",
        start=start.strftime("%Y-%m-%d"),
        end=end_dl.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no rows for SPY.")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0].squeeze()
    close = close.sort_index()
    spy_monthly = close.resample("ME").last().pct_change()
    spy_monthly.name = "SPY"
    return spy_monthly


def _roll_sharpe_window(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    std = arr.std(ddof=1)
    if std == 0:
        return np.nan
    return float(arr.mean() / std * np.sqrt(12.0))


def run_backtest(
    composite_rank: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    annual_rf: float = 0.04,
) -> dict:
    """
    Long-only equal-weight top decile with **one-month lag** (no look-ahead).

    Each month *t*, select the ten tickers with the **largest** `composite_rank`
    (higher rank = stronger signal). Earn the **equal-weighted average** of their
    realized simple returns in the **following calendar month** *t+1* (first
    month-end in `monthly_returns` on or after *t* + 1 month-end).

    SPY month returns are downloaded for the same span (month-end aligned) for benchmarks.
    """
    cr = composite_rank.sort_index()
    rets = monthly_returns.sort_index()
    rets_idx = rets.index

    port_rets: list[float] = []
    port_dates: list[pd.Timestamp] = []
    ic_vals: list[float] = []
    ic_dates: list[pd.Timestamp] = []
    holdings: dict[pd.Timestamp, list[str]] = {}

    for dt_rank in cr.index:
        target = dt_rank + pd.offsets.MonthEnd(1)
        future = rets_idx[rets_idx >= target]
        if future.size == 0:
            continue
        dt_ret = future[0]

        row = cr.loc[dt_rank]
        eligible = row.dropna()
        if eligible.size == 0:
            continue
        top_tickers = eligible.nlargest(10).index.tolist()

        fwd = rets.loc[dt_ret, top_tickers]
        port_rets.append(float(fwd.mean(skipna=True)))
        port_dates.append(dt_ret)
        holdings[dt_ret] = top_tickers

        rnk = cr.loc[dt_rank]
        nxt = rets.loc[dt_ret]
        panel = pd.concat([rnk, nxt], axis=1, keys=["rank", "ret"]).dropna()
        if panel.shape[0] >= 3:
            rho, _ = spearmanr(panel["rank"], panel["ret"])
            ic_vals.append(float(rho) if rho == rho else np.nan)
        else:
            ic_vals.append(np.nan)
        ic_dates.append(dt_rank)

    portfolio_returns = pd.Series(port_rets, index=pd.DatetimeIndex(port_dates, name="month"))
    portfolio_returns = portfolio_returns.sort_index()

    ic_series = pd.Series(ic_vals, index=pd.DatetimeIndex(ic_dates, name="month"))

    spy_raw = _spy_monthly_returns(cr.index.min(), portfolio_returns.index.max())
    spy_returns = spy_raw.reindex(portfolio_returns.index)

    aligned = pd.concat(
        {"portfolio": portfolio_returns, "spy": spy_returns},
        axis=1,
    ).dropna()
    portfolio_returns = aligned["portfolio"]
    spy_returns = aligned["spy"]

    holdings_series = pd.Series(
        [holdings[d] for d in portfolio_returns.index],
        index=portfolio_returns.index,
        name="holdings",
    )

    rf_m = _monthly_rf(annual_rf)
    port_excess = portfolio_returns - rf_m

    rolling_6m_sharpe = port_excess.rolling(window=6, min_periods=6).apply(
        lambda s: _roll_sharpe_window(s.astype(float).to_numpy()),
        raw=False,
    )

    cum_port = (1.0 + portfolio_returns).cumprod()
    cum_spy = (1.0 + spy_returns).cumprod()

    ann_port = _annualized_return(portfolio_returns)
    ann_spy = _annualized_return(spy_returns)
    sharpe_port = _annualized_sharpe(portfolio_returns, annual_rf)
    sharpe_spy = _annualized_sharpe(spy_returns, annual_rf)
    mdd_port = _max_drawdown(cum_port)
    mdd_spy = _max_drawdown(cum_spy)

    results = {
        "monthly_portfolio_returns": portfolio_returns,
        "monthly_spy_returns": spy_returns,
        "cumulative_portfolio": cum_port,
        "cumulative_spy": cum_spy,
        "annualized_portfolio_return": ann_port,
        "annualized_spy_return": ann_spy,
        "portfolio_sharpe": sharpe_port,
        "spy_sharpe": sharpe_spy,
        "max_drawdown_portfolio": mdd_port,
        "max_drawdown_spy": mdd_spy,
        "information_coefficient": ic_series,
        "rolling_6m_sharpe": rolling_6m_sharpe.reindex(portfolio_returns.index),
        "holdings": holdings_series,
    }

    print("")
    print("Signalist - backtest performance summary")
    print("-" * 50)
    print(f"{'Annualized return (portfolio)':<36} {ann_port:>10.2%}")
    print(f"{'Annualized return (SPY)':<36} {ann_spy:>10.2%}")
    print(f"{'Sharpe ratio (portfolio, rf=4%)':<36} {sharpe_port:>10.3f}")
    print(f"{'Sharpe ratio (SPY, rf=4%)':<36} {sharpe_spy:>10.3f}")
    print(f"{'Max drawdown (portfolio)':<36} {mdd_port:>10.2%}")
    print(f"{'Max drawdown (SPY)':<36} {mdd_spy:>10.2%}")
    print(f"{'Mean monthly IC (Spearman)':<36} {ic_series.mean():>10.4f}")
    print(f"{'Backtest months (portfolio)':<36} {len(portfolio_returns):>10d}")
    print("-" * 50)

    return results
