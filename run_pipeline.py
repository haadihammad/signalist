"""
Signalist end-to-end pipeline: load monthly panels, build composite factor, backtest, validate.
"""

import os
import sys
import traceback

import pandas as pd

from backtest.portfolio import run_backtest
from factors.composite import compute_composite


def _load_monthly(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "month"
    return df.sort_index()


def main() -> dict:
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")

    try:
        monthly_prices = _load_monthly(os.path.join(data_dir, "monthly_prices.csv"))
        monthly_returns = _load_monthly(os.path.join(data_dir, "monthly_returns.csv"))
        monthly_volume = _load_monthly(os.path.join(data_dir, "monthly_volume.csv"))

        composite = compute_composite(monthly_prices, monthly_returns, monthly_volume)
        composite_rank = composite["composite_rank"]

        bt = run_backtest(composite_rank, monthly_returns)

        port = bt["monthly_portfolio_returns"]
        spy = bt["monthly_spy_returns"]
        ann_p = bt["annualized_portfolio_return"]
        sharpe_p = bt["portfolio_sharpe"]
        mdd_p = bt["max_drawdown_portfolio"]
        ic = bt["information_coefficient"]
        holdings = bt["holdings"]

        checks: list[tuple[str, bool]] = []

        no_all_nan_rows = not composite_rank.isna().all(axis=1).any()
        checks.append(("Composite rank has no all-NaN rows", no_all_nan_rows))

        same_len = len(port) == len(spy)
        checks.append(("Portfolio vs SPY monthly series same length", same_len))

        ann_ok = (ann_p == ann_p) and (-0.5 <= ann_p <= 0.5)
        checks.append(("Annualized portfolio return in [-0.5, +0.5]", ann_ok))

        sharpe_ok = (sharpe_p == sharpe_p) and (-3.0 <= sharpe_p <= 5.0)
        checks.append(("Portfolio Sharpe in [-3, +5]", sharpe_ok))

        mdd_ok = (mdd_p == mdd_p) and (mdd_p < 0)
        checks.append(("Maximum drawdown is negative", mdd_ok))

        ic_mean = float(ic.mean(skipna=True))
        ic_ok = (ic_mean == ic_mean) and (-0.2 <= ic_mean <= 0.2)
        checks.append(("Mean information coefficient in [-0.2, +0.2]", ic_ok))

        hold_months_ok = holdings.index.equals(port.index)
        checks.append(("Holdings row for every portfolio-return month", hold_months_ok))

        counts_ok = True
        for dt, tickers in holdings.items():
            n = len(tickers)
            if n < 8 or n > 10:
                counts_ok = False
                break
        checks.append(("Holdings count between 8 and 10 each month", counts_ok))

        print("")
        print("Signalist - pipeline validation")
        print("-" * 60)
        for label, ok in checks:
            print(f"  [{ 'PASS' if ok else 'FAIL' }] {label}")
        print("-" * 60)

        all_ok = all(ok for _, ok in checks)

        if all_ok:
            print("SIGNALIST PIPELINE STATUS: READY")
            print("")
            print("Performance summary (aligned sample)")
            print(f"  Annualized return - portfolio: {ann_p:>8.2%}  |  SPY: {bt['annualized_spy_return']:>8.2%}")
            print(
                f"  Sharpe (rf=4% ann.) - portfolio: {sharpe_p:>8.3f}  |  SPY: {bt['spy_sharpe']:>8.3f}"
            )
            print(
                f"  Max drawdown - portfolio: {mdd_p:>8.2%}  |  SPY: {bt['max_drawdown_spy']:>8.2%}"
            )
            print(f"  Mean monthly IC: {ic_mean:>8.4f}")
            print(f"  Months backtested: {len(port)}")
        else:
            print("SIGNALIST PIPELINE STATUS: ERRORS FOUND")

        full_results = {
            "portfolio_returns": bt["monthly_portfolio_returns"],
            "spy_returns": bt["monthly_spy_returns"],
            "cumulative_portfolio": bt["cumulative_portfolio"],
            "cumulative_spy": bt["cumulative_spy"],
            "annualized_portfolio": bt["annualized_portfolio_return"],
            "annualized_spy": bt["annualized_spy_return"],
            "sharpe_portfolio": bt["portfolio_sharpe"],
            "sharpe_spy": bt["spy_sharpe"],
            "max_drawdown_portfolio": bt["max_drawdown_portfolio"],
            "max_drawdown_spy": bt["max_drawdown_spy"],
            "mean_ic": ic_mean,
            "rolling_sharpe": bt["rolling_6m_sharpe"],
            "monthly_holdings": bt["holdings"],
            "ic_series": bt["information_coefficient"],
            "composite_rank": composite_rank,
            "pipeline_ready": all_ok,
        }
        return full_results

    except Exception as exc:
        print("SIGNALIST PIPELINE STATUS: ERRORS FOUND")
        print(f"Pipeline error ({type(exc).__name__}): {exc}")
        traceback.print_exc()
        return {
            "portfolio_returns": pd.Series(dtype=float),
            "spy_returns": pd.Series(dtype=float),
            "cumulative_portfolio": pd.Series(dtype=float),
            "cumulative_spy": pd.Series(dtype=float),
            "annualized_portfolio": float("nan"),
            "annualized_spy": float("nan"),
            "sharpe_portfolio": float("nan"),
            "sharpe_spy": float("nan"),
            "max_drawdown_portfolio": float("nan"),
            "max_drawdown_spy": float("nan"),
            "mean_ic": float("nan"),
            "rolling_sharpe": pd.Series(dtype=float),
            "monthly_holdings": pd.Series(dtype=object),
            "ic_series": pd.Series(dtype=float),
            "composite_rank": pd.DataFrame(),
            "pipeline_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    out = main()
    sys.exit(0 if out.get("pipeline_ready", False) else 1)
