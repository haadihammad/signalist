"""
Signality — clean raw OHLCV exports into analysis-ready monthly panels.
"""

import os

import pandas as pd


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Load raw data
    # -------------------------------------------------------------------------
    # Paths next to this module so runs work from any working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_prices_path = os.path.join(script_dir, "raw_prices.csv")
    raw_volume_path = os.path.join(script_dir, "raw_volume.csv")

    # Parse Date as datetime and use it as the row index (one row per calendar row in file).
    prices = pd.read_csv(raw_prices_path, parse_dates=["Date"], index_col="Date")
    volume = pd.read_csv(raw_volume_path, parse_dates=["Date"], index_col="Date")

    # -------------------------------------------------------------------------
    # 2. Align to business days only
    # -------------------------------------------------------------------------
    # Build a dense US-style business-day grid from first to last observation.
    idx_start = min(prices.index.min(), volume.index.min())
    idx_end = max(prices.index.max(), volume.index.max())
    business_days = pd.bdate_range(start=idx_start, end=idx_end)

    # Missing weekdays (e.g. holidays vs. Yahoo rows) become NaN; fill short gaps only.
    prices = prices.reindex(business_days).ffill(limit=3)
    volume = volume.reindex(business_days).ffill(limit=3)

    # Drop rows that still contain NaNs in either panel so prices and volume stay aligned.
    row_ok = ~(prices.isna().any(axis=1) | volume.isna().any(axis=1))
    prices = prices.loc[row_ok]
    volume = volume.loc[row_ok]

    # -------------------------------------------------------------------------
    # 3. Remove low-liquidity tickers
    # -------------------------------------------------------------------------
    # Median daily shares traded; drop names that are not realistically tradable at scale.
    median_daily_volume = volume.median(axis=0)
    liquid = median_daily_volume >= 1_000_000
    prices = prices.loc[:, liquid]
    volume = volume.loc[:, liquid]

    # -------------------------------------------------------------------------
    # 4. Monthly resampled prices (last trading day of each month)
    # -------------------------------------------------------------------------
    monthly_prices = prices.resample("ME").last()

    # -------------------------------------------------------------------------
    # 5. Monthly simple returns
    # -------------------------------------------------------------------------
    # First month has no prior price (NaN). Row kept so index matches prices/volume for QC.
    monthly_returns = monthly_prices.pct_change()

    # -------------------------------------------------------------------------
    # 6. Monthly volume (average daily volume within each month)
    # -------------------------------------------------------------------------
    monthly_volume = volume.resample("ME").mean()

    # -------------------------------------------------------------------------
    # 7. Save all outputs (initial write)
    # -------------------------------------------------------------------------
    out_prices = os.path.join(script_dir, "monthly_prices.csv")
    out_returns = os.path.join(script_dir, "monthly_returns.csv")
    out_volume = os.path.join(script_dir, "monthly_volume.csv")

    monthly_prices.to_csv(out_prices)
    monthly_returns.to_csv(out_returns)
    monthly_volume.to_csv(out_volume)

    # -------------------------------------------------------------------------
    # 8. Align all monthly panels to one common date index, then re-save
    # -------------------------------------------------------------------------
    # Resample can yield slightly different month-end stamps across series; keep only
    # dates present in every panel so verify_data and downstream code see one calendar.
    common_index = (
        monthly_prices.index.intersection(monthly_returns.index).intersection(monthly_volume.index)
    )
    common_index = common_index.sort_values()
    if len(common_index) == 0:
        raise ValueError("No overlapping dates across monthly_prices, monthly_returns, and monthly_volume.")

    monthly_prices = monthly_prices.reindex(common_index)
    monthly_returns = monthly_returns.reindex(common_index)
    monthly_volume = monthly_volume.reindex(common_index)

    monthly_prices.to_csv(out_prices)
    monthly_returns.to_csv(out_returns)
    monthly_volume.to_csv(out_volume)

    # -------------------------------------------------------------------------
    # 9. Final quality check (reflects aligned, re-saved panels)
    # -------------------------------------------------------------------------
    n_tickers = monthly_returns.shape[1]
    m_start = monthly_prices.index.min()
    m_end = monthly_prices.index.max()
    n_months_prices = monthly_prices.shape[0]
    n_months_returns = monthly_returns.shape[0]
    miss_pct = (
        100.0 * monthly_returns.isna().sum().sum() / monthly_returns.size
        if monthly_returns.size
        else 0.0
    )

    print("Signality — clean_data quality check")
    print(f"  Tickers after liquidity filter: {n_tickers}")
    print(f"  Monthly date range (prices index): {m_start.date()} → {m_end.date()}")
    print(f"  Number of month-end price rows: {n_months_prices}")
    print(f"  Number of monthly return rows: {n_months_returns}")
    print(f"  Missing values in monthly_returns: {miss_pct:.4f}%")
    if miss_pct >= 2.0:
        print("  Warning: missing share is at or above 2%.")


if __name__ == "__main__":
    main()
