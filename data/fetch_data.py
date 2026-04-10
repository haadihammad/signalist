"""
AlphaLens — bulk download of adjusted close and volume via yfinance.
"""

import os

import pandas as pd
import yfinance as yf

# Universe: 50 large-cap / sector representatives for equity research
TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "AVGO",
    "CRM",
    "ADBE",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "BLK",
    "AXP",
    "WFC",
    "SCHW",
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    "MRK",
    "TMO",
    "ABT",
    "MDT",
    "AMZN",
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "TGT",
    "COST",
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "PSX",
    "CAT",
    "BA",
    "HON",
    "UPS",
    "RTX",
    "DE",
    "NEE",
    "DUK",
    "AMT",
    "PLD",
    "LIN",
    "VZ",
]

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"


def main() -> None:
    # Resolve paths: always write CSVs next to this file (…/data/raw_*.csv)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    prices_path = os.path.join(script_dir, "raw_prices.csv")
    volume_path = os.path.join(script_dir, "raw_volume.csv")

    # yfinance uses an exclusive `end`; add one calendar day so END_DATE is included
    end_exclusive = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Single call: adjusted OHLC (auto_adjust=True) + volume, columns grouped by ticker
    raw = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        end=end_exclusive,
        group_by="ticker",
        auto_adjust=True,
        progress=True,
    )

    # With multiple tickers, columns are MultiIndex (ticker, field)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame(
            {t: raw[t]["Close"] for t in TICKERS if t in raw.columns.get_level_values(0)}
        )
        volume = pd.DataFrame(
            {t: raw[t]["Volume"] for t in TICKERS if t in raw.columns.get_level_values(0)}
        )
    else:
        # Fallback if API returns a flat index (e.g. single ticker)
        prices = raw[["Close"]].copy()
        prices.columns = [TICKERS[0]]
        volume = raw[["Volume"]].copy()
        volume.columns = [TICKERS[0]]

    # Ensure datetime index and consistent column order
    prices.index = pd.to_datetime(prices.index)
    volume.index = pd.to_datetime(volume.index)
    prices = prices.sort_index()
    volume = volume.sort_index()

    # Drop tickers where more than 10% of rows are missing (thin history)
    nan_frac = prices.isna().mean(axis=0)
    keep = nan_frac <= 0.10
    prices = prices.loc[:, keep]
    volume = volume.loc[:, prices.columns]

    # Persist
    prices.to_csv(prices_path)
    volume.to_csv(volume_path)

    # Summary
    n_kept = prices.shape[1]
    idx_min = prices.index.min()
    idx_max = prices.index.max()
    print("AlphaLens — fetch summary")
    print(f"  Tickers kept (after NaN filter): {n_kept}")
    print(f"  Date range (index): {idx_min.date()} → {idx_max.date()}")
    print(f"  prices shape: {prices.shape}")
    print(f"  volume shape: {volume.shape}")
    print(f"  Wrote: {prices_path}")
    print(f"  Wrote: {volume_path}")


if __name__ == "__main__":
    main()
