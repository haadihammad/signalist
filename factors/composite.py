"""Multi-factor composite score and cross-sectional ranking."""

import pandas as pd

from factors.mean_reversion import compute_mean_reversion
from factors.momentum import compute_momentum
from factors.volume_anomaly import compute_volume_anomaly


def compute_composite(
    monthly_prices: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    monthly_volume: pd.DataFrame,
) -> dict:
    """
    Build raw momentum, mean-reversion, and volume-anomaly factors, percentile-rank each
    cross-sectionally, blend them, and assign integer composite ranks (1..N, higher=better).
    Months with fewer than 20 valid composite scores are removed from **all** outputs.
    """
    momentum = compute_momentum(monthly_prices, monthly_returns)
    reversion = compute_mean_reversion(monthly_returns)
    volume = compute_volume_anomaly(monthly_returns, monthly_volume)

    # Higher raw factor → higher percentile (stronger signal).
    momentum_rank = momentum.rank(axis=1, pct=True, ascending=False)
    reversion_rank = reversion.rank(axis=1, pct=True, ascending=False)
    volume_rank = volume.rank(axis=1, pct=True, ascending=False)

    composite_score = (
        0.4 * momentum_rank + 0.3 * reversion_rank + 0.3 * volume_rank
    )

    # Largest composite_score → largest integer rank (rank N = best).
    composite_rank = composite_score.rank(axis=1, ascending=True)

    valid_n = composite_score.notna().sum(axis=1)
    keep = valid_n >= 20

    momentum = momentum.loc[keep]
    reversion = reversion.loc[keep]
    volume = volume.loc[keep]
    composite_score = composite_score.loc[keep]
    composite_rank = composite_rank.loc[keep]

    print("Signalist - compute_composite summary")
    print(f"  Date range: {composite_rank.index.min().date()} -> {composite_rank.index.max().date()}")
    print(f"  Valid months (>=20 names): {composite_rank.shape[0]}")
    print(f"  Tickers in panel: {composite_rank.shape[1]}")

    return {
        "momentum": momentum,
        "reversion": reversion,
        "volume": volume,
        "momentum_raw": momentum,
        "reversion_raw": reversion,
        "volume_raw": volume,
        "composite_score": composite_score,
        "composite_rank": composite_rank,
    }
