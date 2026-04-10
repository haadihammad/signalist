"""Volume-conditioned return factor (institutional accumulation intuition)."""

import numpy as np
import pandas as pd


def compute_volume_anomaly(
    monthly_returns: pd.DataFrame, monthly_volume: pd.DataFrame
) -> pd.DataFrame:
    """
    Relate **lagged** return to **current** volume versus a recent volume norm.

    Hypothesis: unusually strong price moves on **thin** volume may reflect orderly
    institutional absorption rather than noise; scaling prior return by relative volume
    highlights that pattern.

    For month *t*:

        \\bar{V}_{t}^{(3)} = \\text{3-month rolling mean of volume ending at } t{-}1
        \\quad (\\texttt{rolling(3).mean().shift(1)})

        R_t = V_t / \\bar{V}_{t}^{(3)}

        f_t = r_{t-1} / R_t

    Rows are NaN until the rolling average exists or where *R_t* is non-finite/zero.
    Values are winsorized **cross-sectionally** each month to the 1st and 99th percentiles.
    """
    if not monthly_returns.index.equals(monthly_volume.index):
        raise ValueError("monthly_returns and monthly_volume must share the same index.")
    if not monthly_returns.columns.equals(monthly_volume.columns):
        raise ValueError("monthly_returns and monthly_volume must share the same columns.")

    vol = monthly_volume.astype(float)
    r = monthly_returns.astype(float)

    roll = vol.rolling(window=3, min_periods=3).mean().shift(1)
    ratio = vol / roll
    score = r.shift(1) / ratio
    score = score.replace([np.inf, -np.inf], np.nan)

    clipped = score.copy()
    for dt in clipped.index:
        row = clipped.loc[dt]
        valid = row[np.isfinite(row)]
        if valid.size < 2:
            continue
        lo, hi = np.nanpercentile(valid.values, [1.0, 99.0])
        clipped.loc[dt] = row.clip(lower=lo, upper=hi)

    clipped[roll.isna()] = np.nan
    return clipped
