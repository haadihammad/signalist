"""Jegadeesh–Titman style momentum factor."""

import numpy as np
import pandas as pd


def compute_momentum(monthly_prices: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Classical skip-month (Jegadeesh & Titman, 1993) momentum from **levels**.

    The formation window uses prices **twelve months apart**, ending one month **before**
    the signal month *t*: cumulative return from the close at *t−12* to the close at *t−1*:

        m_t = (P_{t-1} − P_{t-12}) / P_{t-12}

    The current month *t* is excluded (no look-ahead in the return used for ranking).

    `monthly_returns` is accepted so callers can enforce panel alignment; it is not used
    in the formula. Index and columns must match `monthly_prices`.

    Returns
    -------
    pd.DataFrame
        Momentum scores with the same index and columns as the inputs. Rows are NaN until
        both *P_{t−12}* and *P_{t−1}* exist (fewer than twelve prior months of prices).
    """
    if not monthly_prices.index.equals(monthly_returns.index):
        raise ValueError("monthly_prices and monthly_returns must share the same index.")
    if not monthly_prices.columns.equals(monthly_returns.columns):
        raise ValueError("monthly_prices and monthly_returns must share the same columns.")

    p = monthly_prices.astype(float)
    num = p.shift(1) - p.shift(12)
    den = p.shift(12)
    mom = num / den
    mom = mom.replace([np.inf, -np.inf], np.nan)
    mom[den == 0] = np.nan
    return mom
