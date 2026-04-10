"""Short-horizon mean-reversion (contrarian) factor."""

import pandas as pd


def compute_mean_reversion(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-reversion score tied to the De Bondt & Thaler (1985, 1987) overreaction story:
    extreme **prior** losers are expected to rebound, so a simple contrarian tilt is
    **minus last month's return**.

        f_t = − r_{t−1}

    Large positive scores follow sharp declines (strongest reversion tilt under this rule).

    The first calendar row is NaN (no lagged return). Shape, index, and columns match input.
    """
    return (-1.0 * monthly_returns.shift(1)).astype(float)
