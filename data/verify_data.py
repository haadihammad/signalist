"""
Signality — verify monthly panel CSVs from the data pipeline.
"""

import os

import numpy as np
import pandas as pd


def _load_monthly_csv(path: str) -> pd.DataFrame:
    """Load CSV with first column as parsed datetime index (matches pandas to_csv default)."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = {
        "prices": os.path.join(script_dir, "monthly_prices.csv"),
        "returns": os.path.join(script_dir, "monthly_returns.csv"),
        "volume": os.path.join(script_dir, "monthly_volume.csv"),
    }

    overall_pass = True

    # --- Check 1: all files exist and are non-empty ---
    files_ok = True
    for label, p in paths.items():
        if not os.path.isfile(p):
            files_ok = False
            break
        if os.path.getsize(p) <= 0:
            files_ok = False
            break

    dfs: dict[str, pd.DataFrame] = {}
    if files_ok:
        try:
            loaded: dict[str, pd.DataFrame] = {}
            for label, p in paths.items():
                loaded[label] = _load_monthly_csv(p)
            non_empty = all(df.shape[0] > 0 and df.shape[1] > 0 for df in loaded.values())
            if non_empty:
                dfs = loaded
            else:
                files_ok = False
        except Exception:
            files_ok = False
            dfs = {}

    if files_ok:
        print("Check 1 (files exist and non-empty): PASS")
    else:
        print("Check 1 (files exist and non-empty): FAIL")
        overall_pass = False
        dfs = {}

    cols_match = False
    idx_match = False
    monthly_prices = monthly_returns = monthly_volume = None

    if dfs:
        monthly_prices = dfs["prices"]
        monthly_returns = dfs["returns"]
        monthly_volume = dfs["volume"]

        # --- Check 2: identical ticker columns ---
        cols_match = (
            list(monthly_prices.columns)
            == list(monthly_returns.columns)
            == list(monthly_volume.columns)
        )
        if cols_match:
            print("Check 2 (identical ticker columns): PASS")
        else:
            print("Check 2 (identical ticker columns): FAIL")
            overall_pass = False

        # --- Check 3: identical date indices ---
        idx_match = monthly_prices.index.equals(monthly_returns.index) and monthly_prices.index.equals(
            monthly_volume.index
        )
        if idx_match:
            print("Check 3 (identical date indices): PASS")
        else:
            print("Check 3 (identical date indices): FAIL")
            overall_pass = False

        # --- Check 4: no infinite returns ---
        rvals = monthly_returns.to_numpy(dtype=float, copy=False)
        no_inf = not np.isinf(rvals).any()
        if no_inf:
            print("Check 4 (monthly_returns has no infinite values): PASS")
        else:
            print("Check 4 (monthly_returns has no infinite values): FAIL")
            overall_pass = False

        # --- Check 5: finite returns between -0.8 and +0.8 (NaN allowed, e.g. first month) ---
        finite_mask = np.isfinite(rvals)
        bounds_ok = bool(
            finite_mask.any()
            and ((rvals[finite_mask] >= -0.8) & (rvals[finite_mask] <= 0.8)).all()
        )
        if bounds_ok:
            print("Check 5 (monthly_returns in [-0.8, +0.8]): PASS")
        else:
            print("Check 5 (monthly_returns in [-0.8, +0.8]): FAIL")
            overall_pass = False

        # --- Check 6: volume strictly positive ---
        vvals = monthly_volume.to_numpy(dtype=float, copy=False)
        vol_ok = bool(np.isfinite(vvals).all() and (vvals > 0).all())
        if vol_ok:
            print("Check 6 (monthly_volume > 0 everywhere): PASS")
        else:
            print("Check 6 (monthly_volume > 0 everywhere): FAIL")
            overall_pass = False

    else:
        print("Check 2 (identical ticker columns): FAIL")
        print("Check 3 (identical date indices): FAIL")
        print("Check 4 (monthly_returns has no infinite values): FAIL")
        print("Check 5 (monthly_returns in [-0.8, +0.8]): FAIL")
        print("Check 6 (monthly_volume > 0 everywhere): FAIL")
        overall_pass = False

    print("Tickers passing all checks:")
    if overall_pass and dfs and cols_match:
        print("  " + ", ".join(sorted(monthly_prices.columns)))
    else:
        print("  (none)")

    # Final status: READY only if every global check passed
    if overall_pass:
        print("DATA PIPELINE STATUS: READY")
    else:
        print("DATA PIPELINE STATUS: ERRORS FOUND")


if __name__ == "__main__":
    main()
