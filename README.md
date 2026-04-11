<h1 align="center">Signalist</h1>

<p align="center"><em>Cross-sectional factor research in US equities — momentum, mean reversion, and volume anomaly signals across 48 S&amp;P 500 constituents (2020–2024).</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License MIT"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/haadihammad/signalist/main/assets/dashboard.png" alt="Signalist Dashboard" width="800">
  <br>
  <strong><a href="https://signalist.streamlit.app/">Live Dashboard →</a></strong>
</p>  

## Overview

Signalist tests whether three cross-sectional equity factors—momentum (Jegadeesh & Titman, 1993), short-horizon mean reversion (De Bondt & Thaler, 1985), and a volume-conditioned return signal—combine into a tradable long-only portfolio in a liquid US large-cap universe. The project links each signal to its primary academic reference and evaluates out-of-sample behaviour over a volatile sample.

The backtest reports an annualised strategy return of **12.96%** versus **14.73%** for an SPY benchmark, Sharpe ratios of **0.421** (strategy) and **0.630** (SPY), and a mean monthly information coefficient of **−0.0373**. The negative IC indicates that composite rank had little positive alignment with next-month returns in this window—consistent with weaker momentum payoffs and factor crowding in the post-2018 regime.

**Signalist v2** will add **regime-conditional factor weighting** using **Hidden Markov Model** classification of market states; implementation is in development.

## Methodology

| Factor | Definition | Academic basis |
| --- | --- | --- |
| Momentum | 12–1 month cumulative return (skip-month formation) | Jegadeesh & Titman (1993) |
| Mean reversion | Negative prior-month return | De Bondt & Thaler (1985) |
| Volume anomaly | Prior-month return scaled by relative volume (vs. rolling volume norm) | Institutional accumulation / volume–return interaction literature |

Portfolios are built by cross-sectionally ranking raw factors each month, forming a weighted composite (40% / 30% / 30%), and selecting the **top decile (10 names)** with **equal weights** and **month-end rebalancing**. Realised portfolio returns use **t+1** holding-period returns so ranks at *t* never see returns at *t* (no look-ahead in the performance series).

## Results

| Metric | Strategy | SPY benchmark | Delta |
| --- | --- | --- | --- |
| Annualised return | 12.96% | 14.73% | −1.77% |
| Sharpe ratio | 0.421 | 0.630 | −0.209 |
| Max drawdown | −27.40% | −23.93% | −3.47% |
| Mean monthly IC | −0.0373 | n/a | n/a |
| Months backtested | 59 | 59 | — |

The strategy underperformed the benchmark on a risk-adjusted basis — a result consistent with documented momentum factor decay in the post-2018 period characterised by two structural breaks: COVID-19 (March 2020) and the Federal Reserve tightening cycle (2022). The negative mean IC suggests the composite ranking exhibited mild contrarian behaviour — a finding that motivates regime-conditional weighting in v2.

## Project structure

```
Signalist/
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Python dependencies (unpinned)
├── .gitignore                # Ignores venv, caches, env files, generated data CSVs
├── app.py                    # Streamlit dashboard (Signalist UI)
├── run_pipeline.py           # End-to-end factor + backtest pipeline and validation
├── data/
│   ├── __init__.py           # Package marker for data pipeline imports
│   ├── fetch_data.py         # Downloads raw prices/volume from Yahoo Finance (yfinance)
│   ├── clean_data.py         # Cleans raw series; writes monthly CSV panels
│   └── verify_data.py        # Quality checks on monthly CSV outputs
├── factors/
│   ├── __init__.py           # Package marker
│   ├── momentum.py           # 12–1 momentum factor
│   ├── mean_reversion.py     # Mean-reversion factor (−lagged return)
│   ├── volume_anomaly.py     # Volume-anomaly factor
│   └── composite.py          # Percentile ranks, composite score, composite rank
└── backtest/
    ├── __init__.py           # Package marker
    └── portfolio.py          # Long-only backtest, SPY benchmark, metrics, holdings
```

## Limitations

- **Survivorship bias:** The investable universe is conditioned on current large-cap membership and liquidity screens.
- **No transaction costs:** Reported returns omit commissions, spreads, and market impact.
- **Long-only:** The strategy does not implement short positions or factor-neutral overlays.
- **Structural break sensitivity:** Performance is sensitive to the COVID-19 shock and the 2022 rate cycle.
- **Single-universe concentration:** All inference is from one cross-section and one sample window.

## License and acknowledgements

This project is released under the **MIT License**.

Factor definitions follow Jegadeesh & Titman (1993) and De Bondt & Thaler (1985). Data are sourced from Yahoo Finance via the **yfinance** library.
