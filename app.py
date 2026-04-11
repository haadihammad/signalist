import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import run_pipeline
from factors.composite import compute_composite


st.set_page_config(
    page_title="Signalist",
    layout="wide",
    page_icon="S",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Merriweather:ital,wght@0,400;0,700;1,400&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDeployButton"] {
        display: none;
    }

    .stApp {
        background: #f1efe2;
        font-family: 'Barlow', sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background: #f1efe2;
    }
    [data-testid="stMainBlockContainer"] {
        border-top: 4px solid #E3120B;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] {
        background: #1a1a1a;
    }
    [data-testid="stSidebar"] * {
        color: #cccccc !important;
        font-family: 'Barlow', sans-serif !important;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"],
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] p {
        color: #E3120B !important;
    }

    h1, h2 {
        font-family: 'Merriweather', serif !important;
        color: #1a1a1a !important;
    }
    p, div, span, label, li {
        font-family: 'Barlow', sans-serif;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #cccccc;
        border-top: 3px solid #1a1a1a;
        padding: 12px;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 9px !important;
        letter-spacing: 1.4px !important;
        text-transform: uppercase !important;
        color: #666666 !important;
        font-family: 'Barlow', sans-serif !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 20px !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
        font-family: 'Barlow', sans-serif !important;
    }

    .ec-overline {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.6px;
        color: #E3120B;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .ec-standfirst {
        font-size: 13px;
        color: #444444;
        font-style: italic;
        border-top: 2px solid #1a1a1a;
        padding-top: 8px;
        margin-bottom: 1.0rem;
        line-height: 1.5;
    }
    .ec-chart-title {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        color: #1a1a1a;
        border-top: 2px solid #1a1a1a;
        padding-top: 6px;
        margin-bottom: 4px;
        font-weight: 600;
    }
    [data-testid="stPlotlyChart"] {
        border: 1px solid #cccccc;
        background: #ffffff;
        padding: 4px;
    }
    /* Hide header chrome; sidebar stays expanded with no collapse control */
    [data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_economist_theme(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="#f1efe2",
        plot_bgcolor="#ffffff",
        font_family="Barlow, sans-serif",
        font_color="#1a1a1a",
        title_font_family="Merriweather, serif",
        title_font_size=13,
        title_font_color="#1a1a1a",
        legend=dict(bgcolor="#ffffff", bordercolor="#cccccc", borderwidth=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#eeeeee", linecolor="#1a1a1a")
    fig.update_yaxes(gridcolor="#eeeeee", linecolor="#1a1a1a")
    return fig


def _load_monthly_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


@st.cache_resource
def load_data():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    monthly_prices_path = os.path.join(data_dir, "monthly_prices.csv")
    monthly_returns_path = os.path.join(data_dir, "monthly_returns.csv")
    monthly_volume_path = os.path.join(data_dir, "monthly_volume.csv")

    if not (
        os.path.isfile(monthly_prices_path)
        and os.path.isfile(monthly_returns_path)
        and os.path.isfile(monthly_volume_path)
    ):
        from data import clean_data, fetch_data

        fetch_data.main()
        clean_data.main()

    monthly_prices = _load_monthly_csv(monthly_prices_path)
    monthly_returns = _load_monthly_csv(monthly_returns_path)
    monthly_volume = _load_monthly_csv(monthly_volume_path)

    factor_data = compute_composite(monthly_prices, monthly_returns, monthly_volume)
    results = run_pipeline.main()
    return factor_data, results


def _as_series(x, fallback_index=None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.copy()
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        return x.iloc[:, 0].copy()
    if fallback_index is None:
        return pd.Series(dtype=float)
    return pd.Series(index=fallback_index, dtype=float)


try:
    with st.spinner("Loading Signalist data..."):
        factor_data, results = load_data()
    composite_rank = factor_data.get("composite_rank", pd.DataFrame())

    portfolio_returns = _as_series(results.get("portfolio_returns"))
    spy_returns = _as_series(results.get("spy_returns"), fallback_index=portfolio_returns.index)
    if spy_returns.empty and not portfolio_returns.empty:
        spy_returns = pd.Series(index=portfolio_returns.index, dtype=float)

    cumulative_portfolio = _as_series(
        results.get("cumulative_portfolio"),
        fallback_index=portfolio_returns.index,
    )
    if cumulative_portfolio.empty and not portfolio_returns.empty:
        cumulative_portfolio = (1.0 + portfolio_returns).cumprod()

    cumulative_spy = _as_series(results.get("cumulative_spy"), fallback_index=spy_returns.index)
    if cumulative_spy.empty and not spy_returns.empty:
        cumulative_spy = (1.0 + spy_returns).cumprod()

    annualized_portfolio = float(results.get("annualized_portfolio", np.nan))
    if np.isnan(annualized_portfolio) and not portfolio_returns.empty:
        annualized_portfolio = float((1 + portfolio_returns).prod() ** (12 / len(portfolio_returns)) - 1)

    annualized_spy = float(results.get("annualized_spy", np.nan))
    if np.isnan(annualized_spy) and not spy_returns.empty:
        annualized_spy = float((1 + spy_returns).prod() ** (12 / len(spy_returns)) - 1)

    sharpe_portfolio = float(results.get("sharpe_portfolio", np.nan))
    sharpe_spy = float(results.get("sharpe_spy", np.nan))
    max_drawdown_portfolio = float(results.get("max_drawdown_portfolio", np.nan))
    max_drawdown_spy = float(results.get("max_drawdown_spy", np.nan))
    rolling_sharpe = _as_series(results.get("rolling_sharpe"), fallback_index=portfolio_returns.index)
    ic_series = _as_series(results.get("ic_series"))
    mean_ic = float(results.get("mean_ic", np.nan))
    if np.isnan(mean_ic) and not ic_series.empty:
        mean_ic = float(ic_series.mean(skipna=True))

    monthly_holdings = results.get("monthly_holdings", pd.Series(dtype=object))
    if isinstance(monthly_holdings, dict):
        monthly_holdings = pd.Series(monthly_holdings)
    if not isinstance(monthly_holdings, pd.Series):
        monthly_holdings = pd.Series(dtype=object)
    if monthly_holdings.index.dtype == object and len(monthly_holdings.index) > 0:
        monthly_holdings.index = pd.to_datetime(monthly_holdings.index)
    monthly_holdings = monthly_holdings.sort_index()

    momentum_raw = factor_data.get("momentum_raw", factor_data.get("momentum", pd.DataFrame()))
    reversion_raw = factor_data.get("reversion_raw", factor_data.get("reversion", pd.DataFrame()))
    volume_raw = factor_data.get("volume_raw", factor_data.get("volume", pd.DataFrame()))
except Exception:
    st.error(
        "Data pipeline error — please run run_pipeline.py first and ensure all CSV files exist in the data/ folder."
    )
    st.stop()

PAGES = ["Overview", "Factor Analysis", "Portfolio", "Risk", "Methodology"]

with st.sidebar:
    st.markdown(
        "<div style='font-family: monospace; font-size:16px; font-weight:700; color:#ffffff;'>SIGNALIST</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:10px; color:#E3120B; text-transform:uppercase; letter-spacing:1.2px;'>Quantitative Research</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="visible")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#8f8f8f;'>Universe: 48 S&P 500 constituents<br>"
        "Period: Jan 2020 to Dec 2024<br>"
        "Rebalance: Monthly</div>",
        unsafe_allow_html=True,
    )

if page == "Overview":
    st.markdown("<div class='ec-overline'>Equity Factor Research · 2020–2024</div>", unsafe_allow_html=True)
    st.markdown(
        "<h1>Cross-sectional momentum in a volatility-distorted decade</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            "<div class='ec-standfirst'>A systematic backtest of three equity factors across 48 S&P 500 "
            f"constituents reveals that composite signal ranking generates {annualized_portfolio:.2%} annualised "
            f"returns against a {annualized_spy:.2%} benchmark — with a negative mean information coefficient "
            "suggesting factor decay during high-volatility regimes.</div>"
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Annualised Return",
        f"{annualized_portfolio:.2%}",
        f"{(annualized_portfolio - annualized_spy):.2%}",
    )
    c2.metric(
        "Sharpe Ratio",
        f"{sharpe_portfolio:.3f}",
        f"{(sharpe_portfolio - sharpe_spy):.3f}",
    )
    c3.metric(
        "Max Drawdown",
        f"{max_drawdown_portfolio:.2%}",
        f"{(max_drawdown_portfolio - max_drawdown_spy):.2%}",
    )
    c4.metric("Mean Monthly IC", f"{mean_ic:.3f}")

    cum_df = pd.concat(
        [
            cumulative_portfolio.rename("Strategy"),
            cumulative_spy.rename("SPY"),
        ],
        axis=1,
    ).dropna()
    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=cum_df.index,
            y=cum_df["Strategy"],
            mode="lines",
            name="Strategy",
            line=dict(color="#1a1a1a", width=2),
        )
    )
    fig_cum.add_trace(
        go.Scatter(
            x=cum_df.index,
            y=cum_df["SPY"],
            mode="lines",
            name="SPY",
            line=dict(color="#E3120B", width=1.5, dash="dot"),
        )
    )
    if not cum_df.empty:
        x_last = cum_df.index.max()
        fig_cum.add_annotation(
            x=x_last,
            y=float(cum_df.loc[x_last, "Strategy"]),
            text=f"Strategy: {cum_df.loc[x_last, 'Strategy']:.2f}",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="#1a1a1a"),
        )
        fig_cum.add_annotation(
            x=x_last,
            y=float(cum_df.loc[x_last, "SPY"]),
            text=f"SPY: {cum_df.loc[x_last, 'SPY']:.2f}",
            showarrow=False,
            xanchor="left",
            font=dict(size=10, color="#E3120B"),
        )
    fig_cum.update_xaxes(showgrid=False)
    fig_cum.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    apply_economist_theme(fig_cum, "Cumulative returns — strategy vs benchmark")
    st.plotly_chart(fig_cum, use_container_width=True)
    st.caption(
        "Signalist v1 · Three-factor equity model · Universe: 48 S&P 500 constituents · "
        "Period: 2020–2024 · Built by Haadi Hammad"
    )
    st.markdown(
        "<div style='font-size:11px; color:#666666; font-style:italic;'>"
        "Source: Yahoo Finance via yfinance · Equal-weighted top-decile portfolio · Monthly rebalance"
        "</div>",
        unsafe_allow_html=True,
    )

    lcol, rcol = st.columns(2)
    with lcol:
        ret_colors = np.where(portfolio_returns.fillna(0) >= 0, "#006437", "#E3120B")
        fig_m = go.Figure(
            go.Bar(
                x=portfolio_returns.index,
                y=portfolio_returns.values,
                marker_color=ret_colors,
                name="Strategy",
            )
        )
        fig_m.add_hline(y=0, line_color="#E3120B", line_dash="dash", line_width=1)
        apply_economist_theme(fig_m, "Monthly returns — strategy")
        st.plotly_chart(fig_m, use_container_width=True)

    with rcol:
        fig_rs = go.Figure(
            go.Bar(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                marker_color="#1a1a1a",
                name="Rolling Sharpe",
            )
        )
        fig_rs.add_hline(y=0, line_color="#E3120B", line_dash="dash", line_width=1)
        apply_economist_theme(fig_rs, "Rolling Sharpe ratio (6-month)")
        st.plotly_chart(fig_rs, use_container_width=True)

elif page == "Factor Analysis":
    st.markdown("<div class='ec-overline'>Signal Decomposition</div>", unsafe_allow_html=True)
    st.markdown("<h1>Three orthogonal factors, one composite ranking signal</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ec-standfirst'>Momentum captures return persistence across 12-month horizons. "
        "Mean reversion exploits short-term overreaction. Volume anomaly isolates institutional accumulation "
        "patterns. Each factor is cross-sectionally ranked monthly before composite weighting.</div>",
        unsafe_allow_html=True,
    )

    tickers = sorted(list(composite_rank.columns)) if not composite_rank.empty else []
    if tickers:
        ticker = st.selectbox("Inspect individual ticker", tickers)
    else:
        ticker = None
        st.info("No ticker universe in composite rank panel.")

    if ticker:
        fig_sub = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03)
        fig_sub.add_trace(
            go.Scatter(x=momentum_raw.index, y=momentum_raw[ticker], mode="lines", line=dict(color="#1a1a1a"), name="Momentum"),
            row=1,
            col=1,
        )
        fig_sub.add_trace(
            go.Scatter(x=reversion_raw.index, y=reversion_raw[ticker], mode="lines", line=dict(color="#1a1a1a"), name="Reversion"),
            row=2,
            col=1,
        )
        fig_sub.add_trace(
            go.Scatter(x=volume_raw.index, y=volume_raw[ticker], mode="lines", line=dict(color="#1a1a1a"), name="Volume"),
            row=3,
            col=1,
        )
        fig_sub.update_yaxes(title_text="Momentum", row=1, col=1)
        fig_sub.update_yaxes(title_text="Reversion", row=2, col=1)
        fig_sub.update_yaxes(title_text="Volume", row=3, col=1)
        apply_economist_theme(fig_sub, f"Factor score history — {ticker}")
        st.plotly_chart(fig_sub, use_container_width=True)

    ic_vals_hist = ic_series.dropna()
    if len(ic_vals_hist) > 0:
        fig_ic = go.Figure(
            go.Histogram(x=ic_vals_hist.values, marker_color="#1a1a1a", nbinsx=20, name="IC")
        )
        fig_ic.add_vline(x=0, line_color="#E3120B", line_dash="dash")
        if not np.isnan(mean_ic):
            fig_ic.add_vline(x=mean_ic, line_color="#006437", line_dash="dash")
            fig_ic.add_annotation(
                x=mean_ic,
                y=1,
                yref="paper",
                text=f"Mean IC: {mean_ic:.3f}",
                showarrow=False,
                bgcolor="#ffffff",
                bordercolor="#cccccc",
            )
        apply_economist_theme(fig_ic, "Information coefficient distribution — monthly Spearman IC")
        st.plotly_chart(fig_ic, use_container_width=True)
    else:
        st.info("No information coefficient values available to plot.")
    st.markdown(
        (
            "<div class='ec-standfirst'>A negative mean IC of "
            f"{mean_ic:.3f} indicates that the composite ranking had a mildly contrarian relationship with "
            "subsequent returns during this sample period — consistent with momentum factor decay documented in "
            "post-2018 literature.</div>"
        ),
        unsafe_allow_html=True,
    )

    heat = composite_rank.copy()
    if not heat.empty and heat.size > 0:
        fig_hm = go.Figure(
            go.Heatmap(
                z=heat.values,
                x=heat.columns,
                y=heat.index,
                colorscale=[[0, "#ffffff"], [1, "#000000"]],
                colorbar=dict(title="Rank"),
            )
        )
        apply_economist_theme(fig_hm, "Composite rank heatmap — all tickers all months")
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No composite rank data for heatmap.")

elif page == "Portfolio":
    st.markdown("<div class='ec-overline'>Portfolio Construction</div>", unsafe_allow_html=True)
    st.markdown("<h1>Equal-weighted top-decile selection with monthly rebalancing</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ec-standfirst'>Each month the ten highest-ranked stocks by composite score are held at 10% "
        "weight. Portfolio is fully rebalanced at month-end. Transaction costs are not modelled.</div>",
        unsafe_allow_html=True,
    )

    all_tickers = []
    for _, held in monthly_holdings.items():
        if isinstance(held, (list, tuple, np.ndarray, pd.Series)):
            all_tickers.extend(list(held))
    freq = pd.Series(all_tickers).value_counts().head(20).sort_values(ascending=True)

    if not freq.empty:
        fig_freq = go.Figure(
            go.Bar(
                x=freq.values,
                y=freq.index,
                orientation="h",
                marker_color="#1a1a1a",
                name="Frequency",
            )
        )
        fig_freq.add_vline(x=float(freq.mean()), line_color="#E3120B", line_dash="dash")
        apply_economist_theme(fig_freq, "Holdings frequency — months in top-decile portfolio")
        st.plotly_chart(fig_freq, use_container_width=True)
    else:
        st.info("No holdings data available.")

    holdings_rows = []
    for dt, held in monthly_holdings.items():
        items = list(held) if isinstance(held, (list, tuple, np.ndarray, pd.Series)) else []
        row = {"Month": pd.to_datetime(dt)}
        for i in range(10):
            row[f"Position {i+1}"] = items[i] if i < len(items) else None
        holdings_rows.append(row)
    holdings_df = pd.DataFrame(holdings_rows).sort_values("Month") if holdings_rows else pd.DataFrame()
    if not holdings_df.empty:
        holdings_df["Month"] = holdings_df["Month"].dt.strftime("%Y-%m-%d")

    st.markdown("<div class='ec-chart-title'>Monthly holdings record</div>", unsafe_allow_html=True)
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)

elif page == "Risk":
    st.markdown("<div class='ec-overline'>Risk Analysis</div>", unsafe_allow_html=True)
    st.markdown("<h1>Drawdown anatomy and volatility exposure</h1>", unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='ec-standfirst'>The strategy experienced a maximum drawdown of "
            f"{max_drawdown_portfolio:.2%} versus {max_drawdown_spy:.2%} for the S&P 500. "
            "Periods of elevated market volatility systematically impaired factor performance — a finding "
            "that motivates regime-conditional strategy design.</div>"
        ),
        unsafe_allow_html=True,
    )

    rp = cumulative_portfolio.replace(0, np.nan).copy()
    rs = cumulative_spy.replace(0, np.nan).copy()
    dd_port = (rp - rp.cummax()) / rp.cummax()
    dd_spy = (rs - rs.cummax()) / rs.cummax()

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=dd_port.index,
            y=dd_port.values,
            mode="lines",
            name="Strategy",
            line=dict(color="#1a1a1a"),
            fill="tozeroy",
            fillcolor="rgba(26,26,26,0.20)",
        )
    )
    fig_dd.add_trace(
        go.Scatter(
            x=dd_spy.index,
            y=dd_spy.values,
            mode="lines",
            name="SPY",
            line=dict(color="#E3120B"),
            fill="tozeroy",
            fillcolor="rgba(227,18,11,0.15)",
        )
    )
    fig_dd.add_hline(y=0, line_color="#cccccc", line_dash="dash")
    apply_economist_theme(fig_dd, "Drawdown — strategy vs benchmark")
    st.plotly_chart(fig_dd, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        scatter_df = pd.concat(
            [spy_returns.rename("SPY"), portfolio_returns.rename("Strategy")],
            axis=1,
        ).dropna()
        if len(scatter_df) >= 2:
            fig_sc = go.Figure()
            fig_sc.add_trace(
                go.Scatter(
                    x=scatter_df["SPY"],
                    y=scatter_df["Strategy"],
                    mode="markers",
                    marker=dict(color="#1a1a1a", size=8),
                    name="Months",
                )
            )
            b1, b0 = np.polyfit(scatter_df["SPY"], scatter_df["Strategy"], 1)
            x_line = np.linspace(scatter_df["SPY"].min(), scatter_df["SPY"].max(), 100)
            y_line = b1 * x_line + b0
            fig_sc.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color="#E3120B", dash="dash"),
                    name="Regression",
                )
            )
            apply_economist_theme(fig_sc, "Strategy returns vs benchmark returns")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Not enough overlapping portfolio and SPY returns for scatter plot.")

    with c2:
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=portfolio_returns.dropna(), name="Portfolio", marker_color="#1a1a1a"))
        fig_box.add_trace(go.Box(y=spy_returns.dropna(), name="SPY", marker_color="#E3120B"))
        apply_economist_theme(fig_box, "Return distribution comparison")
        st.plotly_chart(fig_box, use_container_width=True)

    reg_df = pd.concat(
        [portfolio_returns.rename("portfolio"), spy_returns.rename("spy")],
        axis=1,
    ).dropna()
    reg_df["spy_vol_3m"] = reg_df["spy"].rolling(3).std()
    vol_threshold = reg_df["spy_vol_3m"].quantile(0.75)
    if len(reg_df) > 0 and np.isfinite(vol_threshold):
        reg_df["regime"] = np.where(reg_df["spy_vol_3m"] > vol_threshold, "High Vol", "Normal")
        high_vol_mask = reg_df["regime"] == "High Vol"
        normal_mask = ~high_vol_mask

        fig_reg = go.Figure()
        fig_reg.add_trace(
            go.Bar(
                x=reg_df.index[high_vol_mask],
                y=reg_df.loc[high_vol_mask, "portfolio"],
                marker_color="#E3120B",
                name="High Volatility",
            )
        )
        fig_reg.add_trace(
            go.Bar(
                x=reg_df.index[normal_mask],
                y=reg_df.loc[normal_mask, "portfolio"],
                marker_color="#1a1a1a",
                name="Normal",
            )
        )
        apply_economist_theme(fig_reg, "Monthly returns by volatility regime")
        fig_reg.update_layout(
            barmode="overlay",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                bgcolor="#ffffff",
                bordercolor="#cccccc",
                borderwidth=1,
            ),
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("Insufficient data for volatility regime chart.")
    st.markdown(
        "<div class='ec-standfirst'>Factor returns were systematically lower during high-volatility regimes — "
        "the primary motivation for regime-conditional signal weighting in the forthcoming Signalist v2.</div>",
        unsafe_allow_html=True,
    )

else:
    st.markdown("<div class='ec-overline'>Research Methodology</div>", unsafe_allow_html=True)
    st.markdown("<h1>Factor definitions, data sources, and backtest assumptions</h1>", unsafe_allow_html=True)

    st.markdown(
        "<div class='ec-chart-title'>Section 1 — Data</div>"
        "<div class='ec-standfirst'>Universe of 48 S&P 500 constituents selected for liquidity "
        "(median daily volume above 1 million shares). Adjusted closing prices and volume from Yahoo "
        "Finance via yfinance. Sample period January 2019 to December 2024. Monthly frequency via "
        "month-end resampling.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ec-chart-title'>Section 2 — Factors</div>"
        "<div class='ec-standfirst'>Momentum defined as 12-1 month cumulative return following Jegadeesh and "
        "Titman 1993. Mean reversion defined as negative prior month return following De Bondt and Thaler 1985. "
        "Volume anomaly defined as prior month return scaled by relative volume ratio capturing institutional "
        "accumulation patterns. All factors cross-sectionally ranked monthly on a 0-1 percentile scale before "
        "combination.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ec-chart-title'>Section 3 — Portfolio construction</div>"
        "<div class='ec-standfirst'>Composite score weighted 40% momentum, 30% mean reversion, 30% volume anomaly. "
        "Top 10 stocks by composite rank selected monthly. Equal weighting at 10% per position. Portfolio return "
        "computed as equal-weighted average of holdings returns in the following month to prevent look-ahead bias.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ec-chart-title'>Section 4 — Performance measurement</div>"
        "<div class='ec-standfirst'>Sharpe ratio computed with annualised risk-free rate of 4% converted to monthly "
        "equivalent. Information coefficient computed as Spearman rank correlation between composite rank at month t "
        "and actual returns at month t+1 across all stocks. Drawdown computed as percentage decline from rolling "
        "maximum of cumulative returns.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ec-chart-title'>Section 5 — Limitations</div>"
        "<div class='ec-standfirst'>Survivorship bias present as universe was selected from current S&P 500 constituents. "
        "Transaction costs and market impact not modelled. Short selling not implemented — strategy is long-only. "
        "Sample period includes two structural breaks (COVID-19 March 2020, Federal Reserve tightening cycle 2022) "
        "which may have suppressed factor performance relative to longer historical samples. Results should not be "
        "interpreted as indicative of future performance.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Signalist v2 will introduce regime-conditional factor weighting using Hidden Markov Model market state classification. This extension is currently in development.**"
    )
    st.markdown("<div class='ec-chart-title'>About</div>", unsafe_allow_html=True)
    st.markdown(
        "Signalist is an open-source quantitative research project exploring cross-sectional factor investing in "
        "top US equities. Built as an independent research initiative. Source code available on GitHub at "
        "<a href='https://github.com/haadihammad/signalist'>github.com/haadihammad/signalist</a>"
    )
