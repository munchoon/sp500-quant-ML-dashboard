"""
app.py
────────
The dashboard. Everything here is read-only inference + display — no
training happens in this process. Run `python train_models.py` at least
once before this app has anything to load.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
import data_fetcher
import model_engine
import risk_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ═════════════════════════════════════════════════════════════════════════
# Page config + theme
# ═════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="S&P 500 MS-GJR-GARCH + ML Ensemble",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = {
    "bg": "#0A0E17", "surface": "#121826", "surface_alt": "#1A2233",
    "border": "#262E42", "text": "#E8EAF0", "text_dim": "#8993A8",
    "accent": "#F0A93B", "calm": "#2FBF71", "crisis": "#E5484D", "neutral": "#8993A8",
    "long": "#2FBF71", "short": "#E5484D",
}

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp {{ background-color: {COLORS['bg']}; }}
code, .mono {{ font-family: 'IBM Plex Mono', monospace !important; }}

#MainMenu, footer, header {{visibility: hidden;}}

.kpi-card {{
    background: linear-gradient(180deg, {COLORS['surface_alt']} 0%, {COLORS['surface']} 100%);
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
}}
.kpi-label {{
    font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase;
    color: {COLORS['text_dim']}; font-weight: 500; margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace; font-size: 28px; font-weight: 600;
    color: {COLORS['text']}; line-height: 1.2;
}}
.kpi-sub {{ font-size: 13px; color: {COLORS['text_dim']}; margin-top: 4px; }}

.badge {{
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 13px; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
}}
.badge-long {{ background: rgba(47,191,113,0.15); color: {COLORS['long']}; border: 1px solid {COLORS['long']}; }}
.badge-short {{ background: rgba(229,72,77,0.15); color: {COLORS['short']}; border: 1px solid {COLORS['short']}; }}
.badge-calm {{ background: rgba(47,191,113,0.15); color: {COLORS['calm']}; border: 1px solid {COLORS['calm']}; }}
.badge-crisis {{ background: rgba(229,72,77,0.15); color: {COLORS['crisis']}; border: 1px solid {COLORS['crisis']}; }}

.section-title {{
    font-size: 15px; font-weight: 600; color: {COLORS['text']};
    text-transform: uppercase; letter-spacing: 0.06em; margin: 6px 0 14px 0;
    border-left: 3px solid {COLORS['accent']}; padding-left: 10px;
}}
.component-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 14px; background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
    border-radius: 8px; margin-bottom: 8px;
}}
.disclosure {{
    background: {COLORS['surface']}; border-left: 3px solid {COLORS['accent']};
    border-radius: 6px; padding: 12px 16px; font-size: 13.5px; color: {COLORS['text_dim']};
    margin: 10px 0;
}}
hr {{ border-color: {COLORS['border']}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
    font=dict(family="IBM Plex Sans, sans-serif", color=COLORS["text"], size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
)


def base_layout(**overrides) -> dict:
    """PLOTLY_LAYOUT with per-chart overrides, merging (not clobbering)
    nested `xaxis`/`yaxis` dicts so a chart-specific tickmode/ticktext can
    coexist with the shared gridcolor/zerolinecolor styling."""
    layout = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PLOTLY_LAYOUT.items()}
    for key, val in overrides.items():
        if key in ("xaxis", "yaxis") and isinstance(val, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **val}
        else:
            layout[key] = val
    return layout


# ═════════════════════════════════════════════════════════════════════════
# Cached loaders
# ═════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading trained model artifacts...")
def load_engine() -> model_engine.ModelEngine:
    return model_engine.ModelEngine().load()


@st.cache_resource(ttl=900, show_spinner="Fetching live market data and generating today's signal...")
def get_live_signal(_engine: model_engine.ModelEngine, _cache_bust: str):
    # NOTE: cache_resource (not cache_data) — LiveSignal holds pandas/numpy
    # scalars that newer runtimes refuse to pickle, which broke page loads
    # with "Cannot serialize the return value". The object is read-only
    # downstream (never mutated), so sharing it across reruns is safe.
    return _engine.predict_today()


@st.cache_data(ttl=3600, show_spinner="Fetching recent price history...")
def get_ohlc(_cache_bust: str, start: str | None = None) -> pd.DataFrame:
    if start is None:
        start = (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    return data_fetcher.fetch_sp500_ohlc(start=start)


def kpi_card(label: str, value: str, sub: str = "", value_color: str | None = None):
    color = value_color or COLORS["text"]
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def signal_badge(signal: str) -> str:
    cls = "badge-long" if signal == "Long" else "badge-short"
    arrow = "▲" if signal == "Long" else "▼"
    return f'<span class="badge {cls}">{arrow} {signal.upper()}</span>'


def regime_badge(label: str) -> str:
    cls = "badge-calm" if label == "Calm" else "badge-crisis"
    return f'<span class="badge {cls}">{label.upper()}</span>'


# ═════════════════════════════════════════════════════════════════════════
# Boot: load engine, handle missing-artifacts gracefully
# ═════════════════════════════════════════════════════════════════════════
missing = [k for k, p in config.ARTIFACT_FILES.items()
           if k not in ("raw_dataset_cache",) and not p.exists()]
if missing:
    st.title("📈 S&P 500 MS-GJR-GARCH + ML Ensemble")
    st.error("No trained model artifacts found yet.")
    st.markdown(f"""
    This dashboard reads pre-trained artifacts from `/artifacts` — it never trains
    on page load (that would be far too slow for a free hosting tier). Missing: `{', '.join(missing)}`

    **Run this once before using the dashboard:**
    ```bash
    python train_models.py --quick   # ~1-2 min smoke test, verifies everything runs
    python train_models.py           # ~20-45 min full training run
    ```
    Then reload this page.
    """)
    st.stop()

try:
    engine = load_engine()
except Exception as exc:
    st.title("📈 S&P 500 MS-GJR-GARCH + ML Ensemble")
    st.error(f"Failed to load model artifacts: {exc}")
    st.info("If this is your first deploy, make sure `/artifacts` was committed to the repo "
            "(or fetched by your deploy step) and that `FRED_API_KEY` is set in Streamlit secrets.")
    st.stop()

cache_bucket = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")  # refreshes hourly automatically

signal_error = None
try:
    signal = get_live_signal(engine, cache_bucket)
except Exception as exc:
    signal_error = str(exc)
    signal = None
    logger.exception("Live signal generation failed")

# ═════════════════════════════════════════════════════════════════════════
# Header
# ═════════════════════════════════════════════════════════════════════════
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown("### 📈 S&P 500 Directional Forecast — MS-GJR-GARCH + ML Ensemble")
    st.caption("Live daily signal from a Random Forest / XGBoost / LSTM ensemble, "
               "regime-aware via a Markov-Switching GJR-GARCH volatility model.")
with h_right:
    if st.button("🔄 Refresh now", width='stretch'):
        get_live_signal.clear()
        get_ohlc.clear()
        st.rerun()
    st.caption(f"Refreshed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")

if signal_error:
    st.warning(f"⚠️ Couldn't fetch a fresh live signal this session ({signal_error}). "
               f"Showing the trained backtest below; the live KPI cards are unavailable until the next retry.")

st.markdown("<hr>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# KPI row
# ═════════════════════════════════════════════════════════════════════════
test_metrics = engine.get_test_period_metrics()
ens = test_metrics["Ensemble"]

# ── Previous completed session: what the model called vs. what happened ──
# Live-computed every page load (signal.previous, scored from data through
# T-1 with the identical code path as today's signal), so it rolls forward
# daily with no retraining. Falls back to the frozen training backtest tail
# only if live inference failed this session.
_pr = signal.previous if signal is not None else None
if _pr is not None:
    _prev_sig, _prev_pred, _prev_conf = _pr.signal, _pr.predicted_return_pct, _pr.confidence_pct
    _prev_actual, _prev_hit = _pr.actual_return_pct, _pr.hit
    _prev_date, _prev_close, _prev_live = _pr.session_date, _pr.sp500_close, True
    _prev = True
else:
    _prev = engine.backtest_df.iloc[-1] if len(engine.backtest_df) else None
    _prev_live = False
if _prev is not None and not _prev_live:
    _prev_pred = float(_prev["Pred_Ensemble"])
    _prev_sig = "Long" if _prev_pred > 0 else "Short"
    _prev_actual = float(_prev["Actual_Return"])
    _prev_hit = (_prev_pred > 0) == (_prev_actual > 0)
    _prev_date = engine.backtest_df.index[-1]
    _prev_close = float(_prev["SP500_Close"])
    _prev_comp_sigs = [float(_prev[c]) > 0 for c in ("Pred_RF", "Pred_XGB", "Pred_LSTM_Regime")]
    _prev_agree = float(sum(1 for s in _prev_comp_sigs if ("Long" if s else "Short") == _prev_sig) / 3 * 100)
    _prev_ref = engine.backtest_df["Pred_Ensemble"].abs()
    _prev_mag = float((_prev_ref < abs(_prev_pred)).mean() * 100) if len(_prev_ref) else 50.0
    _prev_conf = float(min(100, max(0, 0.5 * _prev_agree + 0.5 * _prev_mag)))
elif _prev is None:
    _prev_sig, _prev_pred, _prev_conf = "—", 0.0, 0.0
    _prev_actual, _prev_hit, _prev_date, _prev_close = 0.0, False, None, 0.0

k1, k2, k3, k4 = st.columns(4)
with k1:
    if signal:
        kpi_card("Signal — next session", signal_badge(signal.signal),
                  f"as of {signal.as_of_date.date()} · confidence {signal.confidence_pct:.0f}%")
    else:
        kpi_card("Signal — next session", "—", "unavailable")
with k2:
    if signal:
        kpi_card("Market Regime", regime_badge(signal.regime_label),
                  f"{signal.regime_prob_crisis_pct:.1f}% crisis probability")
    else:
        kpi_card("Market Regime", "—", "unavailable")
with k3:
    if _prev is not None:
        kpi_card("Previous call — model predicted", signal_badge(_prev_sig),
                  f"{_prev_date.date()} · pred {_prev_pred:+.4f}% · confidence {_prev_conf:.0f}%"
                  + (" · live" if _prev_live else " · backtest"))
    else:
        kpi_card("Previous call — model predicted", "—", "unavailable")
with k4:
    if _prev is not None:
        _outcome = "HIT ✓" if _prev_hit else "MISS ✗"
        kpi_card("Previous session — actual outcome", f"{_prev_actual:+.2f}% {_outcome}",
                  f"S&P close {_prev_close:,.2f}",
                  value_color=COLORS["calm"] if _prev_hit else COLORS["crisis"])
    else:
        kpi_card("Previous session — actual outcome", "—", "unavailable")

st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════════
tab_signal, tab_charts, tab_explain, tab_risk, tab_about = st.tabs(
    ["🎯 Live Signal", "📊 Charts", "🔍 Explainability", "⚠️ Risk", "📄 About the Model"]
)

# ── TAB: Live Signal ───────────────────────────────────────────────────
with tab_signal:
    if not signal:
        st.info("Live signal unavailable this session — see the warning above.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="section-title">Ensemble Breakdown</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="disclosure">
            Predicted return for the next session: <b class="mono">{signal.predicted_return_pct:+.4f}%</b><br>
            S&P 500 last close: <b class="mono">{signal.sp500_last_close:,.2f}</b> (as of {signal.as_of_date.date()})
            </div>
            """, unsafe_allow_html=True)
            for comp in signal.components:
                badge = signal_badge(comp.signal)
                st.markdown(f"""
                <div class="component-row">
                    <span>{comp.name} <span style="color:{COLORS['text_dim']}">(weight {comp.weight:.0%})</span></span>
                    <span class="mono">{comp.predicted_return_pct:+.4f}%&nbsp;&nbsp;{badge}</span>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-title">Regime & Volatility</div>', unsafe_allow_html=True)
            rstats = engine.get_regime_stats()
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=signal.regime_prob_crisis_pct,
                number={"suffix": "%", "font": {"color": COLORS["text"], "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLORS["text_dim"]},
                    "bar": {"color": COLORS["crisis"] if signal.regime_prob_crisis_pct > 50 else COLORS["calm"]},
                    "bgcolor": COLORS["surface"],
                    "steps": [
                        {"range": [0, 30], "color": "rgba(47,191,113,0.15)"},
                        {"range": [30, 70], "color": "rgba(240,169,59,0.12)"},
                        {"range": [70, 100], "color": "rgba(229,72,77,0.15)"},
                    ],
                },
                title={"text": "Crisis regime probability", "font": {"size": 13, "color": COLORS["text_dim"]}},
            ))
            fig.update_layout(**{**PLOTLY_LAYOUT, "height": 220, "margin": dict(l=20, r=20, t=40, b=10)})
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            st.caption(f"Historically: Calm regime ≈ {rstats['calm_mean_return']:.3f}%/day mean, "
                       f"{rstats['calm_vol']:.3f}% vol · Crisis regime ≈ {rstats['crisis_mean_return']:.3f}%/day mean, "
                       f"{rstats['crisis_vol']:.3f}% vol")
            st.metric("Next-session volatility forecast (GJR-GARCH)", f"{signal.next_day_vol_forecast_pct:.3f}%")

        if signal.warnings:
            for w in signal.warnings:
                st.caption(f"ℹ️ {w}")

# ── TAB: Charts ─────────────────────────────────────────────────────────
with tab_charts:
    bt = engine.backtest_df

    _min_y, _max_y = int(bt.index.min().year), int(bt.index.max().year)
    _default_start = max(_min_y, _max_y - 5)
    year_range = st.select_slider(
        "Chart year range",
        options=list(range(_min_y, _max_y + 1)),
        value=(_default_start, _max_y),
        help="Slide to zoom all charts below into specific years. Model-comparison table stays full-period.",
    )
    bt_f = bt[(bt.index.year >= year_range[0]) & (bt.index.year <= year_range[1])]
    if len(bt_f) == 0:
        st.warning(f"No backtest data in {year_range[0]}–{year_range[1]}.")
    st.caption(f"Showing {year_range[0]}–{year_range[1]} · {len(bt_f):,} sessions (full OOS: {_min_y}–{_max_y}).")

    st.markdown('<div class="section-title">Cumulative Net Return — Ensemble vs Buy-and-Hold (OOS test)</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_f.index, y=(bt_f["Ensemble_Equity"] - 1) * 100, name="Ensemble strategy",
                              line=dict(color=COLORS["accent"], width=2)))
    fig.add_trace(go.Scatter(x=bt_f.index, y=(bt_f["BuyHold_Equity"] - 1) * 100, name="Buy & Hold",
                              line=dict(color=COLORS["text_dim"], width=1.5, dash="dot")))
    fig.update_layout(**base_layout(height=380, yaxis_title="Cumulative net return (%)",
                                    xaxis=dict(rangeslider=dict(visible=True), type="date")))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Leveraged Simulation — constant daily leverage</div>',
                unsafe_allow_html=True)
    st.caption("Hypothetical overlay: each day's net strategy return × L, compounded with daily rebalancing. "
               "Gross of borrowing/financing costs — real leveraged products also pay those.")
    lev_choices = st.multiselect("Leverage multiples", options=[1, 2, 3, 4], default=[1, 2, 3],
                                 help="Curves and metrics follow the year-range slider above.",
                                 key="lev_mults")
    if len(bt_f) and lev_choices:
        _LEV_COLORS = {1: COLORS["accent"], 2: COLORS["calm"], 3: "#6C5CE7", 4: COLORS["crisis"]}
        fig = go.Figure()
        lev_rows = []
        _base_rets = bt_f["Ensemble_Net_Return"].to_numpy()
        for L in sorted(lev_choices):
            _lr = _base_rets * L
            _eq = risk_metrics.equity_curve(_lr)
            fig.add_trace(go.Scatter(x=bt_f.index, y=(_eq - 1) * 100, name=f"{L}x leveraged",
                                     line=dict(color=_LEV_COLORS.get(L, COLORS["text_dim"]),
                                               width=2 if L > 1 else 1.5)))
            lev_rows.append({
                "Strategy": f"{L}x",
                "Net Return": f"{(_eq[-1] - 1) * 100:.2f}%",
                "Sharpe": f"{risk_metrics.annualised_sharpe(_lr):.3f}",
                "Max Drawdown": f"{risk_metrics.max_drawdown(_lr):.2f}%",
            })
        fig.update_layout(**base_layout(height=380, yaxis_title="Cumulative net return (%)",
                                        xaxis=dict(rangeslider=dict(visible=True), type="date")))
        st.plotly_chart(fig, width='stretch')
        st.dataframe(pd.DataFrame(lev_rows).set_index("Strategy"), width='stretch')
        st.caption("Metrics computed on the selected year range with the thesis formulas "
                   "(Sharpe = mean/sd·√252, sample std). Sharpe is scale-invariant so it barely moves "
                   "with L — return and drawdown do the moving.")
    elif not lev_choices:
        st.info("Select at least one leverage multiple.")

    st.markdown('<div class="section-title">S&P 500 — Price with Recent Ensemble Signal</div>', unsafe_allow_html=True)
    try:
        ohlc = get_ohlc(cache_bucket, f"{year_range[0]}-01-01")
        ohlc_f = ohlc[(ohlc.index.year >= year_range[0]) & (ohlc.index.year <= year_range[1])]
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(
            x=ohlc_f.index, open=ohlc_f["Open"], high=ohlc_f["High"], low=ohlc_f["Low"], close=ohlc_f["Close"],
            name="S&P 500", increasing_line_color=COLORS["calm"], decreasing_line_color=COLORS["crisis"],
        ))
        overlay = bt_f.reindex(ohlc_f.index).dropna(subset=["Ensemble_Signal"])
        longs = overlay[overlay["Ensemble_Signal"] == 1]
        shorts = overlay[overlay["Ensemble_Signal"] == -1]
        y_offset = ohlc_f["Low"].min() * 0.985
        if len(longs):
            fig.add_trace(go.Scatter(x=longs.index, y=[y_offset] * len(longs), mode="markers", name="Long signal",
                                      marker=dict(symbol="triangle-up", color=COLORS["long"], size=7)))
        if len(shorts):
            fig.add_trace(go.Scatter(x=shorts.index, y=[y_offset] * len(shorts), mode="markers", name="Short signal",
                                      marker=dict(symbol="triangle-down", color=COLORS["short"], size=7)))
        fig.update_layout(**PLOTLY_LAYOUT, height=420, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, width='stretch')
        st.caption("Signal markers shown only for dates within the saved OOS test-period backtest; "
                   "recent dates beyond the backtest window show price only.")
    except Exception as exc:
        st.info(f"Live candlestick unavailable this session ({exc}).")

    st.markdown('<div class="section-title">Regime Probability Over Time (OOS test period)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt_f.index, y=bt_f["Regime_Prob_Crisis"] * 100, fill="tozeroy", name="Crisis probability",
                              line=dict(color=COLORS["crisis"], width=1), fillcolor="rgba(229,72,77,0.25)"))
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["text_dim"])
    fig.update_layout(**base_layout(height=280, yaxis_title="Crisis probability (%)", yaxis_range=[0, 100],
                                    xaxis=dict(rangeslider=dict(visible=True), type="date")))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Model Comparison — OOS Test Period</div>', unsafe_allow_html=True)
    rows = []
    for name, m in test_metrics.items():
        rows.append({
            "Model": name,
            "Directional Acc.": f"{m['directional_accuracy_pct']:.1f}%" if m.get("directional_accuracy_pct") is not None else "—",
            "Sharpe": f"{m['annualised_sharpe']:.3f}" if m.get("annualised_sharpe") is not None else "—",
            "Net Return": f"{m['total_net_return_pct']:.2f}%" if m.get("total_net_return_pct") is not None else "—",
            "Max Drawdown": f"{m['max_drawdown_pct']:.2f}%" if m.get("max_drawdown_pct") is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width='stretch')
    st.caption("\"LSTM\" and \"LSTM + Regime\" show the same standalone Sharpe by construction — "
               "regime-scaling only rescales the *magnitude* of the LSTM's prediction, it never flips its "
               "sign, so the long/short trading signal (and therefore the standalone backtest) is identical. "
               "The scaling matters once this component is blended into the weighted ensemble, where "
               "relative magnitude — not just sign — determines the combined signal.")

# ── TAB: Explainability ──────────────────────────────────────────────────
with tab_explain:
    st.markdown('<div class="section-title">Random Forest Feature Importance (permutation, held-out validation)</div>',
                unsafe_allow_html=True)
    imp = engine.feature_importance_df.sort_values("Importance", ascending=True).tail(15)
    fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h",
                            marker_color=COLORS["accent"]))
    fig.update_layout(**PLOTLY_LAYOUT, height=440, xaxis_title="Mean increase in MSE when shuffled")
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    st.markdown('<div class="section-title">XGBoost SHAP Summary (test-period sample)</div>', unsafe_allow_html=True)
    if engine.shap_reference:
        shap_vals = engine.shap_reference["shap_values"]
        feat_vals = engine.shap_reference["feature_values"]
        names = engine.shap_reference["feature_names"]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_abs)[-15:]

        fig = go.Figure()
        rng = np.random.default_rng(42)
        for rank, idx in enumerate(order):
            fv = feat_vals[:, idx]
            fv_norm = (fv - fv.min()) / (fv.max() - fv.min() + 1e-9)
            jitter = rng.uniform(-0.35, 0.35, size=len(fv))
            fig.add_trace(go.Scatter(
                x=shap_vals[:, idx], y=[rank + j for j in jitter], mode="markers",
                marker=dict(color=fv_norm, colorscale=[[0, "#F0A93B"], [1, "#6C5CE7"]], size=5, opacity=0.65,
                            showscale=(rank == len(order) - 1),
                            colorbar=dict(title="Feature<br>value", tickvals=[0, 1], ticktext=["Low", "High"],
                                          len=0.5) if rank == len(order) - 1 else None),
                name=names[idx], showlegend=False,
                hovertemplate=f"{names[idx]}<br>SHAP=%{{x:.4f}}<extra></extra>",
            ))
        fig.update_layout(**base_layout(
            height=480,
            yaxis=dict(tickmode="array", tickvals=list(range(len(order))), ticktext=[names[i] for i in order]),
            xaxis_title="SHAP value (impact on predicted return)"))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    else:
        st.info("No cached SHAP reference found — re-run train_models.py to generate one.")

    if signal and signal.shap_today:
        st.markdown('<div class="section-title">What\'s Driving Today\'s Prediction (XGBoost)</div>', unsafe_allow_html=True)
        items = sorted(signal.shap_today.items(), key=lambda kv: kv[1])
        feats, vals = [k for k, _ in items], [v for _, v in items]
        colors = [COLORS["crisis"] if v < 0 else COLORS["calm"] for v in vals]
        fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker_color=colors))
        fig.update_layout(**PLOTLY_LAYOUT, height=520, xaxis_title="SHAP contribution to today's predicted return")
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

# ── TAB: Risk ───────────────────────────────────────────────────────────
with tab_risk:
    if not signal:
        st.info("Risk metrics need a live signal — unavailable this session.")
    else:
        st.markdown('<div class="section-title">Value at Risk / Expected Shortfall — Next Session</div>',
                    unsafe_allow_html=True)
        st.caption("Not part of the original thesis — added for daily live use. Historical = empirical "
                   "percentile of trailing 252-day returns. Parametric = closed-form from the GJR-GARCH's "
                   "current volatility forecast and fitted Student-t tail shape.")
        cols = st.columns(len(signal.var_es))
        for col, r in zip(cols, signal.var_es):
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{int(r.confidence*100)}% Confidence</div>
                    <div style="display:flex; justify-content:space-between; margin-top:8px;">
                        <div>
                            <div class="kpi-sub">Historical VaR</div>
                            <div class="kpi-value" style="font-size:20px;">{r.historical_var_pct:.2f}%</div>
                        </div>
                        <div>
                            <div class="kpi-sub">Historical ES</div>
                            <div class="kpi-value" style="font-size:20px; color:{COLORS['crisis']};">{r.historical_es_pct:.2f}%</div>
                        </div>
                    </div>
                    <hr style="margin:10px 0;">
                    <div style="display:flex; justify-content:space-between;">
                        <div>
                            <div class="kpi-sub">Parametric VaR</div>
                            <div class="kpi-value" style="font-size:20px;">{r.parametric_var_pct:.2f}%</div>
                        </div>
                        <div>
                            <div class="kpi-sub">Parametric ES</div>
                            <div class="kpi-value" style="font-size:20px; color:{COLORS['crisis']};">{r.parametric_es_pct:.2f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Return Distribution (trailing 252 sessions)</div>',
                    unsafe_allow_html=True)
        try:
            recent_df = engine.fetch_live_dataset()
            recent_returns = recent_df["SP500_ret"].dropna().tail(config.VAR_LOOKBACK_DAYS)
            fig = go.Figure(go.Histogram(x=recent_returns, nbinsx=60, marker_color=COLORS["accent"]))
            fig.add_vline(x=-signal.var_es[0].historical_var_pct, line_color=COLORS["crisis"], line_dash="dash",
                          annotation_text="95% VaR", annotation_font_color=COLORS["crisis"])
            fig.update_layout(**PLOTLY_LAYOUT, height=320, xaxis_title="Daily return (%)", yaxis_title="Frequency")
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        except Exception as exc:
            st.info(f"Distribution chart unavailable this session ({exc}).")

# ── TAB: About ──────────────────────────────────────────────────────────
with tab_about:
    thesis_w, thesis_m = engine.get_thesis_reference()
    live_w = engine.metadata["ensemble_weights"]

    st.markdown("""
### A Hybrid MS-GJR-GARCH and Machine Learning Ensemble Framework
**Chin Mun Choon · Universiti Malaya · SIT3025 Statistical Science Project (2026)**
*Supervisor: Dr. Tan Shay Kee*

This dashboard is a live, continuously-refreshed Python port of the R Markdown methodology
developed for the FYP thesis above. Two stages, exactly as in the paper:

1. **Econometric layer** — a 2-state Hidden Markov Model classifies each trading day as
   **Calm** or **Crisis** from daily S&P 500 returns; a **GJR-GARCH(1,1)** is fit separately
   within each regime to capture the asymmetric "leverage effect" (negative shocks raise
   volatility more than positive shocks of the same size).
2. **ML ensemble layer** — the regime probability, 22 macro/technical/volatility features,
   and 3 autoregressive lags feed a **Random Forest**, **XGBoost**, and **LSTM** (21-day
   lookback), whose regime-scaled predictions are combined by Sharpe-maximised weights into
   one directional signal, traded long/short with a 5 bps transaction cost per position change.
    """)

    st.markdown('<div class="section-title">Live pipeline vs. thesis reference (OOS test period)</div>',
                unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        "Metric": ["Ensemble weights (RF / XGB / LSTM)", "Sharpe ratio", "Net return", "Max drawdown",
                   "Directional accuracy"],
        "Thesis (Rmd, 2025 data vintage)": [
            f"{thesis_w['rf']:.0%} / {thesis_w['xgb']:.0%} / {thesis_w['lstm']:.0%}",
            f"{thesis_m['sharpe']:.3f}", f"{thesis_m['net_return_pct']:.2f}%",
            f"{thesis_m['max_drawdown_pct']:.2f}%", f"{thesis_m['directional_accuracy_pct']:.1f}%",
        ],
        "This live pipeline": [
            f"{live_w['rf']:.0%} / {live_w['xgb']:.0%} / {live_w['lstm']:.0%}",
            f"{ens['annualised_sharpe']:.3f}", f"{ens['total_net_return_pct']:.2f}%",
            f"{ens['max_drawdown_pct']:.2f}%", f"{ens['directional_accuracy_pct']:.1f}%",
        ],
    })
    st.dataframe(comp_df.set_index("Metric"), width='stretch')

    st.markdown('<div class="section-title">Notes on fidelity — where this app deliberately differs from the Rmd</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="disclosure">
<b>1. Ensemble weights are fit on a validation split, not the test set.</b> The notebook's grid
search maximises Sharpe directly on the OOS test window, which fits the weights to the same
data later used to report performance. This app carves a validation block out of the training
period instead (see <code>config.py</code> / <code>train_models.py</code>), so the numbers above
are a genuinely out-of-sample read — and won't exactly match the thesis's 262.95% / 1.268 Sharpe.<br><br>
<b>2. Newer data vintage.</b> Markets since the thesis was written are now part of the training
history, and Yahoo/FRED occasionally revise historical prints.<br><br>
<b>3. R → Python port.</b> <code>rugarch</code> → <code>arch</code>, <code>depmixS4</code> →
<code>hmmlearn</code>, R's <code>randomForest</code>/<code>xgboost</code>/<code>keras</code> →
their Python equivalents with the same hyperparameters. Feature scaling uses sample
standard deviation (ddof=1) to match R's <code>sd()</code> exactly.<br><br>
<b>4. Live-only additions.</b> Value-at-Risk / Expected Shortfall are not in the thesis — added
for daily use. FRED's VIX print can lag a day; this app patches the most recent point from
Yahoo when needed. The Geopolitical Risk Index (Caldara & Iacoviello) is fetched live from its
source and neutral-filled if temporarily unreachable, so one obscure external feed can't take
the whole dashboard down.
</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Full feature set (23)</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Feature": engine.feature_cols}), width='stretch', hide_index=True)

    st.caption("Not investment advice. This is a research/educational project — trade at your own risk.")
