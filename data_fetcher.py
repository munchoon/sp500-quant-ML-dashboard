"""
data_fetcher.py
────────────────
Live + historical data pipeline for the S&P 500 MS-GJR-GARCH + ML ensemble.

Two entry points matter to callers:

    build_base_daily_dataset(start, end)
        Full macro + technical + volatility dataset (everything except the
        HMM regime columns, which regime_model.py adds after fitting).
        Used by train_models.py for the full 1996-present history, and by
        model_engine.py for a short recent window on every page load.

    build_lagged_feature_matrix(df)
        Turns the base dataset (with regime columns already attached) into
        the exact 23-column, 1-day-lagged feature matrix the models were
        trained on, plus the same-day `Target`.

Every transform below is a direct Python port of a specific R chunk in
SIT3020_FYP_Latest.Rmd — see the inline "Rmd:" comments for traceability.
Re-using the *same* functions for both training and live inference (rather
than writing a separate "fast path") is deliberate: it eliminates train/serve
skew, which is the most common way a live quant signal silently diverges
from its backtest.
"""
from __future__ import annotations

import io
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred

import config

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

_fred_client: Optional[Fred] = None


def _fred() -> Fred:
    global _fred_client
    if _fred_client is None:
        _fred_client = Fred(api_key=config.get_fred_api_key())
    return _fred_client


# ═════════════════════════════════════════════════════════════════════════
# 1. Raw series fetchers
# ═════════════════════════════════════════════════════════════════════════

def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """Rmd: fredr(series_id, observation_start, observation_end)"""
    s = _fred().get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s


def fetch_all_fred_series(start: str, end: str) -> pd.DataFrame:
    """Rmd: fred_data_list <- lapply(fred_series, fredr(...))"""
    frames = {}
    for series_id in config.FRED_SERIES:
        try:
            frames[series_id] = fetch_fred_series(series_id, start, end)
        except Exception as exc:
            logger.warning("FRED series %s failed: %s", series_id, exc)
    return pd.DataFrame(frames)


def fetch_usd_index(start: str, end: str) -> pd.Series:
    """
    Rmd: splice DTWEXM (1995-2019) with DTWEXAFEGS (2020-2025) because the
    Fed discontinued DTWEXM at end-2019.
    """
    splice = pd.Timestamp(config.USD_SPLICE_DATE)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    if start_ts < splice:
        early_end = min(end_ts, splice - timedelta(days=1))
        try:
            parts.append(fetch_fred_series(config.USD_SERIES_EARLY, start, str(early_end.date())))
        except Exception as exc:
            logger.warning("USD early splice (%s) failed: %s", config.USD_SERIES_EARLY, exc)
    if end_ts >= splice:
        late_start = max(start_ts, splice)
        parts.append(fetch_fred_series(config.USD_SERIES_LATE, str(late_start.date()), end))
    usd = pd.concat(parts).sort_index()
    usd = usd[~usd.index.duplicated(keep="last")]
    usd.name = "USD_INDEX"
    return usd


def fetch_sp500_price(start: str, end: str) -> pd.Series:
    """Rmd: getSymbols('^GSPC', src='yahoo'); Ad(sp500_daily)"""
    df = yf.download(config.SP500_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no S&P 500 data — check network/ticker.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a 1-col frame
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = "SP500"
    return close


def fetch_latest_vix_patch(after: pd.Timestamp) -> Optional[pd.Series]:
    """
    DEVIATION (for liveness only): FRED's VIXCLS typically posts with a
    ~1 business-day lag. For same-day dashboard freshness, patch in the most
    recent close from yfinance's ^VIX for any date after FRED's last point.
    Training always uses FRED VIXCLS exclusively, so backtest fidelity is
    unaffected.
    """
    try:
        df = yf.download(config.VIX_TICKER_YF, start=after - timedelta(days=3),
                          progress=False, auto_adjust=True)
        if df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close[close.index > after]
        close.name = "VIXCLS"
        return close if not close.empty else None
    except Exception as exc:
        logger.warning("Live VIX patch failed: %s", exc)
        return None


def fetch_gpr_index() -> pd.DataFrame:
    """
    Rmd: httr::GET(gpr_url) -> readxl::read_excel(sheet='Sheet1')
    Source: Caldara & Iacoviello (2022), matteoiacoviello.com/gpr.htm
    Returns a monthly DataFrame with columns Date, GPR, GPRA, GPRH.
    Degrades gracefully (empty frame) if the site is unreachable — GPR is a
    single feature out of 23 and should never take the whole app down.
    """
    for url in config.GPR_URLS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            engine = "xlrd" if url.endswith(".xls") else "openpyxl"
            raw = pd.read_excel(io.BytesIO(resp.content), sheet_name="Sheet1", engine=engine)
            raw.columns = [str(c).strip() for c in raw.columns]

            if "DATE" in raw.columns:
                date = pd.to_datetime(raw["DATE"])
            elif {"year", "month"}.issubset(raw.columns):
                date = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=1))
            else:
                raise ValueError(f"Unrecognised GPR column layout: {list(raw.columns)}")

            out = pd.DataFrame({"Date": date})
            for col in ["GPR", "GPRA", "GPRH"]:
                out[col] = pd.to_numeric(raw.get(col), errors="coerce")
            out = out.dropna(subset=["GPR"]).sort_values("Date").reset_index(drop=True)
            logger.info("GPR fetched: %d monthly rows from %s to %s",
                        len(out), out.Date.min().date(), out.Date.max().date())
            return out
        except Exception as exc:
            logger.warning("GPR fetch from %s failed: %s", url, exc)
    logger.warning("All GPR sources failed — GPR_lag1 will be forward-filled as NaN/0.")
    return pd.DataFrame(columns=["Date", "GPR", "GPRA", "GPRH"])


def fetch_recession_dummy(start: str, end: str) -> pd.Series:
    """Rmd: fredr('USREC') — used only for regime-vs-NBER validation display."""
    try:
        return fetch_fred_series(config.RECESSION_SERIES, start, end)
    except Exception as exc:
        logger.warning("USREC fetch failed: %s", exc)
        return pd.Series(dtype=float, name="USREC")


# ═════════════════════════════════════════════════════════════════════════
# 2. Technical indicators (Rmd chunk: "1. Price Momentum ... 4. Moving
#    Average Crossover")
# ═════════════════════════════════════════════════════════════════════════

def roc(price: pd.Series, n: int) -> pd.Series:
    """Rmd: TTR::ROC(price, n, type='discrete') * 100 -> (P_t/P_{t-n} - 1)*100"""
    return (price / price.shift(n) - 1.0) * 100.0


def wilder_rsi(price: pd.Series, n: int = config.RSI_WINDOW) -> pd.Series:
    """Rmd: TTR::RSI(price, n=14) — classic Wilder smoothing (EMA, alpha=1/n)."""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss != 0, 100.0)  # no losses in window -> RSI = 100


def sma_distance(price: pd.Series, n: int = config.SMA_WINDOW) -> pd.Series:
    """Rmd: (SP500 / SMA(SP500, n=200)) - 1"""
    sma = price.rolling(n).mean()
    return (price / sma) - 1.0


def rolling_ivol(df: pd.DataFrame, window: int = config.IVOL_WINDOW) -> pd.Series:
    """
    Rmd: rollapplyr(daily_data, width=21, FUN=calc_ivol) where
         calc_ivol regresses SP500_ret ~ USD_ret + Oil_ret + VIX_chg + FedRate_chg
         and returns sd(residuals).
    Implemented with a rolling OLS via numpy.lstsq for speed over ~7,500 rows.
    """
    y_all = df["SP500_ret"].to_numpy()
    X_all = df[config.IVOL_REGRESSORS].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    ones = np.ones((window, 1))
    for i in range(window - 1, n):
        lo = i - window + 1
        y_win = y_all[lo:i + 1]
        X_win = X_all[lo:i + 1]
        if np.isnan(y_win).any() or np.isnan(X_win).any():
            continue
        design = np.hstack([ones, X_win])
        beta, *_ = np.linalg.lstsq(design, y_win, rcond=None)
        resid = y_win - design @ beta
        out[i] = resid.std(ddof=1)
    return pd.Series(out, index=df.index, name="IVOL")


def fetch_sp500_ohlc(start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Full OHLC (not just Close) for the candlestick chart. Kept separate from
    fetch_sp500_price(), which the modeling pipeline uses — the models only
    ever need the Close-derived return series, so there's no reason for the
    heavier OHLC frame to flow through feature engineering.
    """
    df = yf.download(config.SP500_TICKER, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned no S&P 500 OHLC data.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


# ═════════════════════════════════════════════════════════════════════════
# 3. Master pipeline
# ═════════════════════════════════════════════════════════════════════════

def build_base_daily_dataset(start: str = config.DATA_START_DATE,
                              end: Optional[str] = None,
                              patch_live_vix: bool = False) -> pd.DataFrame:
    """
    Reproduces the Rmd's "Build DAILY dataset" chunk plus the momentum/RSI/
    SMA/IVOL chunks. Returns a daily-indexed DataFrame on trading days only,
    with everything needed for feature construction EXCEPT the two HMM
    regime columns (added afterwards by regime_model.add_regime_features).
    """
    end = end or datetime.today().strftime("%Y-%m-%d")

    sp500 = fetch_sp500_price(start, end)
    fred_df = fetch_all_fred_series(start, end)
    usd = fetch_usd_index(start, end)

    if patch_live_vix and "VIXCLS" in fred_df.columns and not fred_df["VIXCLS"].empty:
        patch = fetch_latest_vix_patch(fred_df["VIXCLS"].dropna().index.max())
        if patch is not None:
            fred_df = fred_df.combine_first(patch.to_frame())

    # ── Merge onto a daily calendar and forward-fill (Rmd: na.locf) ──────
    daily = pd.concat([sp500, fred_df, usd], axis=1).sort_index()
    daily = daily.ffill().dropna(how="all")

    # Rmd: daily_xts$DCOILWTICO <- pmax(daily_xts$DCOILWTICO, 0.01)
    daily["DCOILWTICO"] = daily["DCOILWTICO"].clip(lower=0.01)

    # ── Trading days only (Rmd: index(daily_data) %in% trading_days) ─────
    daily = daily.loc[daily.index.isin(sp500.index)].copy()
    daily = daily.dropna(subset=["SP500"] + list(config.FRED_SERIES.keys()) + ["USD_INDEX"])

    # ── Base return / change transforms (Rmd "Add returns and features") ─
    daily["SP500_ret"] = 100 * np.log(daily["SP500"]).diff()
    daily["Oil_ret"] = 100 * np.log(daily["DCOILWTICO"]).diff()
    daily["USD_ret"] = 100 * np.log(daily["USD_INDEX"]).diff()
    daily["InflMom"] = 100 * np.log(daily["CPIAUCSL"]).diff() * 365
    daily["IP_growth"] = 100 * np.log(daily["INDPRO"]).diff() * 365
    daily["VIX_chg"] = daily["VIXCLS"].diff()
    daily["FedRate_chg"] = daily["DFF"].diff()
    daily["Term_chg"] = daily["T10Y2Y"].diff()
    daily["Credit_chg"] = daily["BAA10Y"].diff()
    daily["Unemp"] = daily["UNRATE"].diff()
    daily = daily.dropna(subset=["SP500_ret", "Oil_ret", "USD_ret"])

    # ── Recession dummy (validation/display only) ────────────────────────
    try:
        usrec = fetch_recession_dummy(start, end)
        daily["Recession"] = usrec.reindex(daily.index, method="ffill").fillna(0)
    except Exception:
        daily["Recession"] = 0.0

    # ── Technical indicators ──────────────────────────────────────────────
    daily["Mom_1Y"] = roc(daily["SP500"], config.MOM_1Y_WINDOW)
    daily["Mom_1M"] = roc(daily["SP500"], config.MOM_1M_WINDOW)
    daily["RSI"] = wilder_rsi(daily["SP500"], config.RSI_WINDOW)
    daily["SMA_200_Dist"] = sma_distance(daily["SP500"], config.SMA_WINDOW)
    daily = daily.dropna(subset=["Mom_1Y", "Mom_1M", "RSI", "SMA_200_Dist"])

    # ── IVOL (rolling 21-day residual std) ────────────────────────────────
    daily["IVOL"] = rolling_ivol(daily, config.IVOL_WINDOW)

    # ── Realised volatility features ──────────────────────────────────────
    daily["RealVol"] = daily["SP500_ret"].abs()
    daily["RealVol_5d"] = daily["SP500_ret"].rolling(config.REALVOL_5D_WINDOW).std(ddof=1)

    # ── Geopolitical Risk Index (monthly -> daily LOCF) ───────────────────
    gpr_monthly = fetch_gpr_index()
    if not gpr_monthly.empty:
        gpr_daily = gpr_monthly.set_index("Date")[["GPR", "GPRA", "GPRH"]]
        gpr_daily = gpr_daily.reindex(
            pd.date_range(gpr_daily.index.min(), daily.index.max(), freq="D")
        ).ffill()
        for col in ["GPR", "GPRA", "GPRH"]:
            daily[col] = gpr_daily[col].reindex(daily.index, method="ffill")
        # A handful of leading days before the GPR series' own start (or a
        # brief re-fetch gap) can still be NaN — neutral-fill rather than
        # let one feature's edge case cascade into dropping otherwise-good
        # rows downstream.
        for col in ["GPR", "GPRA", "GPRH"]:
            if daily[col].isna().any():
                daily[col] = daily[col].fillna(daily[col].median())
    else:
        # GPR source unreachable this run: neutral-fill rather than NaN, so
        # a single external data source outage can't cascade into
        # build_lagged_feature_matrix() dropping every row via dropna().
        logger.warning(
            "GPR unavailable this run — GPR_lag1 filled with 0 (neutral). "
            "All other features are unaffected."
        )
        for col in ["GPR", "GPRA", "GPRH"]:
            daily[col] = 0.0

    daily.index.name = "Date"
    return daily


def add_regime_features(df: pd.DataFrame, prob_crisis: pd.Series, is_crisis: pd.Series) -> pd.DataFrame:
    """Attach HMM-derived regime columns computed by regime_model.py."""
    out = df.copy()
    out["Regime_prob_crisis"] = prob_crisis.reindex(out.index)
    out["Regime_binary"] = is_crisis.reindex(out.index)
    return out


def build_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies every _lag1 shift + Ret_lag2/Ret_lag5 + Target, WITHOUT dropping
    any rows. Split out from build_lagged_feature_matrix() so the exact same
    lag machinery can serve two different callers:
      - training (drops rows where Target/features are NaN)
      - live inference (keeps the newest row even though its Target, i.e.
        tomorrow's actual return, is by definition unknown)
    """
    out = df.copy()
    out["VIX_lag1"] = out["VIXCLS"].shift(1)
    out["VIX_chg_lag1"] = out["VIX_chg"].shift(1)
    out["Oil_ret_lag1"] = out["Oil_ret"].shift(1)
    out["USD_ret_lag1"] = out["USD_ret"].shift(1)
    out["FedRate_chg_lag1"] = out["FedRate_chg"].shift(1)
    out["Term_chg_lag1"] = out["Term_chg"].shift(1)
    out["Credit_chg_lag1"] = out["Credit_chg"].shift(1)
    out["InflMom_lag1"] = out["InflMom"].shift(1)
    out["IP_growth_lag1"] = out["IP_growth"].shift(1)
    out["Unemp_lag1"] = out["Unemp"].shift(1)
    out["Mom_1Y_lag1"] = out["Mom_1Y"].shift(1)
    out["Mom_1M_lag1"] = out["Mom_1M"].shift(1)
    out["RSI_lag1"] = out["RSI"].shift(1)
    out["SMA_200_Dist_lag1"] = out["SMA_200_Dist"].shift(1)
    out["IVOL_lag1"] = out["IVOL"].shift(1)
    out["RealVol_lag1"] = out["RealVol"].shift(1)
    out["RealVol_5d_lag1"] = out["RealVol_5d"].shift(1)
    out["Regime_prob_lag1"] = out["Regime_prob_crisis"].shift(1)
    out["Regime_bin_lag1"] = out["Regime_binary"].shift(1)
    out["Ret_lag1"] = out["SP500_ret"].shift(1)
    out["Ret_lag2"] = out["SP500_ret"].shift(2)
    out["Ret_lag5"] = out["SP500_ret"].shift(5)
    out["GPR_lag1"] = out["GPR"].shift(1)

    out["Target"] = out["SP500_ret"]
    out["Direction"] = (out["SP500_ret"] > 0).astype(int)
    return out


def build_lagged_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Rmd: `ml_df <- daily_df %>% mutate(..._lag1 = dplyr::lag(..., 1), ...)`
    Training-time feature matrix: the exact 23-column, 1-day-lagged feature
    set + Target, with any row missing a feature or a Target dropped.
    `df` must already contain Regime_prob_crisis / Regime_binary.
    """
    out = build_lagged_features(df)
    feature_cols = [c for c in config.FEATURE_COLUMNS if c in out.columns]
    required = feature_cols + ["Target"]
    ml_df = out.dropna(subset=required).copy()
    return ml_df, feature_cols


def _extend_with_placeholder_day(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Append one empty future row so shift(1) turns today's now-complete
    values into that row's `_lag1` features — the trick that makes live
    inference reuse the exact same lag machinery as training."""
    next_date = df.index.max() + pd.tseries.offsets.BDay(1)
    placeholder = pd.DataFrame(index=[next_date], columns=df.columns, dtype=float)
    return pd.concat([df, placeholder]), next_date


def build_live_feature_row(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """
    The live, FORWARD-LOOKING counterpart to build_lagged_feature_matrix():
    returns the feature vector for predicting the NEXT trading session,
    built from the most recent complete day's data. `df` must already
    contain Regime_prob_crisis / Regime_binary (from RegimeModel.score()).
    """
    extended, next_date = _extend_with_placeholder_day(df)
    lagged = build_lagged_features(extended)
    row = lagged.loc[next_date, feature_cols]
    if row.isna().any():
        missing = row[row.isna()].index.tolist()
        raise RuntimeError(
            f"Live feature row incomplete — missing {missing}. Fetch a longer "
            f"lookback window (need >{config.SMA_WINDOW} trading days of clean history)."
        )
    return row


def build_live_lstm_sequence(df: pd.DataFrame, feature_cols: list[str],
                              seq_len: int = config.LSTM_SEQ_LEN) -> pd.DataFrame:
    """Same forward-looking trick as build_live_feature_row(), but returns
    the trailing `seq_len`-row window the LSTM needs (unscaled; the caller
    applies the saved FeatureScaler)."""
    extended, next_date = _extend_with_placeholder_day(df)
    lagged = build_lagged_features(extended)
    window = lagged[feature_cols].tail(seq_len)
    if len(window) < seq_len or window.isna().any().any():
        raise RuntimeError(
            f"Not enough recent history for a full {seq_len}-day LSTM sequence — "
            f"fetch a longer live lookback window."
        )
    return window


def get_latest_complete_row(ml_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """
    The most recent row of the TRAINING-style matrix with a known Target —
    i.e. "what would the model have called for the most recently completed
    session, using the prior day's data". Useful for a same-day sanity
    check, but NOT the forward-looking live signal — use
    build_live_feature_row() for that.
    """
    complete = ml_df.dropna(subset=feature_cols)
    if complete.empty:
        raise RuntimeError("No complete feature row available — check upstream data fetch.")
    return complete.iloc[-1]
