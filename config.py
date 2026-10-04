"""
config.py
─────────
Single source of truth for tickers, data series, feature definitions, and
hyperparameters. Every constant here was reverse-engineered directly from
SIT3020_FYP_Latest.Rmd (Chin Mun Choon, Universiti Malaya, SIT3025 FYP) so
that the live Python pipeline is methodologically faithful to the thesis:

    "A Hybrid MS-GJR-GARCH and Machine Learning Ensemble Framework for
    Directional Financial Forecasting: Evidence from S&P 500"

Where the live app deliberately DEVIATES from the notebook (documented
inline below, search "DEVIATION"), it is a conscious fix for something that
would otherwise make the live signal untrustworthy — never a silent change.
"""
from __future__ import annotations
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# Secrets — never hardcode API keys in source. Read from env var first
# (local / GitHub Actions), then fall back to Streamlit secrets when running
# inside app.py. The Rmd had a personal FRED key hardcoded in-line; that is
# intentionally NOT carried over here.
# ─────────────────────────────────────────────────────────────────────────
def get_fred_api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # local import: config.py must stay importable without streamlit
        return st.secrets["FRED_API_KEY"]
    except Exception:
        raise RuntimeError(
            "FRED_API_KEY not found. Set it as an environment variable, a "
            "GitHub Actions secret, or in .streamlit/secrets.toml. Get a "
            "free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

# ─────────────────────────────────────────────────────────────────────────
# Market data tickers (yfinance)
# ─────────────────────────────────────────────────────────────────────────
SP500_TICKER = "^GSPC"     # matches quantmod::getSymbols("^GSPC") in the Rmd
VIX_TICKER_YF = "^VIX"     # used only to patch the most recent day if FRED
                            # hasn't published VIXCLS yet (FRED usually lags
                            # by one business day) — DEVIATION for liveness

# ─────────────────────────────────────────────────────────────────────────
# FRED series — exact IDs from the Rmd's `series_list` / `fred_series`
# ─────────────────────────────────────────────────────────────────────────
FRED_SERIES = {
    "CPIAUCSL":   "CPI (headline, SA)",
    "UNRATE":     "Civilian unemployment rate",
    "INDPRO":     "Industrial production index",
    "DFF":        "Effective federal funds rate (daily)",
    "VIXCLS":     "CBOE VIX (close)",
    "DCOILWTICO": "WTI crude oil, $/bbl",
    "BAA10Y":     "Moody's Baa corporate bond yield minus 10Y Treasury",
    "T10Y2Y":     "10Y minus 2Y Treasury constant maturity spread",
}
# USD index is spliced from two discontinued/succeeding FRED series, exactly
# as in the Rmd: DTWEXM covers 1995-2019, DTWEXAFEGS covers 2020-present.
USD_SERIES_EARLY = "DTWEXM"
USD_SERIES_LATE = "DTWEXAFEGS"
USD_SPLICE_DATE = "2020-01-01"

RECESSION_SERIES = "USREC"  # NBER recession indicator, used only for
                             # regime validation display, not as a feature

# ─────────────────────────────────────────────────────────────────────────
# Geopolitical Risk Index (Caldara & Iacoviello, 2022)
# Source exactly as in the Rmd's httr::GET call.
# ─────────────────────────────────────────────────────────────────────────
GPR_URLS = [
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xlsx",
]

# ─────────────────────────────────────────────────────────────────────────
# Feature engineering constants (exact windows from the Rmd)
# ─────────────────────────────────────────────────────────────────────────
MOM_1Y_WINDOW = 252      # ROC(SP500, n = 252, type = "discrete") * 100
MOM_1M_WINDOW = 21       # ROC(SP500, n = 21,  type = "discrete") * 100
RSI_WINDOW = 14          # TTR::RSI(SP500, n = 14) — Wilder smoothing
SMA_WINDOW = 200         # (SP500 / SMA(SP500, 200)) - 1
IVOL_WINDOW = 21         # rolling residual-std of a 4-factor OLS
REALVOL_5D_WINDOW = 5    # rolling 5-day std of returns

# IVOL regression: SP500_ret ~ USD_ret + Oil_ret + VIX_chg + FedRate_chg
IVOL_REGRESSORS = ["USD_ret", "Oil_ret", "VIX_chg", "FedRate_chg"]

# ─────────────────────────────────────────────────────────────────────────
# The 23-feature set exactly as documented in Thesis Table 3.1.
# (22 of these were used in the notebook's main RF/XGB/LSTM run; GPR_lag1
# was validated as a robustness addition later in the same notebook and is
# part of the officially documented 23-feature specification — see README
# "Notes on fidelity" for the full paper trail.)
# ─────────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    # Macroeconomic & intermarket (9)
    "Oil_ret_lag1", "USD_ret_lag1", "FedRate_chg_lag1", "Term_chg_lag1",
    "Credit_chg_lag1", "InflMom_lag1", "IP_growth_lag1", "Unemp_lag1",
    "GPR_lag1",
    # Technical & momentum (6)
    "Mom_1Y_lag1", "Mom_1M_lag1", "RSI_lag1", "SMA_200_Dist_lag1",
    "VIX_lag1", "VIX_chg_lag1",
    # Volatility (3)
    "IVOL_lag1", "RealVol_lag1", "RealVol_5d_lag1",
    # Market regime (2)
    "Regime_prob_lag1", "Regime_bin_lag1",
    # Autoregressive (3)
    "Ret_lag1", "Ret_lag2", "Ret_lag5",
]
assert len(FEATURE_COLUMNS) == 23

TARGET_COLUMN = "Target"          # SP500_ret, same day (Rmd: Target = SP500_ret)
TRANSACTION_COST = 0.0005          # 5 bps per position change (Rmd: COST <- 0.0005)
TRADING_DAYS_PER_YEAR = 252

# ─────────────────────────────────────────────────────────────────────────
# Train / validation / test split
# Rmd used a straight chronological 60/40 split (train_end <- floor(0.6*n)).
# DEVIATION: the live pipeline carves the last 20% of the *training* window
# out as a validation block used only for ensemble-weight selection and
# early stopping, so nothing about the reported test-period numbers is
# tuned on the test period itself. See ml_models.optimize_ensemble_weights().
# ─────────────────────────────────────────────────────────────────────────
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION_OF_TRAIN = 0.20

DATA_START_DATE = "1995-01-01"
LIVE_LOOKBACK_CALENDAR_DAYS = 1100   # ~3 years: covers 252d momentum + 200d SMA
                                       # warmup with a large margin for the HMM to
                                       # stabilise away from initial-state effects

# ─────────────────────────────────────────────────────────────────────────
# Random Forest (Rmd: randomForest(ntree=1000, mtry=tuned, nodesize=5))
# ─────────────────────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 1000
RF_MIN_SAMPLES_LEAF = 5          # nodesize = 5
RF_MTRY_CANDIDATES_FN = lambda p: sorted(set([  # noqa: E731
    max(1, p // 3), max(1, int(p ** 0.5)), max(1, p // 2)
]))
RF_RANDOM_STATE = 123

# ─────────────────────────────────────────────────────────────────────────
# XGBoost grid (Rmd: expand.grid(max_depth, eta, subsample, colsample))
# ─────────────────────────────────────────────────────────────────────────
XGB_GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
}
XGB_FIXED_PARAMS = dict(
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=1,
    objective="reg:squarederror",
)
XGB_MAX_ROUNDS = 500
XGB_CV_FOLDS = 5
XGB_EARLY_STOPPING_ROUNDS = 30
XGB_RANDOM_STATE = 123

# ─────────────────────────────────────────────────────────────────────────
# LSTM (Rmd: Input -> LSTM(64,seq) -> Drop(.3) -> LSTM(32) -> Drop(.2)
#            -> Dense(16,relu) -> Drop(.1) -> Dense(1,linear))
# ─────────────────────────────────────────────────────────────────────────
LSTM_SEQ_LEN = 21
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
LSTM_DENSE_UNITS = 16
LSTM_DROPOUT_1 = 0.3
LSTM_DROPOUT_2 = 0.2
LSTM_DROPOUT_3 = 0.1
LSTM_EPOCHS = 200
LSTM_BATCH_SIZE = 64
LSTM_VALIDATION_SPLIT = 0.15
LSTM_EARLY_STOP_PATIENCE = 25
LSTM_REDUCE_LR_PATIENCE = 10
LSTM_REDUCE_LR_FACTOR = 0.5
LSTM_MIN_LR = 1e-6
LSTM_RANDOM_STATE = 123

# ─────────────────────────────────────────────────────────────────────────
# Regime-conditional LSTM signal scaling
# (Rmd: regime_scale(pred, regime_prob, calm=.3, crisis=.7, amp=1.3, pen=.7))
# ─────────────────────────────────────────────────────────────────────────
REGIME_CALM_THRESHOLD = 0.3
REGIME_CRISIS_THRESHOLD = 0.7
REGIME_AMPLIFY_FACTOR = 1.3
REGIME_PENALIZE_FACTOR = 0.7

# ─────────────────────────────────────────────────────────────────────────
# Ensemble weight search (Rmd: nested 10% -> 5% -> 1% grid on w1,w2,w3)
# ─────────────────────────────────────────────────────────────────────────
ENSEMBLE_WEIGHT_STEP = 0.01
# Reference point ONLY — the published thesis figure, shown in the "About"
# tab for comparison. The live app derives its own weights on the
# validation split every time train_models.py runs (see DEVIATION above).
THESIS_REFERENCE_WEIGHTS = {"rf": 0.38, "xgb": 0.04, "lstm": 0.58}
THESIS_REFERENCE_METRICS = {
    "sharpe": 1.268, "net_return_pct": 262.95, "max_drawdown_pct": -28.16,
    "directional_accuracy_pct": 53.8, "test_period": "2014-01-01 to 2025-12-31",
}

# ─────────────────────────────────────────────────────────────────────────
# HMM regime model (Rmd: depmixS4::depmix(SP500_ret ~ 1, nstates=2, gaussian))
# ─────────────────────────────────────────────────────────────────────────
HMM_N_STATES = 2
HMM_RANDOM_STATE = 123
HMM_N_ITER = 500

# GJR-GARCH, fit once on the full sample and once per-regime
# (Rmd: ugarchspec(model="gjrGARCH", garchOrder=c(1,1)), dist "std")
GARCH_P, GARCH_O, GARCH_Q = 1, 1, 1
GARCH_DIST = "t"  # Student-t, matches distribution.model = "std" in rugarch

# ─────────────────────────────────────────────────────────────────────────
# Risk metrics (NEW — not present in the thesis; added for the live app
# per the user's requirements. Standard, well-defined formulations.)
# ─────────────────────────────────────────────────────────────────────────
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]
VAR_LOOKBACK_DAYS = 252  # trailing 1Y window for historical VaR/ES

# ─────────────────────────────────────────────────────────────────────────
# Artifact filenames
# ─────────────────────────────────────────────────────────────────────────
ARTIFACT_FILES = {
    "scaler": ARTIFACT_DIR / "feature_scaler.joblib",
    # NOTE: HMM regime state AND both regime-conditional + full-sample GJR-GARCH
    # parameter sets are bundled into this single file by RegimeModel.save() —
    # there are deliberately no separate garch_*.joblib artifacts.
    "hmm": ARTIFACT_DIR / "hmm_regime.joblib",
    "rf": ARTIFACT_DIR / "random_forest.joblib",
    "xgb": ARTIFACT_DIR / "xgboost_model.json",
    "lstm": ARTIFACT_DIR / "lstm_model.keras",
    "ensemble_weights": ARTIFACT_DIR / "ensemble_weights.json",
    "feature_importance": ARTIFACT_DIR / "feature_importance.json",
    "backtest_history": ARTIFACT_DIR / "backtest_history.parquet",
    "training_history": ARTIFACT_DIR / "lstm_training_history.json",
    "metadata": ARTIFACT_DIR / "metadata.json",
    "raw_dataset_cache": ARTIFACT_DIR / "daily_dataset_cache.parquet",
}
