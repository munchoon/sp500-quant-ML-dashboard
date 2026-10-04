# S&P 500 Directional Forecast — MS-GJR-GARCH + ML Ensemble

A live, daily-refreshing web dashboard built from the methodology in my FYP thesis:

> **A Hybrid MS-GJR-GARCH and Machine Learning Ensemble Framework for Directional
> Financial Forecasting: Evidence from S&P 500**
> Chin Mun Choon · Universiti Malaya · SIT3025 Statistical Science Project (2026)
> Supervisor: Dr. Tan Shay Kee

The thesis pipeline was built in R (`rugarch`, `depmixS4`, `randomForest`, `xgboost`,
`keras`). This repo is a from-scratch Python port — every feature formula, model
hyperparameter, and evaluation metric is reverse-engineered directly from the original
`.Rmd` source so the live signal is methodologically faithful to the paper, not a
loose reinterpretation of it. Where the live app deliberately does something
differently, it's disclosed in [Notes on fidelity](#notes-on-fidelity) below and in
the app's own "About" tab — never a silent change.

**[Read the "Notes on fidelity" section before trusting a specific number this app
shows you against the thesis's headline 262.95% / 1.268 Sharpe figures.]**

---

## What it does

1. **Econometric layer** — a 2-state Hidden Markov Model classifies each trading day
   as **Calm** or **Crisis**; a **GJR-GARCH(1,1)** is fit separately within each regime
   to capture the asymmetric volatility "leverage effect."
2. **ML ensemble layer** — regime probability + 22 macro/technical/volatility features
   + 3 return lags feed a **Random Forest**, **XGBoost**, and **LSTM** (21-day lookback).
   Their regime-scaled predictions combine via Sharpe-maximised weights into one
   long/short signal, backtested with a 5 bps transaction cost per position change.
3. **Live dashboard** — today's signal, regime read, risk metrics (VaR/ES — new,
   not in the thesis), interactive charts, and SHAP-based explainability, refreshed
   from free data sources on every page load.

## Project structure

```
config.py            Single source of truth: tickers, FRED series, feature list,
                      hyperparameters — every constant traceable to a Rmd chunk.
data_fetcher.py       Data pipeline: FRED / Yahoo Finance / GPR index fetch,
                      every feature transform, live vs. training feature-row builders.
regime_model.py        HMM regime detection + regime-conditional GJR-GARCH (arch/hmmlearn).
ml_models.py            Random Forest / XGBoost / LSTM training, regime-conditional
                      signal scaling, ensemble weight search.
risk_metrics.py        Sharpe/MDD/Calmar (ported from the Rmd) + VaR/ES (new).
train_models.py        OFFLINE orchestration — run this yourself, not on a server.
model_engine.py         INFERENCE — loads trained artifacts, generates today's signal.
                      Never re-fits anything; every page load is just .predict().
app.py                  The Streamlit dashboard.
artifacts/              Trained model files (created by train_models.py).
```

**Why training and inference are separate scripts:** an LSTM plus a 36-configuration
XGBoost grid search takes 20–45 minutes on a laptop CPU. Streamlit Community Cloud
and Hugging Face Spaces' free tiers cannot do that on every page load without timing
out. `train_models.py` does the heavy lifting *once* and writes everything the app
needs into `/artifacts`; `app.py` only ever loads those artifacts and calls
`.predict()`, which is why the dashboard stays fast.

---

## Quick start (local)

```bash
git clone <your-repo-url>
cd sp500-quant-dashboard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Get a free FRED API key** (instant approval, ~2 minutes):
https://fred.stlouisfed.org/docs/api/api_key.html

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then paste your key into .streamlit/secrets.toml
```

**Train once:**

```bash
python train_models.py --quick     # ~1-2 min smoke test — confirms everything runs
python train_models.py             # ~20-45 min full run — this is what the app actually uses
```

`--quick` uses tiny grids/epoch counts purely to verify the pipeline runs end to end;
its predictions are not meaningful. Always follow it with a full run before using the
dashboard for real.

**Run the dashboard:**

```bash
streamlit run app.py
```

---

## Deployment (free hosting)

All three options below load pre-trained artifacts from `/artifacts` — none of them
train on the server. **Train locally first, then commit `/artifacts` to your repo**
(it's ~20 MB total: a 20 MB Random Forest, <1 MB LSTM, ~2 KB XGBoost JSON, ~200 KB HMM,
~150 KB backtest cache — well within normal git limits, no LFS needed).

### Recommended: Hugging Face Spaces

TensorFlow (for the LSTM) is a genuinely heavy dependency. HF Spaces' free CPU tier
gives **16 GB RAM / 2 vCPU**, versus Streamlit Community Cloud's roughly 1 GB — this
is the difference between "just works" and "occasionally OOMs on cold start."

1. Create a Space at huggingface.co/new-space → SDK: **Streamlit**.
2. Push this repo to the Space's git remote (`git push hf main`).
3. Space Settings → **Repository secrets** → add `FRED_API_KEY`.
4. Done — it builds from `requirements.txt` automatically.

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. share.streamlit.io → New app → point at your repo, main file `app.py`.
3. App Settings → **Secrets** → paste:
   ```toml
   FRED_API_KEY = "your-key-here"
   ```
4. Deploy. **If you hit memory errors on cold start**, this is almost always
   TensorFlow — switch to Hugging Face Spaces, or see the "Trimming memory" note below.

### Render

1. New → Web Service → connect the repo.
2. Build command: `pip install -r requirements.txt`
   Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Environment → add `FRED_API_KEY`.
4. Render's free tier sleeps after inactivity — the first request after a sleep will
   be slow (cold start + artifact load), which is normal.

### Keeping the signal fresh

The app itself re-fetches live market data and re-runs inference on every page load
(cached for 15 minutes via `st.cache_data`, with a manual "Refresh now" button) — so
**the signal is always current without any scheduled job.** What does *not* update
automatically is the trained model itself (RF/XGBoost/LSTM weights, ensemble weights).
Re-run `python train_models.py --refresh-data` locally every 1–4 weeks and re-commit
`/artifacts` to keep the models current with recent market structure. A GitHub Actions
workflow that does this on a cron schedule and opens a PR with updated artifacts is a
natural next step if you want this fully automated.

---

## Notes on fidelity

Every number in this app should be traceable to either the Rmd or an explicitly
disclosed change. The live pipeline differs from the notebook in exactly these ways:

1. **Ensemble weights are fit on a validation split, not the test set.** The
   notebook's grid search maximises Sharpe directly on the OOS test window — the code
   comment claims "training-set Sharpe maximisation," but it actually operates on
   `pred_rf_aligned` / `pred_xgb_aligned` / `pred_lstm_regime`, all sliced from
   `test_ml`. That fits the ensemble weights to the exact data later used to report
   performance. `train_models.py` instead carves a validation block out of the
   *training* period (`config.VALIDATION_FRACTION_OF_TRAIN`) and searches weights
   there, so the test-period metrics this app reports are genuinely out-of-sample.
   This is the main reason the live Sharpe/return numbers won't exactly match the
   thesis's 262.95% net return / 1.268 Sharpe — and why they shouldn't.
2. **Newer data vintage.** Markets since the thesis was written are now part of the
   training history; Yahoo/FRED also occasionally revise historical prints.
3. **R → Python port.** `rugarch` → `arch` (GJR-GARCH via the `o=1` asymmetry term),
   `depmixS4` → `hmmlearn` (2-state Gaussian HMM), R's `randomForest` /  `xgboost` /
   `keras` → their Python equivalents with the *same* hyperparameters
   (`ntree=1000, mtry∈{tuned}, nodesize=5`; the same 36-config XGBoost grid; the same
   LSTM architecture, dropout rates, and callback settings). Feature scaling uses
   **sample standard deviation (ddof=1)** specifically to match R's `sd()` — NumPy/
   sklearn default to population std (ddof=0), which would introduce a small,
   avoidable discrepancy.
4. **Forward-looking live inference.** A subtle one: a training-style feature matrix
   requires a known `Target` (today's actual return) for every row, which means its
   *last* row necessarily describes a session that's already closed — useful for
   validation, useless as a live signal. `data_fetcher.build_live_feature_row()`
   instead appends a one-day-ahead placeholder and reuses the exact same lag-1
   machinery, so the live signal is genuinely for the *next* session, not a
   backward-looking echo of the most recent one.
5. **Live-only additions not in the thesis:** Value-at-Risk / Expected Shortfall
   (historical + GARCH-parametric), a live SHAP explanation of *today's* specific
   prediction, and a same-day VIX patch from Yahoo Finance for days FRED's `VIXCLS`
   hasn't posted yet (training always uses FRED exclusively).
6. **One (correct, not a bug) quirk worth knowing:** "LSTM" and "LSTM + Regime" show
   identical standalone Sharpe/return in the model-comparison table. Regime-scaling
   only rescales the LSTM's predicted *magnitude* — it amplifies or dampens, but by a
   positive factor either way, so it can never flip the sign of the prediction. Since
   the trading signal is `sign(prediction)`, a standalone backtest of the regime-scaled
   series is mathematically guaranteed to match the unscaled series. The scaling only
   changes anything once it's blended into the weighted ensemble, where relative
   *magnitude* (not just sign) determines the combined signal.

## Disclaimer

This is a research and educational project, not investment advice. Past out-of-sample
backtest performance, however carefully computed, is not a guarantee of future
results. Trade at your own risk.
