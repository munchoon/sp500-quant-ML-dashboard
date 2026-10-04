"""
train_models.py
─────────────────
Offline / periodic training pipeline. Run this locally, or on a schedule via
GitHub Actions — NOT on every Streamlit page load. Streamlit Community Cloud
and HF Spaces' free tiers cannot train an LSTM + a 36-configuration XGBoost
grid search on demand without timing out; this script does that heavy work
once and writes everything app.py / model_engine.py need into /artifacts.

    python train_models.py                  # full run (~20-45 min on a laptop CPU)
    python train_models.py --quick           # tiny smoke test (~1-2 min, do this first)
    python train_models.py --refresh-data    # bypass the cached raw dataset

Methodology note: the ensemble-weight search and the Random Forest's
permutation feature importance are computed on a VALIDATION slice carved
out of the *training* period — never on the test period. The original Rmd
selects ensemble weights by grid-searching Sharpe directly on the OOS test
set, which quietly fits the weights to the exact data later used to report
performance. See config.py's "DEVIATION" comment and the docstring on
ml_models.optimize_ensemble_weights() for the full explanation. Because of
this (deliberate, disclosed) fix, and because live market data differs from
the frozen 2025 vintage in the thesis, the numbers this script reports will
not exactly reproduce the thesis's 262.95% / 1.268 Sharpe figures — the
"About" tab in the dashboard shows both side by side.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

import config
import data_fetcher
import ml_models
import regime_model
import risk_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models")


def get_or_fetch_base_dataset(refresh: bool) -> pd.DataFrame:
    cache_path = config.ARTIFACT_FILES["raw_dataset_cache"]
    if not refresh and cache_path.exists():
        logger.info("Loading cached raw dataset from %s (pass --refresh-data to refetch).", cache_path)
        return pd.read_parquet(cache_path)
    logger.info("Fetching full history from FRED / Yahoo Finance / GPR — this takes a few minutes...")
    base_df = data_fetcher.build_base_daily_dataset(start=config.DATA_START_DATE)
    base_df.to_parquet(cache_path)
    logger.info("Fetched %d rows (%s to %s); cached to %s",
                len(base_df), base_df.index.min().date(), base_df.index.max().date(), cache_path)
    return base_df


def chronological_splits(ml_df: pd.DataFrame):
    n = len(ml_df)
    n_trainval = int(np.floor(config.TRAIN_FRACTION * n))
    n_val = int(np.floor(config.VALIDATION_FRACTION_OF_TRAIN * n_trainval))
    n_train = n_trainval - n_val

    train, val, test = ml_df.iloc[:n_train], ml_df.iloc[n_train:n_trainval], ml_df.iloc[n_trainval:]
    logger.info("Split -> train: %d rows (%s→%s) | val: %d rows (%s→%s) | test: %d rows (%s→%s)",
                len(train), train.index.min().date(), train.index.max().date(),
                len(val), val.index.min().date(), val.index.max().date(),
                len(test), test.index.min().date(), test.index.max().date())
    return train, val, test


def _align_to_lstm(*arrays, seq_len: int = config.LSTM_SEQ_LEN):
    """RF/XGBoost predict on every row; the LSTM loses its first `seq_len`
    rows to sequence formation (Rmd: tail(pred_rf, n_lstm)). Trim everything
    else down to match before combining into the ensemble."""
    return tuple(a[seq_len:] for a in arrays)


def evaluate_and_log(name: str, pred: np.ndarray, actual: np.ndarray) -> dict:
    metrics, net_ret, signal = risk_metrics.evaluate_strategy(pred, actual)
    d = metrics.__dict__.copy()
    logger.info("  %-16s Sharpe=%6.3f  NetReturn=%8.2f%%  MaxDD=%7.2f%%  DirAcc=%5.1f%%",
                name, d["annualised_sharpe"], d["total_net_return_pct"],
                d["max_drawdown_pct"], d["directional_accuracy_pct"])
    return d, net_ret, signal


def main(quick: bool, refresh_data: bool):
    t0 = datetime.now(timezone.utc)
    logger.info("═" * 70)
    logger.info("Training run started %s | quick_mode=%s", t0.isoformat(timespec="seconds"), quick)
    logger.info("═" * 70)

    # ── 1. Data ─────────────────────────────────────────────────────────
    base_df = get_or_fetch_base_dataset(refresh_data)
    returns = base_df["SP500_ret"]

    # ── 2. Regime model ────────────────────────────────────────────────
    logger.info("Fitting HMM regime model + regime-conditional GJR-GARCH...")
    rmodel = regime_model.RegimeModel()
    rmodel.fit(returns, recession=base_df.get("Recession"))
    prob_crisis, is_crisis = rmodel.historical_regime_columns()
    logger.info("Regime split: calm=%.1f%% (mean=%.3f%%, vol=%.3f%%) | "
                "crisis=%.1f%% (mean=%.3f%%, vol=%.3f%%) | NBER match=%s",
                rmodel.regime_stats.pct_of_sample["calm"], rmodel.regime_stats.mean_return["calm"],
                rmodel.regime_stats.volatility["calm"], rmodel.regime_stats.pct_of_sample["crisis"],
                rmodel.regime_stats.mean_return["crisis"], rmodel.regime_stats.volatility["crisis"],
                f"{rmodel.nber_match_rate:.1f}%" if rmodel.nber_match_rate is not None else "n/a")

    df_with_regime = data_fetcher.add_regime_features(base_df, prob_crisis, is_crisis)
    ml_df, feature_cols = data_fetcher.build_lagged_feature_matrix(df_with_regime)
    logger.info("Feature matrix: %d rows x %d features (%s → %s)",
                len(ml_df), len(feature_cols), ml_df.index.min().date(), ml_df.index.max().date())

    # ── 3. Splits + scaling ────────────────────────────────────────────
    train, val, test = chronological_splits(ml_df)
    scaler = ml_models.FeatureScaler().fit(train[feature_cols])
    X_train, y_train = scaler.transform(train[feature_cols]), train[config.TARGET_COLUMN]
    X_val, y_val = scaler.transform(val[feature_cols]), val[config.TARGET_COLUMN]
    X_test, y_test = scaler.transform(test[feature_cols]), test[config.TARGET_COLUMN]

    # ── 4. Random Forest ────────────────────────────────────────────────
    logger.info("Training Random Forest (ntree=%d)...", config.RF_N_ESTIMATORS)
    rf, best_mtry, importance_df = ml_models.tune_and_fit_random_forest(X_train, y_train, X_val, y_val)
    pred_rf_val, pred_rf_test = rf.predict(X_val), rf.predict(X_test)

    # ── 5. XGBoost ──────────────────────────────────────────────────────
    logger.info("Training XGBoost (%s grid search)...", "QUICK 2-config" if quick else "FULL 36-config — slowest step")
    booster, xgb_best = ml_models.tune_and_fit_xgboost(X_train, y_train, quick_mode=quick)
    pred_xgb_val = booster.predict(xgb.DMatrix(X_val))
    pred_xgb_test = booster.predict(xgb.DMatrix(X_test))

    # SHAP (on a bounded sample of the test set, for the Explainability tab)
    logger.info("Computing SHAP values for the Explainability tab...")
    import shap
    shap_sample = X_test.sample(n=min(800, len(X_test)), random_state=config.RF_RANDOM_STATE).sort_index()
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(shap_sample)

    # ── 6. LSTM ─────────────────────────────────────────────────────────
    logger.info("Training LSTM (seq_len=%d, up to %d epochs, early stopping)...", config.LSTM_SEQ_LEN, config.LSTM_EPOCHS)
    Xtr_seq, ytr_seq = ml_models.build_lstm_sequences(X_train.to_numpy(), y_train.to_numpy())
    Xval_seq, _ = ml_models.build_lstm_sequences(X_val.to_numpy(), y_val.to_numpy())
    Xtest_seq, _ = ml_models.build_lstm_sequences(X_test.to_numpy(), y_test.to_numpy())
    lstm, lstm_history = ml_models.build_and_train_lstm(Xtr_seq, ytr_seq)
    pred_lstm_val_raw = lstm.predict(Xval_seq, verbose=0).ravel()
    pred_lstm_test_raw = lstm.predict(Xtest_seq, verbose=0).ravel()
    logger.info("LSTM stopped after %d epochs (best val_loss=%.4f)",
                len(lstm_history), lstm_history["val_loss"].min())

    # ── 7. Align RF/XGB down to the LSTM's shorter index, regime-scale ───
    seq_len = config.LSTM_SEQ_LEN
    pred_rf_val_a, pred_xgb_val_a, y_val_a, regime_val_a = _align_to_lstm(
        pred_rf_val, pred_xgb_val, y_val.to_numpy(), val["Regime_prob_lag1"].to_numpy())
    pred_rf_test_a, pred_xgb_test_a, y_test_a, regime_test_a = _align_to_lstm(
        pred_rf_test, pred_xgb_test, y_test.to_numpy(), test["Regime_prob_lag1"].to_numpy())
    dates_test_a = test.index[seq_len:]

    pred_lstm_val_regime = ml_models.regime_scale(pred_lstm_val_raw, regime_val_a)
    pred_lstm_test_regime = ml_models.regime_scale(pred_lstm_test_raw, regime_test_a)

    # ── 8. Ensemble weights — fit on VALIDATION ONLY ─────────────────────
    logger.info("Optimising ensemble weights on the validation split (step=%.2f)...", config.ENSEMBLE_WEIGHT_STEP)
    weights = ml_models.optimize_ensemble_weights(
        pred_rf_val_a, pred_xgb_val_a, pred_lstm_val_regime, y_val_a, step=config.ENSEMBLE_WEIGHT_STEP)
    logger.info("Weights -> RF=%.2f  XGB=%.2f  LSTM=%.2f  (validation Sharpe=%.3f)  "
                "[thesis reference: RF=%.2f XGB=%.2f LSTM=%.2f]",
                weights.rf, weights.xgb, weights.lstm, weights.validation_sharpe,
                config.THESIS_REFERENCE_WEIGHTS["rf"], config.THESIS_REFERENCE_WEIGHTS["xgb"],
                config.THESIS_REFERENCE_WEIGHTS["lstm"])

    # ── 9. Honest evaluation on the untouched TEST period ─────────────────
    pred_ensemble_test = (weights.rf * pred_rf_test_a + weights.xgb * pred_xgb_test_a
                           + weights.lstm * pred_lstm_test_regime)

    logger.info("── Out-of-sample test-period performance ──────────────────────")
    model_metrics = {"Buy-and-Hold": {
        "annualised_sharpe": risk_metrics.annualised_sharpe(y_test_a),
        "total_net_return_pct": float(np.sum(y_test_a)),
        "max_drawdown_pct": risk_metrics.max_drawdown(y_test_a),
        "directional_accuracy_pct": None, "sortino": None, "calmar": None,
        "pct_time_long": None, "pct_time_short": None,
    }}
    backtest_curves = {}
    for name, pred in [("Random Forest", pred_rf_test_a), ("XGBoost", pred_xgb_test_a),
                        ("LSTM", pred_lstm_test_raw), ("LSTM + Regime", pred_lstm_test_regime),
                        ("Ensemble", pred_ensemble_test)]:
        d, net_ret, sig = evaluate_and_log(name, pred, y_test_a)
        model_metrics[name] = d
        backtest_curves[name] = net_ret

    # ── 10. Persist backtest history for the dashboard charts ────────────
    backtest_df = pd.DataFrame({
        "Date": dates_test_a,
        "Actual_Return": y_test_a,
        "SP500_Close": test["SP500"].to_numpy()[seq_len:],
        "Regime_Prob_Crisis": regime_test_a,
        "Pred_RF": pred_rf_test_a, "Pred_XGB": pred_xgb_test_a,
        "Pred_LSTM": pred_lstm_test_raw, "Pred_LSTM_Regime": pred_lstm_test_regime,
        "Pred_Ensemble": pred_ensemble_test,
        "Ensemble_Signal": risk_metrics.signal_from_prediction(pred_ensemble_test),
        "Ensemble_Net_Return": backtest_curves["Ensemble"],
        "Ensemble_Equity": risk_metrics.equity_curve(backtest_curves["Ensemble"]),
        "BuyHold_Equity": risk_metrics.equity_curve(y_test_a),
    }).set_index("Date")
    backtest_df.to_parquet(config.ARTIFACT_FILES["backtest_history"])

    # ── 11. Save every artifact ────────────────────────────────────────
    logger.info("Saving artifacts to %s", config.ARTIFACT_DIR)
    scaler.save(config.ARTIFACT_FILES["scaler"])
    rmodel.save(config.ARTIFACT_FILES["hmm"])
    import joblib
    joblib.dump(rf, config.ARTIFACT_FILES["rf"])
    booster.save_model(str(config.ARTIFACT_FILES["xgb"]))
    lstm.save(config.ARTIFACT_FILES["lstm"])

    with open(config.ARTIFACT_FILES["ensemble_weights"], "w") as f:
        json.dump({"rf": weights.rf, "xgb": weights.xgb, "lstm": weights.lstm,
                    "validation_sharpe": weights.validation_sharpe}, f, indent=2)

    importance_df.to_json(config.ARTIFACT_FILES["feature_importance"], orient="records", indent=2)

    np.savez(str(config.ARTIFACT_FILES["feature_importance"]).replace(".json", "_shap.npz"),
             shap_values=shap_values, feature_values=shap_sample.to_numpy(),
             feature_names=np.array(feature_cols))

    lstm_history.to_json(config.ARTIFACT_FILES["training_history"], orient="records", indent=2)

    def _clean(d):
        return {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in d.items()}

    metadata = {
        "trained_at_utc": t0.isoformat(timespec="seconds"),
        "training_duration_seconds": (datetime.now(timezone.utc) - t0).total_seconds(),
        "quick_mode": quick,
        "data_range": [str(base_df.index.min().date()), str(base_df.index.max().date())],
        "n_rows_total": len(ml_df),
        "split_dates": {
            "train": [str(train.index.min().date()), str(train.index.max().date())],
            "validation": [str(val.index.min().date()), str(val.index.max().date())],
            "test": [str(test.index.min().date()), str(test.index.max().date())],
        },
        "feature_columns": feature_cols,
        "rf_best_mtry": int(best_mtry),
        "xgb_best_params": xgb_best["params"],
        "xgb_best_n_rounds": xgb_best["n_rounds"],
        "ensemble_weights": {"rf": weights.rf, "xgb": weights.xgb, "lstm": weights.lstm},
        "test_period_metrics": {k: _clean(v) for k, v in model_metrics.items()},
        "thesis_reference_weights": config.THESIS_REFERENCE_WEIGHTS,
        "thesis_reference_metrics": config.THESIS_REFERENCE_METRICS,
        "regime_stats": {
            "calm_pct": rmodel.regime_stats.pct_of_sample["calm"],
            "crisis_pct": rmodel.regime_stats.pct_of_sample["crisis"],
            "calm_mean_return": rmodel.regime_stats.mean_return["calm"],
            "crisis_mean_return": rmodel.regime_stats.mean_return["crisis"],
            "calm_vol": rmodel.regime_stats.volatility["calm"],
            "crisis_vol": rmodel.regime_stats.volatility["crisis"],
            "nber_match_rate_pct": rmodel.nber_match_rate,
        },
    }
    with open(config.ARTIFACT_FILES["metadata"], "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    logger.info("═" * 70)
    logger.info("Training complete in %.1f minutes. Artifacts written to %s", elapsed / 60, config.ARTIFACT_DIR)
    logger.info("Ensemble test-period Sharpe: %.3f | Net return: %.2f%% | Max DD: %.2f%%",
                model_metrics["Ensemble"]["annualised_sharpe"],
                model_metrics["Ensemble"]["total_net_return_pct"],
                model_metrics["Ensemble"]["max_drawdown_pct"])
    logger.info("(Thesis reference: Sharpe %.3f | Net return %.2f%% | Max DD %.2f%% — "
                "differs because of the val/test weight-fitting fix and a newer data vintage)",
                config.THESIS_REFERENCE_METRICS["sharpe"], config.THESIS_REFERENCE_METRICS["net_return_pct"],
                config.THESIS_REFERENCE_METRICS["max_drawdown_pct"])
    logger.info("═" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the S&P 500 MS-GJR-GARCH + ML ensemble pipeline.")
    parser.add_argument("--quick", action="store_true",
                         help="Tiny grids / low epoch counts for a fast smoke test. Run this first.")
    parser.add_argument("--refresh-data", action="store_true",
                         help="Bypass the cached raw dataset and refetch from FRED/Yahoo/GPR.")
    args = parser.parse_args()
    if args.quick:
        config.LSTM_EPOCHS = 10
        config.LSTM_EARLY_STOP_PATIENCE = 5
    main(quick=args.quick, refresh_data=args.refresh_data)
