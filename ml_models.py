"""
ml_models.py
─────────────
The three base learners plus the ensemble logic. Every hyperparameter and
architecture choice is ported from the Rmd's chapter 6 (Random Forest →
XGBoost → LSTM → regime-conditional scaling → Sharpe-maximised ensemble
weights) — see config.py for the exact numeric constants and the docstring
on each function below for the specific Rmd chunk it mirrors.

TensorFlow is imported lazily inside the LSTM functions only, so the rest
of this module (and anything that imports it) stays usable without paying
TensorFlow's import cost or requiring it to be installed.
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import config

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Feature scaler — z-score on TRAINING statistics only, ddof=1 to exactly
# match R's scale()/sd() (sklearn's StandardScaler defaults to ddof=0).
# Rmd: train_means <- colMeans(...); train_sds <- apply(..., sd); scale(...)
# ═════════════════════════════════════════════════════════════════════════

class FeatureScaler:
    def __init__(self):
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None
        self.columns_: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        self.columns_ = list(X.columns)
        self.mean_ = X.mean()
        std = X.std(ddof=1)
        self.std_ = std.mask(std == 0, 1.0)  # Rmd: train_sds[train_sds == 0] <- 1
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X[self.columns_] - self.mean_) / self.std_

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FeatureScaler":
        return joblib.load(path)


# ═════════════════════════════════════════════════════════════════════════
# Random Forest (Rmd chunk "MODEL 1: RANDOM FOREST")
# ═════════════════════════════════════════════════════════════════════════

def tune_and_fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series):
    """
    Rmd: mtry_grid <- unique(c(floor(p/3), floor(sqrt(p)), floor(p/2)));
         pick mtry by min OOB MSE at ntree=500, then refit at ntree=1000.
    Feature importance is computed by permutation on the held-out
    VALIDATION split (not train, not test) — the closest honest analog to
    R's OOB %IncMSE available without re-deriving per-tree OOB masks.
    """
    p = X_train.shape[1]
    mtry_grid = config.RF_MTRY_CANDIDATES_FN(p)
    logger.info("RF: testing mtry values %s", mtry_grid)

    oob_mse = {}
    for m in mtry_grid:
        rf_tmp = RandomForestRegressor(
            n_estimators=500, max_features=m, min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            oob_score=True, random_state=config.RF_RANDOM_STATE, n_jobs=-1,
        )
        rf_tmp.fit(X_train, y_train)
        oob_pred = rf_tmp.oob_prediction_
        oob_mse[m] = float(np.mean((y_train.to_numpy() - oob_pred) ** 2))

    best_mtry = min(oob_mse, key=oob_mse.get)
    logger.info("RF: best mtry = %d (OOB MSE = %.4f)", best_mtry, oob_mse[best_mtry])

    rf_final = RandomForestRegressor(
        n_estimators=config.RF_N_ESTIMATORS, max_features=best_mtry,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF, oob_score=True,
        random_state=config.RF_RANDOM_STATE, n_jobs=-1,
    )
    rf_final.fit(X_train, y_train)

    perm = permutation_importance(
        rf_final, X_val, y_val, n_repeats=15, random_state=config.RF_RANDOM_STATE,
        scoring="neg_mean_squared_error", n_jobs=-1,
    )
    importance_df = (
        pd.DataFrame({"Feature": X_train.columns, "Importance": perm.importances_mean})
        .sort_values("Importance", ascending=False).reset_index(drop=True)
    )
    return rf_final, best_mtry, importance_df


# ═════════════════════════════════════════════════════════════════════════
# XGBoost (Rmd chunk "MODEL 2: XGBOOST")
# ═════════════════════════════════════════════════════════════════════════

def tune_and_fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, quick_mode: bool = False):
    """
    Rmd: expand.grid(max_depth, eta, subsample, colsample) -> 5-fold xgb.cv
         with early_stopping_rounds=30 -> refit xgb.train on the winner.
    quick_mode shrinks the grid for fast local smoke-tests; a real training
    run should use the full grid (default).
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    grid = config.XGB_GRID
    combos = list(itertools.product(grid["max_depth"], grid["learning_rate"],
                                     grid["subsample"], grid["colsample_bytree"]))
    if quick_mode:
        combos = combos[:2]
    logger.info("XGBoost: tuning over %d configurations...", len(combos))

    best = {"rmse": np.inf, "params": None, "n_rounds": config.XGB_MAX_ROUNDS}
    for max_depth, eta, subsample, colsample in combos:
        params = {
            "objective": config.XGB_FIXED_PARAMS["objective"],
            "max_depth": max_depth, "eta": eta, "subsample": subsample,
            "colsample_bytree": colsample,
            "min_child_weight": config.XGB_FIXED_PARAMS["min_child_weight"],
            "gamma": config.XGB_FIXED_PARAMS["gamma"],
            "lambda": config.XGB_FIXED_PARAMS["reg_lambda"],
            "seed": config.XGB_RANDOM_STATE,
        }
        cv_results = xgb.cv(
            params, dtrain, num_boost_round=config.XGB_MAX_ROUNDS, nfold=config.XGB_CV_FOLDS,
            early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS, metrics="rmse",
            seed=config.XGB_RANDOM_STATE, verbose_eval=False,
        )
        best_rmse = cv_results["test-rmse-mean"].min()
        if best_rmse < best["rmse"]:
            best.update(rmse=float(best_rmse), params=params,
                        n_rounds=int(cv_results["test-rmse-mean"].idxmin()) + 1)

    logger.info("XGBoost: best params %s, n_rounds=%d, cv rmse=%.4f",
                best["params"], best["n_rounds"], best["rmse"])
    booster = xgb.train(best["params"], dtrain, num_boost_round=best["n_rounds"])
    return booster, best


# ═════════════════════════════════════════════════════════════════════════
# LSTM (Rmd chunk "MODEL 3: LSTM")
# ═════════════════════════════════════════════════════════════════════════

def build_lstm_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = config.LSTM_SEQ_LEN):
    """Rmd: build_sequences() — sliding window, predicts one step ahead."""
    n, n_feat = X.shape
    n_seq = n - seq_len
    X_seq = np.zeros((n_seq, seq_len, n_feat), dtype=np.float32)
    y_seq = np.zeros(n_seq, dtype=np.float32)
    for i in range(n_seq):
        X_seq[i] = X[i:i + seq_len]
        y_seq[i] = y[i + seq_len]
    return X_seq, y_seq


def build_and_train_lstm(X_train_seq: np.ndarray, y_train_seq: np.ndarray):
    """
    Rmd architecture exactly:
    Input(seq=21,feat) -> LSTM(64,return_seq) -> Drop(.3) -> LSTM(32)
    -> Drop(.2) -> Dense(16,relu) -> Drop(.1) -> Dense(1,linear)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.random.set_seed(config.LSTM_RANDOM_STATE)
    np.random.seed(config.LSTM_RANDOM_STATE)

    seq_len, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
    inputs = keras.Input(shape=(seq_len, n_features), name="input_layer")
    x = layers.LSTM(config.LSTM_UNITS_1, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(config.LSTM_DROPOUT_1, name="dropout_1")(x)
    x = layers.LSTM(config.LSTM_UNITS_2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(config.LSTM_DROPOUT_2, name="dropout_2")(x)
    x = layers.Dense(config.LSTM_DENSE_UNITS, activation="relu", name="dense_1")(x)
    x = layers.Dropout(config.LSTM_DROPOUT_3, name="dropout_3")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)
    model = keras.Model(inputs, outputs, name="LSTM_ReturnPredictor")

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.LSTM_EARLY_STOP_PATIENCE,
                                       restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=config.LSTM_REDUCE_LR_FACTOR,
                                           patience=config.LSTM_REDUCE_LR_PATIENCE,
                                           min_lr=config.LSTM_MIN_LR, verbose=0),
    ]
    history = model.fit(
        X_train_seq, y_train_seq, epochs=config.LSTM_EPOCHS, batch_size=config.LSTM_BATCH_SIZE,
        validation_split=config.LSTM_VALIDATION_SPLIT, callbacks=callbacks,
        shuffle=False, verbose=0,
    )
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
    return model, hist_df


# ═════════════════════════════════════════════════════════════════════════
# Regime-conditional LSTM signal scaling (Rmd: regime_scale())
# ═════════════════════════════════════════════════════════════════════════

def regime_scale(pred: np.ndarray, regime_prob: np.ndarray,
                  calm_thresh: float = config.REGIME_CALM_THRESHOLD,
                  crisis_thresh: float = config.REGIME_CRISIS_THRESHOLD,
                  amp_factor: float = config.REGIME_AMPLIFY_FACTOR,
                  pen_factor: float = config.REGIME_PENALIZE_FACTOR) -> np.ndarray:
    """
    Rmd: amplify when regime and prediction direction agree, penalise when
    they conflict, leave untouched in the ambiguous middle band.
    """
    pred = np.atleast_1d(np.asarray(pred, dtype=float))
    regime_prob = np.atleast_1d(np.asarray(regime_prob, dtype=float))

    calm_long = (regime_prob < calm_thresh) & (pred > 0)
    crisis_short = (regime_prob > crisis_thresh) & (pred < 0)
    conflict = (
        ((regime_prob >= calm_thresh) & (regime_prob <= crisis_thresh))
        | ((regime_prob < calm_thresh) & (pred < 0))
        | ((regime_prob > crisis_thresh) & (pred > 0))
    )
    scaled = pred.copy()
    scaled[calm_long] *= amp_factor
    scaled[crisis_short] *= amp_factor
    scaled[conflict] *= pen_factor
    return scaled


# ═════════════════════════════════════════════════════════════════════════
# Ensemble weight optimisation (Rmd: nested grid search maximising Sharpe)
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleWeights:
    rf: float
    xgb: float
    lstm: float
    validation_sharpe: float


def optimize_ensemble_weights(pred_rf: np.ndarray, pred_xgb: np.ndarray, pred_lstm_regime: np.ndarray,
                               actual: np.ndarray, cost: float = config.TRANSACTION_COST,
                               step: float = config.ENSEMBLE_WEIGHT_STEP) -> EnsembleWeights:
    """
    Rmd: grid search w1 (RF), w2 (XGB), w3 (LSTM+Regime) on a `step`-resolution
    simplex, maximising annualised Sharpe of the resulting long/short strategy.

    IMPORTANT DEVIATION FROM THE RMD: the notebook ran this search directly
    on the OOS test set, which quietly fits the ensemble weights to the very
    data used to report performance (the run comment claims "training-set
    Sharpe maximisation" but the code operates on `pred_rf_aligned` /
    `pred_xgb_aligned` / `pred_lstm_regime`, both sliced from `test_ml`).
    Call this function with VALIDATION-split predictions only — see
    train_models.py — so the live app's weights are chosen without ever
    looking at the test period.
    """
    from risk_metrics import annualised_sharpe, net_strategy_returns, signal_from_prediction

    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    best_sharpe, best_w = -np.inf, (1 / 3, 1 / 3, 1 / 3)
    for w1 in grid:
        remaining = round(1 - w1, 4)
        if remaining < 0:
            continue
        for w2 in np.round(np.arange(0, remaining + 1e-9, step), 4):
            w3 = round(1 - w1 - w2, 4)
            if w3 < 0:
                continue
            ens_pred = w1 * pred_rf + w2 * pred_xgb + w3 * pred_lstm_regime
            signal = signal_from_prediction(ens_pred)
            net_ret = net_strategy_returns(signal, actual, cost)
            sharpe = annualised_sharpe(net_ret)
            if np.isfinite(sharpe) and sharpe > best_sharpe:
                best_sharpe, best_w = sharpe, (w1, w2, w3)

    return EnsembleWeights(rf=best_w[0], xgb=best_w[1], lstm=best_w[2], validation_sharpe=float(best_sharpe))
