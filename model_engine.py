"""
model_engine.py
──────────────────
The inference module. Loads the artifacts train_models.py produced and
turns "today's market data" into a live trading signal, regime read, and
risk report. Nothing in here re-fits any model — every fit happens offline
in train_models.py; this module only calls .predict() / .score(), which is
why a page load stays fast even though the underlying pipeline is heavy.

The one piece of care this module takes that's easy to get wrong: the
signal it returns is for the NEXT trading session, built from the most
recently completed day's data (see data_fetcher.build_live_feature_row —
this is deliberately NOT the same as the last row of a training-style
feature matrix, which would describe a session that's already closed).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import xgboost as xgb

import config
import data_fetcher
import ml_models
import regime_model
import risk_metrics

logger = logging.getLogger(__name__)


@dataclass
class ComponentPrediction:
    name: str
    predicted_return_pct: float
    weight: float
    signal: str  # "Long" | "Short"


@dataclass
class PreviousSessionReview:
    """What the model called for the last completed session vs. what happened.

    Recomputed live on every page load from the truncated data frame (data
    through T-1 scored with the identical code path as today's signal), so
    this card rolls forward daily with no retraining — unlike the frozen
    training backtest."""
    session_date: pd.Timestamp       # the completed session being reviewed
    sp500_close: float
    predicted_return_pct: float
    signal: str                      # "Long" | "Short"
    confidence_pct: float
    actual_return_pct: float
    hit: bool


@dataclass
class LiveSignal:
    as_of_date: pd.Timestamp          # most recent session used as input
    target_session_label: str          # human label, e.g. "next trading session"
    sp500_last_close: float
    predicted_return_pct: float
    signal: str                        # "Long" | "Short"
    confidence_pct: float
    components: list[ComponentPrediction]
    regime_label: str                  # "Calm" | "Crisis"
    regime_prob_crisis_pct: float
    next_day_vol_forecast_pct: float
    var_es: list[risk_metrics.VarEsResult]
    shap_today: Optional[dict] = field(default=None)  # {feature: shap_value}
    warnings: list[str] = field(default_factory=list)
    previous: Optional[PreviousSessionReview] = field(default=None)


class ModelEngine:
    def __init__(self):
        self.scaler: Optional[ml_models.FeatureScaler] = None
        self.regime: Optional[regime_model.RegimeModel] = None
        self.rf = None
        self.xgb_booster: Optional[xgb.Booster] = None
        self.lstm_session = None
        self.lstm_input_name: Optional[str] = None
        self.weights: Optional[dict] = None
        self.metadata: Optional[dict] = None
        self.feature_importance_df: Optional[pd.DataFrame] = None
        self.shap_reference: Optional[dict] = None
        self.backtest_df: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[list[str]] = None

    # ── Loading ─────────────────────────────────────────────────────────
    def load(self) -> "ModelEngine":
        import json

        self.scaler = ml_models.FeatureScaler.load(config.ARTIFACT_FILES["scaler"])
        self.regime = regime_model.RegimeModel.load(config.ARTIFACT_FILES["hmm"])
        self.rf = joblib.load(config.ARTIFACT_FILES["rf"])
        self.xgb_booster = xgb.Booster()
        self.xgb_booster.load_model(str(config.ARTIFACT_FILES["xgb"]))
        # ONNX inference — no TensorFlow required (see config.ARTIFACT_FILES["lstm"]).
        self.lstm_session = ort.InferenceSession(
            str(config.ARTIFACT_FILES["lstm"]), providers=["CPUExecutionProvider"])
        self.lstm_input_name = self.lstm_session.get_inputs()[0].name

        with open(config.ARTIFACT_FILES["ensemble_weights"]) as f:
            self.weights = json.load(f)
        with open(config.ARTIFACT_FILES["metadata"]) as f:
            self.metadata = json.load(f)
        self._feature_cols = self.metadata["feature_columns"]

        self.feature_importance_df = pd.read_json(config.ARTIFACT_FILES["feature_importance"])
        shap_npz_path = str(config.ARTIFACT_FILES["feature_importance"]).replace(".json", "_shap.npz")
        try:
            npz = np.load(shap_npz_path, allow_pickle=True)
            self.shap_reference = {
                "shap_values": npz["shap_values"], "feature_values": npz["feature_values"],
                "feature_names": npz["feature_names"].tolist(),
            }
        except FileNotFoundError:
            logger.warning("SHAP reference file not found at %s", shap_npz_path)

        self.backtest_df = pd.read_parquet(config.ARTIFACT_FILES["backtest_history"])
        logger.info("ModelEngine loaded artifacts trained at %s", self.metadata.get("trained_at_utc"))
        return self

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    # ── Live data ────────────────────────────────────────────────────────
    def fetch_live_dataset(self) -> pd.DataFrame:
        """Fetch a recent window and score it against the SAVED (fixed) HMM —
        never refits. Returns df_with_regime (not yet lagged/dropna'd)."""
        base_df = data_fetcher.build_base_daily_dataset(
            start=(pd.Timestamp.today() - pd.Timedelta(days=config.LIVE_LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d"),
            patch_live_vix=True,
        )
        prob_crisis = self.regime.score(base_df["SP500_ret"])
        is_crisis = (prob_crisis > 0.5).astype(int)
        return data_fetcher.add_regime_features(base_df, prob_crisis, is_crisis)

    # ── Inference ────────────────────────────────────────────────────────
    def _score_frame(self, live_row: pd.Series, lstm_window: pd.DataFrame,
                     regime_prob: float) -> tuple[float, list[ComponentPrediction]]:
        """Run RF/XGB/LSTM + ensemble weighting on one feature frame. Shared
        by today's signal and the previous-session review so both use the
        identical code path."""
        scaled_row = self.scaler.transform(live_row.to_frame().T)
        scaled_seq = self.scaler.transform(lstm_window)

        pred_rf = float(self.rf.predict(scaled_row)[0])
        pred_xgb = float(self.xgb_booster.predict(xgb.DMatrix(scaled_row))[0])
        lstm_input = scaled_seq.to_numpy().reshape(1, config.LSTM_SEQ_LEN, len(self.feature_cols)).astype(np.float32)
        pred_lstm_raw = float(self.lstm_session.run(None, {self.lstm_input_name: lstm_input})[0].ravel()[0])

        pred_lstm_regime = float(ml_models.regime_scale(
            np.array([pred_lstm_raw]), np.array([regime_prob]))[0])

        w = self.weights
        ensemble_pred = w["rf"] * pred_rf + w["xgb"] * pred_xgb + w["lstm"] * pred_lstm_regime

        components = [
            ComponentPrediction("Random Forest", pred_rf, w["rf"], "Long" if pred_rf > 0 else "Short"),
            ComponentPrediction("XGBoost", pred_xgb, w["xgb"], "Long" if pred_xgb > 0 else "Short"),
            ComponentPrediction("LSTM (regime-scaled)", pred_lstm_regime, w["lstm"],
                                 "Long" if pred_lstm_regime > 0 else "Short"),
        ]
        return ensemble_pred, components

    def _review_previous_session(self, df_with_regime: pd.DataFrame) -> Optional[PreviousSessionReview]:
        """Score the frame ending the day BEFORE the last completed session —
        i.e. what the model called for the session that just closed — and pair
        it with the realised outcome. Returns None (caller falls back to the
        frozen backtest tail) if history is too short."""
        try:
            if len(df_with_regime.index) < 2:
                return None
            session_date = df_with_regime.index.max()
            trunc = df_with_regime.iloc[:-1]
            live_row = data_fetcher.build_live_feature_row(trunc, self.feature_cols)
            lstm_window = data_fetcher.build_live_lstm_sequence(trunc, self.feature_cols)
            regime_prob = float(live_row["Regime_prob_lag1"])
            ensemble_pred, components = self._score_frame(live_row, lstm_window, regime_prob)
            actual = float(df_with_regime["SP500_ret"].iloc[-1])
            sig = "Long" if ensemble_pred > 0 else "Short"
            return PreviousSessionReview(
                session_date=session_date,
                sp500_close=float(df_with_regime["SP500"].iloc[-1]),
                predicted_return_pct=ensemble_pred,
                signal=sig,
                confidence_pct=self._confidence_score(ensemble_pred, components),
                actual_return_pct=actual,
                hit=(ensemble_pred > 0) == (actual > 0),
            )
        except Exception as exc:
            logger.warning("Previous-session review failed (%s); dashboard falls back to backtest.", exc)
            return None

    def predict_today(self) -> LiveSignal:
        warnings: list[str] = []
        df_with_regime = self.fetch_live_dataset()

        live_row = data_fetcher.build_live_feature_row(df_with_regime, self.feature_cols)
        lstm_window = data_fetcher.build_live_lstm_sequence(df_with_regime, self.feature_cols)
        as_of_date = df_with_regime.index.max()

        current_regime_prob = float(live_row["Regime_prob_lag1"])
        ensemble_pred, components = self._score_frame(live_row, lstm_window, current_regime_prob)
        signal = "Long" if ensemble_pred > 0 else "Short"

        confidence_pct = self._confidence_score(ensemble_pred, components)

        regime_label = "Crisis" if current_regime_prob > 0.5 else "Calm"

        # ── Risk metrics, using the regime-mixture forecast + GARCH vol ──
        recent_returns = df_with_regime["SP500_ret"]
        mu_forecast = float(self.regime.mixture_forecast(
            pd.Series([1 - current_regime_prob]), pd.Series([current_regime_prob])).iloc[0])
        try:
            sigma_forecast = self.regime.forecast_next_day_volatility(recent_returns)
        except Exception as exc:
            logger.warning("Volatility forecast failed (%s); falling back to trailing std.", exc)
            warnings.append("GARCH volatility forecast unavailable this run — used trailing 21-day std instead.")
            sigma_forecast = float(recent_returns.tail(21).std(ddof=1))
        dof = float(self.regime.garch_full_params.get("nu", 6.0))
        var_es = risk_metrics.full_risk_report(recent_returns, mu_forecast, sigma_forecast, dof)

        # ── SHAP for today's specific XGBoost prediction (local explain) ──
        shap_today = None
        try:
            import shap
            scaled_row = self.scaler.transform(live_row.to_frame().T)
            explainer = shap.TreeExplainer(self.xgb_booster)
            vals = explainer.shap_values(scaled_row)
            shap_today = dict(zip(self.feature_cols, np.ravel(vals).tolist()))
        except Exception as exc:
            logger.warning("Live SHAP computation failed: %s", exc)
            warnings.append("Live SHAP explanation unavailable this run.")

        previous = self._review_previous_session(df_with_regime)

        return LiveSignal(
            as_of_date=as_of_date,
            target_session_label="next trading session",
            sp500_last_close=float(df_with_regime["SP500"].iloc[-1]),
            predicted_return_pct=ensemble_pred,
            signal=signal,
            confidence_pct=confidence_pct,
            components=components,
            regime_label=regime_label,
            regime_prob_crisis_pct=current_regime_prob * 100,
            next_day_vol_forecast_pct=sigma_forecast,
            var_es=var_es,
            shap_today=shap_today,
            warnings=warnings,
            previous=previous,
        )

    def _confidence_score(self, ensemble_pred: float, components: list[ComponentPrediction]) -> float:
        """0-100 score blending (a) how many components agree with the
        ensemble's direction and (b) how large today's |prediction| is
        relative to the model's own historical prediction distribution
        (from the saved test-period backtest)."""
        agree = np.mean([1.0 if c.signal == ("Long" if ensemble_pred > 0 else "Short") else 0.0
                          for c in components]) * 100
        ref = self.backtest_df["Pred_Ensemble"].abs()
        magnitude_pct = float((ref < abs(ensemble_pred)).mean() * 100) if len(ref) else 50.0
        return float(np.clip(0.5 * agree + 0.5 * magnitude_pct, 0, 100))

    # ── Convenience accessors for the dashboard ────────────────────────
    def get_test_period_metrics(self) -> dict:
        return self.metadata["test_period_metrics"]

    def get_thesis_reference(self) -> tuple[dict, dict]:
        return self.metadata["thesis_reference_weights"], self.metadata["thesis_reference_metrics"]

    def get_regime_stats(self) -> dict:
        return self.metadata["regime_stats"]
