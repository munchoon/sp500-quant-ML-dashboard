"""
regime_model.py
────────────────
The econometric layer: a 2-state Gaussian HMM classifies each trading day
as "Calm" or "Crisis" (Rmd: depmixS4::depmix(SP500_ret ~ 1, nstates=2)),
and a GJR-GARCH(1,1) is fit separately on each regime's returns (Rmd:
ugarchspec(model="gjrGARCH") fit on regime1_returns / regime2_returns) to
produce the MS-GJR-GARCH mixture forecast used as a baseline signal and to
feed Regime_prob_lag1 / Regime_bin_lag1 into the ML ensemble.

Design note on train vs. live:
  Fitting (hmm.fit(), arch's .fit()) happens ONLY in train_models.py, on
  the full history. Live inference in model_engine.py never re-fits —
  it calls RegimeModel.score() to run the (cheap) forward-backward
  inference pass over a recent window using the SAVED, fixed model
  parameters, exactly the same "fit rarely, infer often" pattern used for
  the RF/XGBoost/LSTM models. This keeps every page load fast and keeps
  one consistent mental model across the whole pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

import config

logger = logging.getLogger(__name__)


@dataclass
class RegimeStats:
    mean_return: dict = field(default_factory=dict)   # {"calm": x, "crisis": y}
    volatility: dict = field(default_factory=dict)
    sharpe: dict = field(default_factory=dict)
    pct_of_sample: dict = field(default_factory=dict)


class RegimeModel:
    """Bundles the fitted HMM + regime-conditional GJR-GARCH parameters."""

    def __init__(self):
        self.hmm: Optional[GaussianHMM] = None
        self.crisis_state_idx: Optional[int] = None
        self.garch_calm_params: Optional[pd.Series] = None
        self.garch_crisis_params: Optional[pd.Series] = None
        self.garch_full_params: Optional[pd.Series] = None
        self.transition_matrix: Optional[np.ndarray] = None
        self.regime_stats: Optional[RegimeStats] = None
        self.nber_match_rate: Optional[float] = None

    # ── Fitting (offline / train_models.py only) ──────────────────────────
    def fit(self, returns: pd.Series, recession: Optional[pd.Series] = None) -> "RegimeModel":
        returns = returns.dropna()
        X = returns.to_numpy().reshape(-1, 1)

        self.hmm = GaussianHMM(
            n_components=config.HMM_N_STATES,
            covariance_type="diag",
            n_iter=config.HMM_N_ITER,
            random_state=config.HMM_RANDOM_STATE,
        )
        self.hmm.fit(X)

        post = self.hmm.predict_proba(X)  # (n, 2) smoothed posterior — analog of depmixS4::posterior()
        variances = self.hmm.covars_.reshape(config.HMM_N_STATES, -1)[:, 0]
        self.crisis_state_idx = int(np.argmax(variances))
        calm_idx = 1 - self.crisis_state_idx
        self.transition_matrix = self.hmm.transmat_.copy()

        prob_crisis = pd.Series(post[:, self.crisis_state_idx], index=returns.index, name="Regime_prob_crisis")
        state_path = pd.Series(np.argmax(post, axis=1), index=returns.index)
        is_crisis = (state_path == self.crisis_state_idx).astype(int)
        is_crisis.name = "Regime_binary"

        self.regime_stats = RegimeStats(
            mean_return={"calm": returns[is_crisis == 0].mean(), "crisis": returns[is_crisis == 1].mean()},
            volatility={"calm": returns[is_crisis == 0].std(ddof=1), "crisis": returns[is_crisis == 1].std(ddof=1)},
            sharpe={
                "calm": returns[is_crisis == 0].mean() / returns[is_crisis == 0].std(ddof=1),
                "crisis": returns[is_crisis == 1].mean() / returns[is_crisis == 1].std(ddof=1),
            },
            pct_of_sample={"calm": float((is_crisis == 0).mean() * 100), "crisis": float((is_crisis == 1).mean() * 100)},
        )

        if recession is not None:
            rec = recession.reindex(returns.index).fillna(0)
            crisis_days = int((is_crisis == 1).sum())
            match = int(((is_crisis == 1) & (rec == 1)).sum())
            self.nber_match_rate = 100.0 * match / crisis_days if crisis_days else np.nan

        # ── Regime-conditional GJR-GARCH (Rmd: fit_r1 / fit_r2) ───────────
        self.garch_calm_params = self._fit_gjr_garch(returns[is_crisis == 0])
        self.garch_crisis_params = self._fit_gjr_garch(returns[is_crisis == 1])
        # Full-sample GJR-GARCH, used only for the parametric-VaR volatility forecast
        self.garch_full_params = self._fit_gjr_garch(returns)

        self._last_prob_crisis = prob_crisis
        self._last_is_crisis = is_crisis
        return self

    @staticmethod
    def _fit_gjr_garch(returns: pd.Series) -> pd.Series:
        """Rmd: ugarchspec(model='gjrGARCH', garchOrder=c(1,1)), dist='std'"""
        am = arch_model(
            returns.to_numpy(), mean="Constant",
            vol="GARCH", p=config.GARCH_P, o=config.GARCH_O, q=config.GARCH_Q,
            dist=config.GARCH_DIST, rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)
        return res.params

    # ── Historical (in-sample, full-history) outputs for training ─────────
    def historical_regime_columns(self) -> tuple[pd.Series, pd.Series]:
        return self._last_prob_crisis, self._last_is_crisis

    def mixture_forecast(self, prob_calm: pd.Series, prob_crisis: pd.Series) -> pd.Series:
        """Rmd: daily_df$MS_GARCH_pred <- prob_calm*mu_r1 + prob_crisis*mu_r2"""
        mu_calm = self.garch_calm_params["mu"]
        mu_crisis = self.garch_crisis_params["mu"]
        return prob_calm * mu_calm + prob_crisis * mu_crisis

    # ── Live scoring (model_engine.py) ─────────────────────────────────────
    def score(self, returns_window: pd.Series) -> pd.Series:
        """
        Run forward-backward inference (NOT re-fitting) over a recent window
        of returns using the saved, fixed HMM parameters. Returns the
        crisis-state posterior probability for every day in the window; the
        last value is "today's" regime probability.
        """
        X = returns_window.dropna().to_numpy().reshape(-1, 1)
        post = self.hmm.predict_proba(X)
        return pd.Series(post[:, self.crisis_state_idx], index=returns_window.dropna().index,
                          name="Regime_prob_crisis")

    def forecast_next_day_volatility(self, latest_returns: pd.Series) -> float:
        """
        One-step-ahead conditional volatility (%) from the full-sample
        GJR-GARCH, refreshed against the latest live data without
        re-estimating parameters (arch's `.fix()` applies saved params to
        new data — the GARCH analog of `model.predict()`).
        Falls back to a quick refit on the recent window if `.fix()` fails
        for any reason (e.g. a version mismatch), so VaR never breaks.
        """
        try:
            am = arch_model(
                latest_returns.to_numpy(), mean="Constant",
                vol="GARCH", p=config.GARCH_P, o=config.GARCH_O, q=config.GARCH_Q,
                dist=config.GARCH_DIST, rescale=False,
            )
            fixed = am.fix(self.garch_full_params)
            fcast = fixed.forecast(horizon=1, reindex=False)
            return float(np.sqrt(fcast.variance.values[-1, 0]))
        except Exception as exc:
            logger.warning("GARCH .fix() forecast failed (%s); refitting on recent window.", exc)
            recent = latest_returns.dropna().tail(1000)
            am = arch_model(recent.to_numpy(), mean="Constant", vol="GARCH",
                             p=config.GARCH_P, o=config.GARCH_O, q=config.GARCH_Q,
                             dist=config.GARCH_DIST, rescale=False)
            res = am.fit(disp="off", show_warning=False)
            fcast = res.forecast(horizon=1, reindex=False)
            return float(np.sqrt(fcast.variance.values[-1, 0]))

    # ── Persistence ─────────────────────────────────────────────────────
    def save(self, path=config.ARTIFACT_FILES["hmm"]):
        joblib.dump(self, path)
        logger.info("RegimeModel saved to %s", path)

    @staticmethod
    def load(path=config.ARTIFACT_FILES["hmm"]) -> "RegimeModel":
        return joblib.load(path)
