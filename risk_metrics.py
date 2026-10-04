"""
risk_metrics.py
─────────────────
Two families of metrics:

1. Strategy performance metrics — Sharpe, max drawdown, Calmar, net return —
   ported EXACTLY from the Rmd's `eval_full()` function so the live app's
   numbers are computed the same way the thesis's numbers were.

2. Value-at-Risk / Expected Shortfall — NOT present anywhere in the thesis.
   This is new, added because the live app's requirements explicitly ask
   for it. Two independent methods are provided (historical simulation and
   GARCH-parametric) so the dashboard can show them side by side rather
   than presenting one number as gospel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

import config


# ═════════════════════════════════════════════════════════════════════════
# 1. Strategy performance metrics (Rmd: eval_full())
# ═════════════════════════════════════════════════════════════════════════

def signal_from_prediction(pred: np.ndarray) -> np.ndarray:
    """Rmd: sig <- ifelse(pred > 0, 1, -1) — strictly long/short, no cash state."""
    return np.where(np.asarray(pred) > 0, 1, -1)


def net_strategy_returns(signal: np.ndarray, actual_returns: np.ndarray,
                          cost: float = config.TRANSACTION_COST) -> np.ndarray:
    """Rmd: net_ret <- signal*actual - cost*abs(diff(c(0, signal)))"""
    signal = np.asarray(signal, dtype=float)
    actual_returns = np.asarray(actual_returns, dtype=float)
    gross = signal * actual_returns
    turnover = np.abs(np.diff(np.concatenate([[0.0], signal])))
    return gross - cost * 100 * turnover  # cost is a fraction; returns are in % -> *100


def annualised_sharpe(net_returns: np.ndarray, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    """Rmd: sharpe <- mean(net_ret)/sd(net_ret) * sqrt(252)"""
    net_returns = np.asarray(net_returns)
    sd = net_returns.std(ddof=1)
    return float(net_returns.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def sortino_ratio(net_returns: np.ndarray, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    """NEW (not in thesis): Sharpe variant using only downside deviation."""
    net_returns = np.asarray(net_returns)
    downside = net_returns[net_returns < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else np.nan
    return float(net_returns.mean() / dd * np.sqrt(periods_per_year)) if dd and dd > 0 else float("nan")


def equity_curve(net_returns: np.ndarray) -> np.ndarray:
    """Rmd: cum_wealth <- cumprod(1 + net_ret / 100)"""
    return np.cumprod(1 + np.asarray(net_returns) / 100.0)


def drawdown_series(net_returns: np.ndarray) -> np.ndarray:
    """Rmd: (cum_wealth - rolling_max) / rolling_max * 100"""
    wealth = equity_curve(net_returns)
    running_max = np.maximum.accumulate(wealth)
    return (wealth - running_max) / running_max * 100.0


def max_drawdown(net_returns: np.ndarray) -> float:
    dd = drawdown_series(net_returns)
    return float(dd.min()) if len(dd) else float("nan")


def calmar_ratio(net_returns: np.ndarray, periods_per_year: int = config.TRADING_DAYS_PER_YEAR) -> float:
    """Rmd: calmar <- ann_ret / abs(max_dd), ann_ret <- mean(net_ret) * 252"""
    ann_ret = float(np.mean(net_returns) * periods_per_year)
    mdd = max_drawdown(net_returns)
    return float(ann_ret / abs(mdd)) if mdd and abs(mdd) > 0 else float("nan")


@dataclass
class StrategyMetrics:
    directional_accuracy_pct: float
    annualised_sharpe: float
    sortino: float
    calmar: float
    total_net_return_pct: float
    buy_and_hold_return_pct: float
    max_drawdown_pct: float
    pct_time_long: float
    pct_time_short: float


def evaluate_strategy(pred: np.ndarray, actual: np.ndarray,
                       cost: float = config.TRANSACTION_COST) -> tuple[StrategyMetrics, np.ndarray, np.ndarray]:
    """Direct port of the Rmd's eval_full(). Returns metrics, net_returns, signal."""
    pred, actual = np.asarray(pred), np.asarray(actual)
    signal = signal_from_prediction(pred)
    net_ret = net_strategy_returns(signal, actual, cost)

    metrics = StrategyMetrics(
        directional_accuracy_pct=float(np.mean(np.sign(pred) == np.sign(actual)) * 100),
        annualised_sharpe=annualised_sharpe(net_ret),
        sortino=sortino_ratio(net_ret),
        calmar=calmar_ratio(net_ret),
        total_net_return_pct=float(np.sum(net_ret)),
        buy_and_hold_return_pct=float(np.sum(actual)),
        max_drawdown_pct=max_drawdown(net_ret),
        pct_time_long=float(np.mean(signal == 1) * 100),
        pct_time_short=float(np.mean(signal == -1) * 100),
    )
    return metrics, net_ret, signal


# ═════════════════════════════════════════════════════════════════════════
# 2. Value-at-Risk / Expected Shortfall (NEW — not in the thesis)
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class VarEsResult:
    confidence: float
    historical_var_pct: float
    historical_es_pct: float
    parametric_var_pct: float
    parametric_es_pct: float


def historical_var_es(returns_pct: pd.Series, confidence: float = 0.95,
                       lookback: int = config.VAR_LOOKBACK_DAYS) -> tuple[float, float]:
    """
    Historical-simulation VaR/ES on the trailing `lookback` days of realised
    returns. Reported as positive percentages (magnitude of loss).
    """
    window = returns_pct.dropna().tail(lookback)
    alpha = 1 - confidence
    var = -np.percentile(window, alpha * 100)
    tail = window[window <= -var]
    es = -tail.mean() if len(tail) > 0 else var
    return float(var), float(es)


def parametric_var_es(mu_pct: float, sigma_pct: float, confidence: float = 0.95,
                       dist: str = "t", dof: float = 6.0) -> tuple[float, float]:
    """
    Closed-form-equivalent VaR/ES from a fitted location-scale distribution
    (Student-t by default, matching the GJR-GARCH's fitted innovation
    distribution; falls back to Normal). Uses numerical integration
    (`rv.expect`) for ES so no distribution-specific formula has to be
    hand-derived (and risks a sign/scaling bug).
    """
    alpha = 1 - confidence
    if dist == "t" and dof and dof > 2.01:
        scale = sigma_pct * np.sqrt((dof - 2) / dof)
        rv = stats.t(df=dof, loc=mu_pct, scale=scale)
    else:
        rv = stats.norm(loc=mu_pct, scale=sigma_pct)

    var_threshold = rv.ppf(alpha)
    var = -float(var_threshold)
    lower_bound = mu_pct - 60 * sigma_pct  # effectively -infinity for these tail integrals
    es = -float(rv.expect(lambda x: x, lb=lower_bound, ub=var_threshold) / alpha)
    return var, es


def full_risk_report(returns_pct: pd.Series, latest_mu_pct: float, latest_sigma_pct: float,
                      garch_dof: float = 6.0,
                      levels: list[float] = config.VAR_CONFIDENCE_LEVELS) -> list[VarEsResult]:
    results = []
    for c in levels:
        h_var, h_es = historical_var_es(returns_pct, c)
        p_var, p_es = parametric_var_es(latest_mu_pct, latest_sigma_pct, c, dist="t", dof=garch_dof)
        results.append(VarEsResult(c, h_var, h_es, p_var, p_es))
    return results
