"""Evaluation metrics for regression, classification, and portfolio tasks."""

import numpy as np
import pandas as pd


# ── Regression (Tasks 1 & 2) ─────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  mae(y_true, y_pred),
        "MSE":  mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }


# ── Classification (Task 3) ──────────────────────────────────────────────────

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score
    )
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }


# ── Portfolio (Task 4) ────────────────────────────────────────────────────────

def annualized_return(portfolio_values: np.ndarray, trading_days: int = 252) -> float:
    """Annualized return from a time series of portfolio values."""
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    n_years = len(portfolio_values) / trading_days
    return float((1 + total_return) ** (1 / n_years) - 1)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Annualized Sharpe Ratio from daily returns."""
    excess = returns - risk_free_rate / trading_days
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(trading_days) * excess.mean() / excess.std())


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - peak) / peak
    return float(drawdowns.min())


def portfolio_report(
    portfolio_values: np.ndarray,
    trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return {
        "Annualized Return": annualized_return(portfolio_values, trading_days),
        "Sharpe Ratio":      sharpe_ratio(daily_returns, risk_free_rate, trading_days),
        "Max Drawdown":      max_drawdown(portfolio_values),
    }


def print_report(report: dict, title: str = "") -> None:
    if title:
        print(f"\n{'─' * 35}")
        print(f"  {title}")
        print(f"{'─' * 35}")
    for k, v in report.items():
        print(f"  {k:<20} {v:.4f}")
