"""Data loading, windowing, splitting, and normalization utilities."""

import numpy as np
import pandas as pd
from pathlib import Path


NASDAQ_COLS = ["Date", "Low", "Open", "Volume", "High", "Close", "Adjusted Close"]
VN_COLS     = ["TradingDate", "Open", "High", "Low", "Close", "Volume"]
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def load_nasdaq(ticker: str, data_dir: str = "Nasdaq data/csv") -> pd.DataFrame:
    path = Path(data_dir) / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def load_vietnam(ticker: str, data_dir: str = "Vietnam data/stock-historical-data") -> pd.DataFrame:
    matches = list(Path(data_dir).glob(f"{ticker}-*-History.csv"))
    if not matches:
        raise FileNotFoundError(f"No history file found for ticker {ticker}")
    df = pd.read_csv(matches[0], parse_dates=["TradingDate"]).sort_values("TradingDate").reset_index(drop=True)
    return df


def make_windows(
    series: np.ndarray,
    window_size: int = 30,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over `series` and return (X, y) pairs.

    Args:
        series: 1-D or 2-D array of shape (T,) or (T, F).
        window_size: Number of past timesteps used as input.
        horizon: How many steps ahead to predict (y is the value at t+horizon).

    Returns:
        X: (N, window_size) or (N, window_size, F)
        y: (N,) scalar targets
    """
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size + horizon - 1] if series.ndim == 1
                 else series[i + window_size + horizon - 1, 3])  # Close index
    return np.array(X), np.array(y)


def make_multi_step_windows(
    series: np.ndarray,
    window_size: int = 30,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return windows where y is a sequence of k future Close prices."""
    X, y = [], []
    close_idx = 3  # index of Close in OHLCV
    for i in range(len(series) - window_size - k + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size : i + window_size + k, close_idx]
                 if series.ndim == 2 else series[i + window_size : i + window_size + k])
    return np.array(X), np.array(y)


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
) -> tuple:
    """Split (X, y) chronologically into train / val / test — no shuffling.

    Ratios apply sequentially: test_ratio from the end, val_ratio from what remains.
    """
    n = len(X)
    n_test = int(n * test_ratio)
    n_val  = int((n - n_test) * val_ratio)

    X_test,  y_test  = X[-n_test:],               y[-n_test:]
    X_val,   y_val   = X[-(n_test + n_val):-n_test], y[-(n_test + n_val):-n_test]
    X_train, y_train = X[:-(n_test + n_val)],     y[:-(n_test + n_val)]

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_windows(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample MinMax normalization over the window axis.

    Each window is normalized independently using its own min and max so that
    the model sees relative price movements rather than absolute levels.

    Returns:
        X_norm: normalized windows
        mins:   per-sample min, shape (N, 1) or (N, 1, 1)
        maxs:   per-sample max, same shape
    """
    if X.ndim == 2:  # (N, T)
        mins = X.min(axis=1, keepdims=True)
        maxs = X.max(axis=1, keepdims=True)
    else:            # (N, T, F)
        mins = X.min(axis=(1, 2), keepdims=True)
        maxs = X.max(axis=(1, 2), keepdims=True)

    X_norm = (X - mins) / (maxs - mins + 1e-8)
    return X_norm, mins, maxs


def denormalize(y_norm: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Reverse per-sample MinMax normalization on scalar targets."""
    return y_norm * (maxs.squeeze() - mins.squeeze() + 1e-8) + mins.squeeze()
