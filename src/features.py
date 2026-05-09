"""Feature engineering: OHLCV preparation, technical indicators, Vietnam extras."""

import numpy as np
import pandas as pd
from pathlib import Path


def ohlcv_array(df: pd.DataFrame, date_col: str = "Date") -> np.ndarray:
    """Extract OHLCV as a float array of shape (T, 5), ordered Open/High/Low/Close/Volume."""
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return df[cols].astype(float).values


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log return and percentage return columns based on Close."""
    df = df.copy()
    df["pct_return"]  = df["Close"].pct_change()
    df["log_return"]  = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna().reset_index(drop=True)


def add_moving_averages(df: pd.DataFrame, windows: list[int] = [5, 10, 20]) -> pd.DataFrame:
    """Add simple moving averages of Close for each window size."""
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
    return df.dropna().reset_index(drop=True)


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI) of Close."""
    df = df.copy()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df.dropna().reset_index(drop=True)


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Band upper/lower bounds and %B position."""
    df   = df.copy()
    roll = df["Close"].rolling(window)
    df["bb_mid"]   = roll.mean()
    df["bb_upper"] = df["bb_mid"] + num_std * roll.std()
    df["bb_lower"] = df["bb_mid"] - num_std * roll.std()
    df["bb_pct_b"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)
    return df.dropna().reset_index(drop=True)


def load_financial_ratios(ticker: str, data_dir: str = "Vietnam data/financial-ratio") -> pd.DataFrame:
    matches = list(Path(data_dir).glob(f"{ticker}-*-Finance.csv"))
    if not matches:
        return pd.DataFrame()
    return pd.read_csv(matches[0])


def load_industry(ticker: str, data_dir: str = "Vietnam data/industry-analysis") -> pd.DataFrame:
    matches = list(Path(data_dir).glob(f"{ticker}*.csv"))
    if not matches:
        return pd.DataFrame()
    return pd.read_csv(matches[0])


def build_vn_feature_matrix(
    price_df: pd.DataFrame,
    ratio_df: pd.DataFrame | None = None,
    technical: bool = True,
) -> pd.DataFrame:
    """Combine price, technical indicators, and financial ratios into one DataFrame."""
    df = price_df.rename(columns={"TradingDate": "Date"}).copy()

    if technical:
        df = add_returns(df)
        df = add_moving_averages(df)
        df = add_rsi(df)
        df = add_bollinger_bands(df)

    if ratio_df is not None and not ratio_df.empty:
        # Financial ratios are typically quarterly — forward-fill to daily
        ratio_df = ratio_df.sort_values("yearReport") if "yearReport" in ratio_df.columns else ratio_df
        for col in ["priceToEarning", "priceToBook", "roe", "roa"]:
            if col in ratio_df.columns:
                df[col] = ratio_df[col].iloc[-1]  # latest available value

    return df.reset_index(drop=True)


def make_classification_labels(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0,
) -> pd.Series:
    """Binary buy/sell labels: 1 if future return > threshold, else 0."""
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    return (future_return > threshold).astype(int)
