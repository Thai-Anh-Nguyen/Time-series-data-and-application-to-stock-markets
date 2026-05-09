# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS313 Deep Learning for AI final project (Spring 2026). Applies deep learning to time-series financial data for stock market analysis and prediction across Nasdaq and Vietnamese markets.

## Running the Notebook

```bash
jupyter notebook notebook.ipynb
```

## Data

**Nasdaq data** (`data_nasdaq_csv/csv/`): ~2000+ tickers, CSV format with columns `Date, Low, Open, Volume, High, Close, Adjusted Close`.

**Vietnam data** (`data-vn-20230228/`):
- `stock-historical-data/` — per-ticker CSVs with `Open, High, Low, Close, Volume, TradingDate`
- `financial-ratio/` — per-ticker financial ratios (P/E, ROE, ROA, etc.)
- `industry-analysis/` — sector/industry metrics per ticker
- `ticker-overview.csv` — ~1629 tickers with exchange, industry, company metadata
- `companies.csv` — company details

**Reference implementation**: `sample-code-APPL/sample-code-APPL/final-project-sample-code.ipynb` — end-to-end AAPL prediction using Conv1D with windowing, per-sample MinMax normalization, and MSE evaluation.

## Architecture & Methodology

### Data pipeline pattern (from sample code)
1. Load CSV → DataFrame
2. Construct sliding windows (`window_size=30` days) over a single price feature
3. Chronological split via `train_test_split(..., shuffle=False)`: 80% train → 80/20 → train/val, 20% test
4. Per-sample MinMax normalization (normalize each window independently using its own min/max)
5. Denormalize predictions before evaluation/visualization

### Required modeling constraints
- **Chronological splits only** — no shuffling; order is Train → Val → Test
- **Time-series CV only** — rolling window or expanding window; no standard k-fold
- **Core architectures**: LSTM, GRU, or Transformer (Conv1D used in sample as baseline)

### Task breakdown
| Task | Type | Metrics |
|------|------|---------|
| Task 1 | Nasdaq price prediction (multi-feature, nth-day, k-day ahead) | MAE, RMSE, MSE |
| Task 2 | Vietnam price prediction (same structure) | MAE, RMSE, MSE |
| Task 3 | Trading signal classification (buy/sell) for Vietnam | Accuracy, Precision, Recall, F1 |
| Task 4 | Portfolio construction & risk management | Return, Sharpe Ratio, Max Drawdown |
| Task 5 | (Extra credit) REST API + SaaS UI + Airflow/Airbyte/dbt pipeline | — |

### Key implementation notes
- "nth-day prediction": predict a specific future day (e.g., day N+7)
- "k-day prediction": predict a sequence of k future days (multi-step output)
- Multi-feature input: use all available OHLCV features, not just a single price column
- Vietnam financial ratios and industry data can be used as additional features for Tasks 2–4
