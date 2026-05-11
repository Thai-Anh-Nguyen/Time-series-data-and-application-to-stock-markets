# DL4AI Final Project — Time-Series Stock Market Prediction

> CS313 Deep Learning for Artificial Intelligence | Spring 2026 | Student ID: 240039

## Overview

This project applies deep learning to time-series financial data for stock market analysis and prediction. It covers price forecasting for both Nasdaq and Vietnamese markets, trading signal identification, and portfolio construction with risk management.

## Project Structure

```
├── 240039-project-notebook.ipynb   # Main notebook (all tasks)
├── 240039-project-report.pdf       # Written report (2000+ words)
├── Final-project-DL4AI.pdf         # Assignment specification
├── README.md
├── requirements.txt
├── document.md                     # Long-form pipeline reference (every modeling decision)
├── Local variable.yml              # Grep-friendly index of constants / dicts / helpers
├── CLAUDE.md                       # Editing-orientation notes
├── src/
│   ├── preprocessing.py            # Windowing, splitting, normalization
│   ├── features.py                 # Technical indicators, feature engineering
│   └── evaluation.py               # MAE/RMSE/MSE, classification, portfolio metrics
├── models/                         # Saved model weights and checkpoints
├── Nasdaq data/
│   └── csv/                        # 1564 Nasdaq tickers (OHLCV + Adj. Close)
├── Vietnam data/
│   ├── stock-historical-data/      # 1629 Vietnam tickers (OHLCV + TradingDate)
│   ├── financial-ratio/            # Per-ticker P/E, ROE, ROA, etc.
│   ├── industry-analysis/          # Sector/industry metrics
│   ├── dividend-history/           # Per-ticker dividend records
│   ├── ticker-overview.csv         # Exchange, industry, company metadata
│   ├── companies.csv               # Company details
│   └── crawl-vn-data.ipynb         # Data collection notebook
└── sample-code-APPL/
    └── sample-code-APPL/
        └── final-project-sample-code.ipynb   # Reference: AAPL Conv1D baseline
```

## Tasks

| Task | Description | Metrics |
|------|-------------|---------|
| Task 1 | Nasdaq price prediction (multi-feature, nth-day, k-day ahead) | MAE, RMSE, MSE |
| Task 2 | Vietnam price prediction (multi-feature, nth-day, k-day ahead) | MAE, RMSE, MSE |
| Task 3 | Trading signal identification (buy/sell) for Vietnam market | Accuracy, Precision, Recall, F1 |
| Task 4 | Portfolio construction, risk management, and optimization | Return, Sharpe Ratio, Max Drawdown |
| Task 5 *(extra credit)* | Model deployment, SaaS interface, AI engineering pipeline | — |

## Running the project

### 1. Environment

**Requirements:** Python 3.9+, TensorFlow 2.15+, scikit-learn, pandas, NumPy, SciPy, Matplotlib, seaborn, Jupyter.

A clean virtual environment is recommended (the project pins TensorFlow 2.15+, which conflicts with some older system installs):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only TensorFlow is sufficient — every model in the notebook trains in well under a minute per ticker on CPU. A GPU is not required.

### 2. Data layout

Before running the notebook, confirm the data folders are populated:

```
Nasdaq data/csv/<TICKER>.csv                            # e.g. AAPL.csv, MSFT.csv, NVDA.csv
Vietnam data/stock-historical-data/<TICKER>-VNINDEX-History.csv   # e.g. VCB-VNINDEX-History.csv
```

The notebook reads three Nasdaq tickers (AAPL / MSFT / NVDA) and six Vietnamese tickers (VCB / HPG / FPT / VNM / MSN / MWG). Other tickers in the data folders are not used. If the Vietnam folder is empty, regenerate it via [Vietnam data/crawl-vn-data.ipynb](Vietnam%20data/crawl-vn-data.ipynb).

### 3. Launch the notebook

```bash
jupyter notebook 240039-project-notebook.ipynb
```

Then run cells **top-to-bottom**. Tasks depend on each other through in-memory state (result dicts), so don't skip ahead.

### 4. Task-by-task run order

| Task | What it does | Produces | Depends on |
|------|--------------|----------|------------|
| §0   | Imports, checkpoint utilities, raw-frame loaders | `RAW_FRAMES`, `VN_RAW_FRAMES`, `add_technical_features` | — |
| §1.1 / §1.2 / §1.3 | Nasdaq next-day / kᵗʰ-day / k-consecutive-day forecasts | `results_k1`, `results_kth`, `results_kday` | §0 |
| §2.1 / §2.2 / §2.3 | Vietnam mirrors of §1 | `results_vn_k1`, `results_vn_kth`, **`results_vn_kday`** | §0 |
| §3.1 / §3.2 | Buy / sell binary classifiers (Vietnam) | **`results_vn_sell`**, `results_vn_buy` | §2.1 (reuses pipeline) |
| §4.1 / §4.2 / §4.3 | Profitability, risk, portfolio backtest | `test_panel`, `risk_wide`, `portfolio_results` | **§2.3** (k=7 forecast) + **§3.2** (sell probability) |

Total wall-clock on CPU is roughly **15–20 minutes** for a full re-run from scratch (≈ 50 LSTM training loops at 10 epochs each).

### 5. Re-running without retraining

Every training cell uses `train_with_checkpoint(...)` followed by `load_model(...)`. Trained weights are persisted to `models/<task>/<name>.keras`. On a re-run:

- Re-execute the training cell — `ModelCheckpoint` will overwrite the file with the best-by-`val_loss` epoch, then the `load_model` line restores that best version.
- Or comment out the `model.fit` call and keep only `model = load_model(...)` to skip training entirely. The result dicts (`results_*`) and downstream cells will still populate correctly.

Task 4 itself trains **no new models** — it consumes `results_vn_kday[(t, 7)]` and `results_vn_sell[t]`. Re-running Task 4 after editing scoring parameters only requires re-executing §4.1 onwards.

### 6. Reference documentation

- **[document.md](document.md)** — long-form pipeline reference: every helper, every modeling decision, look-ahead audit for Task 4.
- **[Local variable.yml](Local%20variable.yml)** — grep-friendly index of every constant, dict, helper, and checkpoint path defined by the notebook.
- **[CLAUDE.md](CLAUDE.md)** — short orientation for editing the codebase.

## Methodology

### Data Splitting
All splits are **chronological** — no shuffling. Order: Train → Validation → Test.

### Cross-Validation
Time-series-aware only: **rolling window** or **expanding window**. Standard k-fold is not used.

### Models
Core architectures: **LSTM, GRU, or Transformer**. Conv1D used in reference baseline.

### Normalization
Per-sample MinMax normalization — each window normalized independently using its own min/max. Predictions are denormalized before evaluation.

### Evaluation Metrics
- **Regression** (Tasks 1–2): MAE, RMSE, MSE
- **Classification** (Task 3): Accuracy, Precision, Recall, F1-score
- **Portfolio** (Task 4): Annualized Return, Sharpe Ratio, Maximum Drawdown

## Extra Credit (Task 5)

- **Task 5.1** — Model deployed as a REST API (TensorFlow Serving or FastAPI)
- **Task 5.2** — Web-based SaaS UI; inference via TensorFlow.js or API
- **Task 5.3** — Automated data pipeline using Airflow, Airbyte, dbt, and SQL/MongoDB

## Results



## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [Airbyte](https://airbyte.com/)
- [dbt](https://docs.getdbt.com/)
