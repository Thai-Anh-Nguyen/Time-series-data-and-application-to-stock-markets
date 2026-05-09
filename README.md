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

## Setup

**Requirements:** Python 3.9+, TensorFlow 2.x, Scikit-learn, Pandas, NumPy, Matplotlib

```bash
pip install -r requirements.txt
jupyter notebook 240039-project-notebook.ipynb
```

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

*(To be completed)*

## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [Airbyte](https://airbyte.com/)
- [dbt](https://docs.getdbt.com/)
