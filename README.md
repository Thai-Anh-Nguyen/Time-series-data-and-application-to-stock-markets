# DL4AI Final Project — Time-Series Stock Market Prediction

> CS313 Deep Learning for Artificial Intelligence | Spring 2026

## Overview

This project applies deep learning techniques to time-series financial data for stock market analysis and prediction. It covers price forecasting for both Nasdaq and Vietnamese markets, trading signal identification, and portfolio construction with risk management.

## Project Structure

```
DL4AI-240039-project/
├── 240039-project-notebook.ipynb   # Main notebook with all tasks
├── 240039-project-report.pdf       # Written report (2000+ words)
├── README.md
├── data/
│   ├── nasdaq/
│   │   └── csv/                         # Nasdaq historical stock prices
│   └── vietnam/
│       ├── stock-historical-data/       # Vietnam historical stock prices
│       ├── dividend-history/
│       ├── financial-ratio/
│       ├── industry-analysis/
│       ├── companies.csv
│       └── ticker-overview.csv
├── models/                              # Saved/exported model files
├── src/                                 # Reusable modules and utilities
│   ├── preprocessing.py
│   ├── features.py
│   └── evaluation.py
└── requirements.txt
```

## Tasks

| Task | Description | Weight |
|------|-------------|--------|
| Task 1 | Nasdaq stock price prediction (multi-feature, nth-day, k-day) | 15% |
| Task 2 | Vietnam stock price prediction (multi-feature, nth-day, k-day) | 15% |
| Task 3 | Trading signal identification (buy/sell) for Vietnam market | 20% |
| Task 4 | Portfolio composition, risk management, and optimization | 30% |
| Task 5 | *(Extra credit)* Model deployment, SaaS, AI engineering workflow | 30% |
| Task 6 | Report, GitHub repository, and README | 20% |

## Setup

### Requirements

- Python 3.9+
- TensorFlow 2.x
- Scikit-learn
- Pandas, NumPy, Matplotlib

### Installation

```bash
git clone https://github.com/<your-username>/DL4AI-<StudentID>-project.git
cd DL4AI-<StudentID>-project
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook <StudentID>-project-notebook.ipynb
```

## Methodology

### Data Splitting

All splits are **chronological** — no shuffling. Order: Train → Validation → Test, where the test set represents future observations.

### Cross-Validation

Time-series-aware methods only: **rolling window** or **expanding window** cross-validation. Standard k-fold is not used.

### Models

Core architecture: deep learning models (LSTM, GRU, or Transformer-based). Classical ML and statistical methods may be used as baselines or supplementary components.

### Evaluation Metrics

- **Regression** (Tasks 1–2): MAE, RMSE, MSE
- **Classification** (Task 3): Accuracy, Precision, Recall, F1
- **Portfolio** (Task 4): Return, Sharpe Ratio, Maximum Drawdown

## Extra Credit (Task 5)

- **Task 5.1** — Model deployed as a REST API using TensorFlow Serving or FastAPI
- **Task 5.2** — Web-based SaaS interface; inference via TensorFlow.js or API calls
- **Task 5.3** — Automated pipeline using Airflow, Airbyte, dbt, and SQL/MongoDB

## Results

*(To be filled in as tasks are completed)*

## References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Airbyte](https://airbyte.com/)
- [dbt](https://docs.getdbt.com/)
