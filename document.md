# Project pipeline — detailed reference

Living companion to [240039-project-notebook.ipynb](240039-project-notebook.ipynb). The notebook holds the runnable code with concise markdown; this file is the long-form explanation of **why** each step is the way it is, what the numbers mean, and where the gotchas are. Keep it in sync as new tasks are added.

Quick links: [variable namespace map](Local%20variable.yml) · [checkpoints folder](models/) · [project instructions](CLAUDE.md).

---

## Reading order

| In the notebook | In this document |
|---|---|
| `# Loading libraries` | [§0 Environment & imports](#0-environment--imports) |
| `### Checkpoint utilities` | [§0.1 Checkpointing](#01-checkpointing) |
| `# TASK 1` → `1.1 Multiple-feature extension` | [§1.1 Multiple-feature next-day forecast (AAPL)](#11-multiple-feature-next-day-forecast) |
| `### 1.1 extension — apply the same pipeline to MSFT and NVDA` | [§1.1.h Multi-ticker extension](#11h-multi-ticker-extension) |
| `1.2 kᵗʰ day forecast` | [§1.2 kᵗʰ-day-ahead forecast](#12-kᵗʰ-day-ahead-forecast) |
| `1.3 k consecutive days forecast` | [§1.3 k consecutive days forecast](#13-k-consecutive-days-forecast) |
| `# TASK 2` → `2.1 Vietnam multi-feature next-day forecast` | [§2.1 Vietnam next-day forecast (6 tickers)](#21-vietnam-next-day-forecast) |
| `2.2 Vietnam kᵗʰ day forecast` | [§2.2 Vietnam kᵗʰ-day-ahead forecast](#22-vietnam-kᵗʰ-day-ahead-forecast) |
| `2.3 Vietnam k consecutive days forecast` | [§2.3 Vietnam k consecutive days forecast](#23-vietnam-k-consecutive-days-forecast) |
| `# TASK 3` → `3.1 Buying signal identification` | [§3.1 Buy signal (binary classification)](#31-buy-signal) |
| `3.2 Selling signal identification` | [§3.2 Sell signal (binary classification)](#32-sell-signal) |
| `# TASK 4` → `4.1 Profitable stock selection` | [§4.1 Profitability score](#41-profitability-score) |
| `4.2 Risk management` | [§4.2 Risk score (cross-sectional blend)](#42-risk-score) |
| `4.3 Portfolio composition — prudent vs aggressive investor` | [§4.3 Portfolio composition & backtest](#43-portfolio-composition--backtest) |

---

## 0. Environment & imports

| Library | Used for |
|---|---|
| `numpy`, `pandas` | Array & dataframe handling |
| `matplotlib` | Plots (split visualisation, training curves, predicted-vs-real) |
| `tensorflow.keras` | `Sequential`, `Input`, `LSTM`, `Dense` — model definition |
| `tensorflow.keras.callbacks.ModelCheckpoint` | Per-epoch best-by-`val_loss` checkpointing |
| `tensorflow.keras.models.load_model` | Reload the best epoch after training |
| `pathlib.Path` | Path arithmetic for the checkpoint tree |
| `sklearn.model_selection.train_test_split` | Chronological split via `shuffle=False` |
| `sklearn.metrics` | `mean_squared_error`, `mean_absolute_error` for test-set evaluation |

No GPU is assumed — each LSTM trains in roughly a second per epoch on CPU.

## 0.1 Checkpointing

Every trained model is saved under `models/<task>/<name>.keras`. After 1.1 and 1.2 run end-to-end the tree looks like:

```
models/
├── task1.1/
│   ├── AAPL_lstm.keras
│   ├── MSFT_lstm.keras
│   └── NVDA_lstm.keras
├── task1.2/
│   ├── AAPL_k3.keras
│   ├── AAPL_k7.keras
│   ├── MSFT_k3.keras
│   ├── MSFT_k7.keras
│   ├── NVDA_k3.keras
│   └── NVDA_k7.keras
└── task1.3/
    ├── AAPL_k3.keras
    ├── AAPL_k7.keras
    ├── MSFT_k3.keras
    ├── MSFT_k7.keras
    ├── NVDA_k3.keras
    └── NVDA_k7.keras
```

Two helpers do the work:

- `checkpoint_path(task, name)` builds the `models/<task>/<name>.keras` path and creates the parent directory on demand.
- `train_with_checkpoint(model, train, val, save_path, ...)` wraps `model.fit` with a `ModelCheckpoint(monitor='val_loss', save_best_only=True)` callback so the file on disk is the **best epoch**, not the last. Every training cell follows the same pattern:

  ```python
  history = train_with_checkpoint(model, tr, va, save_path=checkpoint_path('task1.1', 'AAPL_lstm'))
  model   = load_model(checkpoint_path('task1.1', 'AAPL_lstm'))  # use best weights downstream
  ```

Re-running a training cell overwrites the file with a fresh best — there's no versioning. If you want to keep an experiment, copy the `.keras` file out of `models/` by hand.

---

## 1.1 Multiple-feature next-day forecast

**Goal (from the spec):** extend the sample notebook (single Adjusted-Close feature, Conv1D, AAPL only) to use **multiple features** and a recurrent architecture, and run it on more than one Nasdaq ticker.

### 1.1.a Data sources

Three CSVs loaded directly from `Nasdaq data/csv/`:

| Ticker | Rows | Date range (approx.) |
|---|---|---|
| AAPL | 10 590 | 1980-12-12 → present |
| MSFT |  9 264 | 1986-03 → present |
| NVDA |  6 013 | 1999-01 → present |

Raw columns: `Date, Low, Open, Volume, High, Close, Adjusted Close`. The `Date` column is a string in `dd-mm-yyyy` format — the pipeline converts it to `datetime64` immediately so sliding windows respect calendar order.

### 1.1.b Feature engineering (`add_technical_features`)

Seven engineered features are added on top of the six raw OHLCV+AdjClose columns. **All are causal** (each value at time *t* only depends on values at times ≤ *t*), so computing them across the whole series before splitting does **not** leak future information into the training set.

| Feature | Formula | Why we add it |
|---|---|---|
| `daily_return` | `Close.pct_change()` | Magnitude of day-over-day moves; centres around 0 |
| `MA5` | rolling mean of `Close`, window 5 | Short-term trend; smooths daily noise |
| `MA20` | rolling mean of `Close`, window 20 | ~ Monthly trend; the workhorse moving average |
| `RSI` (14) | `100 − 100/(1 + avg_gain/avg_loss)` over 14 days | Momentum oscillator bounded in [0, 100]. > 70 ≈ overbought, < 30 ≈ oversold |
| `MACD` | `EMA12(Close) − EMA26(Close)` | Long-vs-short EMA spread — captures trend changes |
| `MACD_signal` | `EMA9(MACD)` | Smoother of MACD; sign changes ≈ entry/exit signals |
| `volatility` | rolling std of `daily_return`, 20 days | Risk / regime indicator |

Rows with `NaN` (the first 20 rows, where rolling windows aren't full yet) are dropped via `.dropna()` after feature construction. The final input has **13 features per timestep**.

### 1.1.c Sliding-window construction

For each ticker we build labelled `(X, y)` pairs:

- `window_size = 30`: each `X[i]` is a 30-day matrix of shape `(30, 13)`
- `horizon = 1`: each `y[i]` is the **Adjusted Close on the day immediately after the window**
- Stride 1 → maximum data utilisation

Index arithmetic (used identically by 1.2 with `k = horizon`):

```
for i in range(len(arr) - window_size - horizon + 1):
    X.append(arr[i : i + window_size])           # 30 days of features
    y.append(label[i + window_size + horizon-1]) # the (horizon)-th day after the window
```

`horizon = 1` → label is `arr[i + 30]`, the very next day.

### 1.1.d Chronological 70 / 15 / 15 split

The project spec **forbids shuffling** — finance data is non-stationary and shuffling causes massive label leakage (training on the future, testing on the past). We use `sklearn.model_selection.train_test_split` with `shuffle=False`, called twice:

1. Carve off the last 15% chronologically → `X_test`, `y_test`.
2. From the remaining 90%, carve off the last `15 / 85 ≈ 17.6%` → `X_val`, `y_val`.
3. What's left is `X_train`, `y_train`.

Result for AAPL: **7 378 train / 1 581 val / 1 581 test** windows. The notebook plots the AAPL Adjusted-Close trend with vertical lines at the two split boundaries so the chronology is visually obvious.

### 1.1.e Per-window MinMax normalisation

This matches the sample notebook's style and is **the** subtle bit of the pipeline.

For each window `i` and each feature `f`:

```
fmin_{i,f} = X[i, :, f].min()    # over the 30 timesteps
fmax_{i,f} = X[i, :, f].max()
X_norm[i, :, f] = (X[i, :, f] - fmin_{i,f}) / (fmax_{i,f} - fmin_{i,f} + 1e-8)
```

So every individual 30-day window is scaled into `[0, 1]` using **its own** statistics. There is **no** global scaler, and no leakage between windows.

For the **label** (Adjusted Close at *t + horizon*) we use the **same window's min/max of the Adjusted Close column**:

```
y_norm[i] = (y[i] - lmin_i) / (lmax_i - lmin_i + 1e-8)
```

Two important consequences:

1. **`y_norm` is not bounded in [0, 1]**. If the future day's price exceeds the window's range (e.g., AAPL hitting an all-time high right after a flat month), `y_norm > 1`. Likewise `y_norm < 0` is possible. The sanity-check cell in the notebook reports e.g. `y ∈ [−1.6, 2.0]` for AAPL — that's expected, not a bug.
2. **Denormalisation is per-window.** We keep `label_min` and `label_max` arrays of shape `(N,)` alongside each split. To recover dollars from a prediction:

```
y_pred_dollars = y_pred_norm * (label_max - label_min + 1e-8) + label_min
```

The `denormalize_label` helper does exactly this; we use it everywhere we compare against real prices.

The pipeline returns three `Split` namedtuples with fields `.X, .y, .label_min, .label_max`.

### 1.1.f Model architecture

A deliberately minimal LSTM, chosen to (a) satisfy the spec's "core architectures: LSTM, GRU, or Transformer" requirement and (b) stay directly comparable to the sample notebook's Conv1D baseline:

```python
Sequential([
    Input(shape=(30, 13)),  # window_size × n_features
    LSTM(64),
    Dense(1),               # regression head — single scalar output
])
```

- **Optimiser:** Adam (Keras default learning rate `1e-3`)
- **Loss / metric:** MSE on the normalised target
- **Epochs:** 10
- **Batch size:** 64
- **Validation:** explicit `validation_data=(AAPL_val.X, AAPL_val.y)` — never `validation_split`, because Keras' default `validation_split` would take the **last** windows of the training set and we want our validation to be the chronologically separated `_val` arrays the pipeline produced.

### 1.1.g Evaluation

Predictions are made on `AAPL_test.X`, then **denormalised** to dollars before computing metrics. We report:

| Metric | Value (10 epochs) | What it means |
|---|---|---|
| MSE (normalised) | 0.02466 | Direct comparison to sample's reported 0.033 — we do meaningfully better |
| MSE ($²) | 4.0539 | Squared error in actual dollars |
| RMSE ($) | 2.0134 | "Typical" prediction error in dollars |
| MAE ($) | 1.2391 | Average absolute error in dollars |

The final cell plots predicted vs real Adjusted Close on real calendar dates for all 1 581 test days. The x-axis is derived by walking the original `feat_aapl` DataFrame at offset `N_train + N_val + window_size + horizon − 1` — same formula used (with `k` in place of `horizon`) in 1.2.

### 1.1.h Multi-ticker extension

The pipeline is ticker-agnostic, so applying it to MSFT and NVDA is just: reuse the splits that `build_pipeline` already built in [cell 63ec1fdb](240039-project-notebook.ipynb), build a fresh `build_lstm(input_shape)`, train with `train_with_checkpoint`, reload the best epoch.

Two helpers are introduced here (and reused by 1.2):

- `build_lstm(input_shape)` — the `LSTM(64) → Dense(1)` model with Adam + MSE, factored out so 1.1 and 1.2 share the same definition.
- `evaluate_test(model, test)` — predict → denormalise → return `(y_real, y_pred, metrics_dict)`. Defined here (rather than later in 1.2) because the cross-ticker comparison table needs it.

Result containers are organised so 1.2 can extend them naturally:

- `TICKERS = ['AAPL', 'MSFT', 'NVDA']` — canonical iteration order.
- `SPLITS_K1` — `dict[ticker → (train, val, test)]`, indexed by string.
- `results_k1` — `dict[ticker → {model, history, train, val, test}]`. Populated in three places: the `'AAPL'` entry is filled from 1.1's `AAPL_LSTM_model`, then the loop adds `'MSFT'` and `'NVDA'`.
- `MSFT_LSTM_model`, `NVDA_LSTM_model` — re-exported pointers into `results_k1` so anyone grepping for `AAPL_LSTM_model` can find the parallel names.

**No per-ticker plots.** AAPL's predicted-vs-real plot in 1.1.g already demonstrates the model's behaviour qualitatively; the cross-ticker comparison table is the report-relevant artifact. We deliberately do **not** mirror the plot for MSFT and NVDA — the table conveys "how well did each ticker do" without padding the notebook.

**Cross-ticker comparison.** Dollar metrics (`MSE ($²)`, `RMSE ($)`, `MAE ($)`) are not directly comparable across tickers — NVDA's prices live in a different range than AAPL's, so equal-skill models will show very different dollar errors. The **normalised MSE** is on the per-window MinMax scale and is the right column to compare across tickers.

---

## 1.2 kᵗʰ-day-ahead forecast

**Goal (from the spec):** instead of predicting tomorrow, predict the price **k^th days ahead** (e.g., 3rd or 7th day after the input window).

### 1.2.a Why this is almost free

`build_pipeline(..., horizon=k)` from 1.1 already parameterises the forecast horizon — the label at index `i` is `label[i + window_size + k - 1]`. So 1.2 is just "run the same pipeline with different `horizon` values". We deliberately keep:

- Same feature set (13 columns)
- Same `window_size = 30`
- Same split ratios (70 / 15 / 15, no shuffling)
- Same per-window MinMax normalisation
- Same LSTM(64) → Dense(1) architecture (`build_lstm`, defined in 1.1.h)
- Same Adam / MSE / 10 epochs / batch 64
- Same checkpointing (`train_with_checkpoint`)

…so any change in test metrics is **purely a function of `(ticker, k)`**, not of the model or training recipe.

### 1.2.b Training loop over {AAPL, MSFT, NVDA} × {3, 7}

The notebook loops over `TICKERS × K_VALUES = 6` combinations, calling `build_pipeline(RAW_FRAMES[ticker], ..., horizon=k)` for each, then training a fresh LSTM with checkpointing to `models/task1.2/<TICKER>_k<k>.keras`. Results are stored in a dict keyed by `(ticker, k)`:

```python
results_kth = {
    ('AAPL', 3): {'model': …, 'history': …, 'train': …, 'val': …, 'test': …},
    ('AAPL', 7): {…},
    ('MSFT', 3): {…},
    ('MSFT', 7): {…},
    ('NVDA', 3): {…},
    ('NVDA', 7): {…},
}
```

Train sizes shift very slightly with `k` because `len(arr) - window_size - horizon + 1` shrinks by `k − 1` windows total — at `k = 7` AAPL has 7 373 train windows vs 7 378 at `k = 1`. Negligible.

### 1.2.c Comparison table — all (ticker, k) in one place

The k = 1 row for each ticker comes from `results_k1[ticker]` (1.1 extension), and k ∈ {3, 7} from `results_kth[(ticker, k)]`. One nested loop prints a block per ticker. Reference values from a single training run (numbers will fluctuate run-to-run since we don't seed):

| ticker | k | MSE (norm) | MSE ($²) | RMSE ($) | MAE ($) |
|---|---:|---:|---:|---:|---:|
| AAPL | 1 | ≈ 0.025 | ≈ 4.05 | ≈ 2.01 | ≈ 1.24 |
| AAPL | 3 | ≈ 0.077 | ≈ 11.2 | ≈ 3.34 | ≈ 2.11 |
| AAPL | 7 | ≈ 0.185 | ≈ 25.1 | ≈ 5.01 | ≈ 3.29 |
| MSFT | 1 | — | — | — | — |
| MSFT | 3 | — | — | — | — |
| MSFT | 7 | — | — | — | — |
| NVDA | 1 | — | — | — | — |
| NVDA | 3 | — | — | — | — |
| NVDA | 7 | — | — | — | — |

(Run the notebook to populate the MSFT and NVDA rows; AAPL is given for sanity-checking against the 1.1 baseline.) Error grows roughly monotonically with `k` for each ticker — the further out we forecast, the less the recent 30-day window's information constrains the target. By k = 7, RMSE is typically **~2.5×** the k = 1 value within the same ticker.

### 1.2.d Plotting on real dates (AAPL only — the off-by-one trap)

Only AAPL is plotted. We deliberately don't mirror the predicted-vs-real plot for MSFT and NVDA — the table already conveys the cross-ticker story, and the *new* axis 1.2 introduces (different k values) is fully covered by the two AAPL panels.

Each panel plots predicted vs real Adjusted Close on the calendar dates of the test labels. The dates have to be looked up in `feat_aapl` (the feature-engineered AAPL DataFrame) at the right offset:

```
r              = results_kth[('AAPL', k)]
test_start_idx = r['train'].X.shape[0] + r['val'].X.shape[0]
label_offset   = window_size + k - 1
dates          = feat_aapl['Date'].iloc[test_start_idx + label_offset
                                       : test_start_idx + label_offset + len(y_pred)]
```

**Don't reuse the 1.1 split sizes.** `N_train + N_val` changes very slightly with `k` (because total windows shrink), so the test-window offset must come from the **per-(ticker, k) pipeline's own** `r['train']` and `r['val']`, not from the `AAPL_train` / `AAPL_val` variables from 1.1.

---

## 1.3 k consecutive days forecast

**Goal (from the spec):** instead of predicting a *single* future day, predict the **next k days as a sequence**. For k = 3 the model outputs `[day+1, day+2, day+3]`; for k = 7 it outputs `[day+1, …, day+7]`.

### 1.3.a What changes vs 1.2

The 1.2 pipeline produces a **scalar** label (the kᵗʰ future day). 1.3 produces a **length-k vector** label (all k future days). Everything else is held fixed so any change in error is a function of the new label structure, not of hyperparameters:

| Component | 1.2 | 1.3 |
|---|---|---|
| Label | scalar `label[i + window_size + k − 1]` | vector `label[i + window_size : i + window_size + k]`, shape `(k,)` |
| Model head | `Dense(1)` | `Dense(k)` |
| Loss | MSE on scalar | MSE on vector (Keras averages over `k` automatically) |
| Norm/denorm | per-window MinMax (Adj Close min/max) | **same** stats, broadcast across the k label components |
| Window count per ticker | `len(arr) − window_size − k + 1` | identical |
| Train/val/test ratios | 70 / 15 / 15, chronological | identical |

The new helpers live in cell `99e59430`:

- `MultiStepSplit` — namedtuple like `Split`, but `.y` has shape `(N, k)`.
- `build_pipeline_multistep(df, name, k, ...)` — windowed features + vector labels + per-window MinMax (using the input window's Adj Close min/max for *all* k components, so denormalisation needs only one `(label_min, label_max)` pair per window).
- `build_lstm_multistep(input_shape, k)` — same `LSTM(64)` trunk, but `Dense(k)` head.
- `denormalize_label_multistep(y_norm, label_min, label_max)` — invert MinMax for `(N, k)` predictions.
- `evaluate_test_multistep(model, test)` — returns `(y_real (N,k), y_pred (N,k), metrics_dict)` with both aggregate metrics and **per-step RMSE/MAE** arrays of length k.

### 1.3.b Training loop

Same shape as 1.2's loop, just calling the multi-step pipeline and head:

```python
results_kday = {}
for ticker in TICKERS:
    for k in K_VALUES:                                          # [3, 7]
        tr, va, te = build_pipeline_multistep(RAW_FRAMES[ticker], ..., k=k)
        model = build_lstm_multistep(tr.X.shape[1:], k=k)
        hist  = train_with_checkpoint(
            model, tr, va,
            save_path=checkpoint_path('task1.3', f'{ticker}_k{k}'),
        )
        model = load_model(checkpoint_path('task1.3', f'{ticker}_k{k}'))
        results_kday[(ticker, k)] = {'model': model, 'history': hist,
                                     'train': tr, 'val': va, 'test': te}
```

`results_kday` uses the same `(ticker, k)` tuple convention as `results_kth` — kept separate so 1.2 and 1.3 don't trample each other's models.

### 1.3.c Per-step error growth (table)

The single most useful artifact in 1.3 is the **per-step RMSE/MAE** array. The aggregate-over-all-steps RMSE (which the table also reports) mixes near and far horizons together; the per-step breakdown shows the error *growth curve* directly:

```
[AAPL  k=7]  MSE(norm)=…  RMSE($)=…  MAE($)=…
              step:        1        2        3        4        5        6        7
              RMSE:    a.aaa    b.bbb    c.ccc    d.ddd    e.eee    f.fff    g.ggg
              MAE :    …
```

Typical pattern (numbers vary between runs since we don't seed):

- step-1 RMSE is in the same ballpark as 1.2's k=1 baseline for the same ticker.
- step-k RMSE is in the same ballpark as 1.2's k=k single-day forecast.
- Intermediate steps interpolate, usually slightly above a straight line — error grows fastest at the start and tapers (relative to a baseline of "we already know roughly where prices will be"). This is the right shape to report.

The aggregate `MSE (norm)` column remains the only cross-ticker-comparable metric.

### 1.3.d Plotting on real dates (AAPL only — the trajectory plot)

1.3 introduces a genuinely new visual dimension: the **forecast trajectory itself**. 1.1 plots a single horizon over many dates; 1.2 plots a single horizon (per k) over many dates; 1.3 can plot a **full k-day predicted path** from a single starting date.

The plot picks six evenly-spaced starting indices in the AAPL k=7 test set and draws the 7 predicted Adj Close values against the realised 7. Date arithmetic:

```python
test_start_idx     = r['train'].X.shape[0] + r['val'].X.shape[0]
first_label_offset = window_size                 # the *first* of the k predicted days
for idx in sample_indices:
    base = test_start_idx + first_label_offset + idx
    dates = feat_aapl['Date'].iloc[base : base + k]
    ax.plot(dates, y_real[idx], ...)             # y_real[idx] has shape (k,)
    ax.plot(dates, y_pred[idx], ...)
```

Note: `first_label_offset = window_size` (no `+ k − 1`), because the *first* predicted day is `window_size` rows after the start of that window. The `+ k − 1` offset only applies when locating the kᵗʰ day (as in 1.2's plot).

MSFT and NVDA are intentionally not plotted — the per-step table already shows their cross-ticker behaviour, and the trajectory-vs-realised picture is qualitatively the same shape, just at different price levels.

---

## 2.1 Vietnam next-day forecast

**Goal (from the spec):** mirror Task 1 on the Vietnam stock dataset — multi-feature input, recurrent architecture, multiple tickers — instead of the Nasdaq one.

### 2.1.a Ticker set (6 large-cap HOSE names)

| Ticker | Sector | Rows | Date range (approx.) |
|---|---|---|---|
| VCB | Banking | 3 413 | 2009-06 — present |
| HPG | Steel / industrials | 3 809 | 2007-11 — present |
| FPT | Technology | 4 038 | 2006-12 — present |
| VNM | Consumer staples (dairy) | 4 264 | 2006-01 — present |
| MSN | Consumer / retail conglomerate | 3 322 | 2009-11 — present |
| MWG | Tech retail | 2 157 | 2014-07 — present |

These were chosen for **breadth of sector coverage** and **history depth**: all have ≥ 2 100 trading days, enough to keep the same 70 / 15 / 15 split (smallest, MWG, still yields ~1 500 train windows).

### 2.1.b Pipeline differences vs Task 1

The Vietnam CSVs have a different schema than the Nasdaq ones, but the *pipeline* is structurally identical — we thread the per-dataset feature/label column lists through the same `build_pipeline` / `build_pipeline_multistep` helpers (which now take optional `feature_cols=` / `label_col=` kwargs). Task 1 cells call without kwargs and default to the Nasdaq globals; Task 2 cells pass `VN_FEATURE_COLS` / `VN_LABEL_COL`.

| Aspect | Task 1 (Nasdaq) | Task 2 (Vietnam) |
|---|---|---|
| Raw columns | `Date, Low, Open, Volume, High, Close, Adjusted Close` | `Open, High, Low, Close, Volume, TradingDate` (+ leading unnamed index) |
| Date format | `dd-mm-yyyy` strings | `yyyy-mm-dd` strings |
| Label column | `Adjusted Close` | `Close` (no Adj Close exists for Vietnam) |
| Feature count | 13 (OHLCV + Adj Close + 7 indicators) | 12 (OHLCV + 7 indicators) |
| Window size | 30 | 30 |
| Split | 70 / 15 / 15, chronological | identical |
| Per-window MinMax | yes (using Adj Close stats for the label) | yes (using `Close` stats for the label) |
| Model | `LSTM(64) → Dense(1)` | identical (`build_lstm` reused as-is) |
| Optimiser / loss / epochs / batch | Adam / MSE / 10 / 64 | identical |
| Checkpoint subdir | `models/task1.1/` | `models/task2.1/` |

The single Vietnam-specific helper is `load_vietnam(ticker)` — it reads the CSV, drops the leading unnamed index column, renames `TradingDate → Date`, and parses the date. `add_technical_features` then works on the renamed frame without modification (it already auto-detects already-parsed datetime columns).

### 2.1.c Training & evaluation

The 2.1 training cell loops over `VN_TICKERS` and trains an LSTM per ticker. `results_vn_k1[ticker]` is the `dict[ticker → {model, history, train, val, test}]` mirror of `results_k1`.

The **cross-ticker comparison table** uses `evaluate_test` unchanged — the metric dict's keys still say `'MSE ($²)'` / `'RMSE ($)'` etc., but for the Vietnam table we relabel them as VND in the print headers (no `evaluate_test` change is required; the dict values are unit-agnostic). The dollar-style metrics aren't comparable across tickers (VCB trades ~80–100 k VND, MWG ~30–60 k VND, etc.); the **normalised MSE** is.

### 2.1.d Plotting strategy

Per the same logic as Task 1.1.h ("MSFT/NVDA not plotted; table covers them"), only **VCB** gets a predicted-vs-real Close plot. The split-boundary visualisation is also VCB only.

---

## 2.2 Vietnam kᵗʰ-day-ahead forecast

Mirrors Task 1.2. `build_pipeline(..., horizon=k)` is reused; only the value of `k` and the checkpoint subdir change (`models/task2.2/`). We train at **k ∈ {3, 7}** for all 6 tickers, store results in `results_vn_kth[(ticker, k)]`, and print a unified k ∈ {1, 3, 7} table.

The k = 1 row in the table comes from `results_vn_k1[ticker]`; k ∈ {3, 7} rows from `results_vn_kth[(ticker, k)]`. As with Nasdaq, total windows shrink by `k − 1` per (ticker, k) pipeline — negligible — but the **plot's date offset** must still come from each `(ticker, k)`'s own `r['train']` / `r['val']` rather than reusing 2.1's sizes.

Only VCB is plotted (two panels for k = 3 and k = 7); the table covers the cross-ticker comparison.

---

## 2.3 Vietnam k consecutive days forecast

Mirrors Task 1.3. `build_pipeline_multistep(..., k=k)` is reused with `feature_cols=VN_FEATURE_COLS, label_col=VN_LABEL_COL`. The model head becomes `Dense(k)`; everything else stays the same. Results live in `results_vn_kday[(ticker, k)]` (kept separate from `results_kday` so Task 1 and Task 2 don't trample each other).

The relevant report artifact remains the **per-step RMSE/MAE** array — step-1 RMSE should be close to that ticker's 2.1 (k=1) baseline, step-k RMSE close to that ticker's 2.2 (k=k) single-day-ahead forecast, intermediate steps interpolate.

VCB k = 7 gets a six-panel trajectory plot at evenly-spaced starting dates from the test set. Same offset trap as 1.3: `first_label_offset = window_size` (no `+ k − 1`) for trajectory plots.

---

## 3.1 Buy signal

**Goal (from the spec):** build a model that outputs a probability / score suggesting *now* is a good time to enter. Justify the labelling and model design. The PDF specifically asks whether manual rule-based features (SMA / MACD / RSI crossovers) should drive the labels.

### 3.1.a Label scheme — future-return threshold

Window ending at day *t* is positive (label = 1) iff

```
Close[t + h] / Close[t] − 1 > +τ
```

with `h = 5` (one trading week) and `τ = 0.02` (2 % move). The model's sigmoid output is the **probability of a 5-day-forward 2 % up-move**.

Why this scheme and not a rule-based label:

- **Rule-derived labels** (e.g., MACD crosses above signal *and* RSI < 70) reduce the task to imitating the rule. The DL classifier would learn the rule, with no value over running the rule directly.
- **Triple-barrier** (López de Prado: upper barrier `+τ` hit before lower barrier `−τ` within `h` days) is a defensible alternative — captures stop-loss / take-profit asymmetry. We mention it as an alternative in the report but stick with the fixed-horizon scheme so 3.1 and 3.2 read as clean mirrors.
- **Future-return thresholds** let the model discover *which* features (the same SMA/MACD/RSI we already include as **inputs**) actually predict the move.

`(h, τ) = (5, 0.02)` was chosen so the positive class is roughly 30–50 % on the six tickers — frequent enough that the model can't degenerate to "always predict 0", rare enough that the task isn't trivial. Positive-rate survey on the full series:

| Ticker | Rows | Buy + % |
|---|---:|---:|
| VCB | 3 388 | 30.7 % |
| HPG | 3 784 | 35.7 % |
| FPT | 4 013 | 29.6 % |
| VNM | 4 239 | 27.0 % |
| MSN | 3 297 | 30.7 % |
| MWG | 2 132 | 35.2 % |

### 3.1.b Pipeline reuse vs Task 2

Everything below the label-generation step is identical to Task 2:

| Component | Task 2 (regression) | Task 3 (classification) |
|---|---|---|
| Input window | 30 days × 12 features (`VN_FEATURE_COLS`) | identical |
| Per-window MinMax of inputs | yes | identical |
| Chronological 70 / 15 / 15 split | yes (`shuffle=False`) | identical |
| Label | scalar Close at `t+horizon`, MinMax-normalised | scalar 0 / 1, **not normalised** |
| Trunk | `LSTM(64)` | identical |
| Head | `Dense(1)` linear | `Dense(1, sigmoid)` |
| Regularisation | none | `Dropout(0.2)` between LSTM and head |
| Loss | MSE | binary cross-entropy |
| Metrics | MSE / RMSE / MAE | Accuracy / Precision / Recall / F1 / ROC-AUC |
| Checkpoints | `models/task2.*/` | `models/task3.1/<TICKER>_buy.keras` |

The new helpers live in the Task 3 helpers cell:

- `ClfSplit` — namedtuple like `Split`, with `.X` and `.y` only (no `label_min` / `label_max` — labels are 0/1, no denormalisation needed).
- `make_signal_labels(close, horizon, threshold, direction)` — vectorised label generator. NaN-pads the last `horizon` rows; pipeline drops them via the `len(arr) − window_size − horizon + 1` window cap.
- `build_pipeline_classification(df, name, horizon, threshold, direction, ...)` — Task 2 pipeline with classification labels. Accepts the same `feature_cols=` / `label_col=` kwargs as `build_pipeline`.
- `build_lstm_classifier(input_shape)` — `LSTM(64) → Dropout(0.2) → Dense(1, sigmoid)`, BCE loss, Adam optimiser. Built-in metrics `[accuracy, AUC, Precision, Recall]` for per-epoch monitoring.
- `tune_decision_threshold(model, val_split, grid=None)` — sweeps `θ ∈ {0.05, …, 0.95}` on the val split, returns the θ that maximises F1 (ties broken by precision).
- `evaluate_test_clf(model, test_split, decision_threshold)` — predicts → applies θ → returns `(y_true, y_prob, y_pred, metrics)`. Metrics include AUC (`nan` if a class is missing) and the 2×2 confusion matrix.

### 3.1.c Why `Dropout(0.2)` (the one architectural difference)

Classification on noisy financial data overfits much faster than regression — a regression head has to hit the right *price*; a classification head only has to be on the right side of the decision boundary. Adding `Dropout(0.2)` between `LSTM(64)` and the sigmoid keeps the model from memorising the small Vietnam training sets (the smallest, MWG, has only 1 471 train windows). All other knobs (Adam, 10 epochs, batch 64) stay the same.

### 3.1.d Decision-threshold tuning on val (not 0.5)

A sigmoid output is a probability — turning it into a 0 / 1 decision needs a cutoff. We **don't fix θ = 0.5**. Instead:

1. After training, predict probabilities on the val set.
2. Sweep `θ ∈ {0.05, 0.10, …, 0.95}` (19 candidates).
3. Pick the θ that maximises **F1 on val** (ties broken by precision).
4. Apply that θ on the test set for the final metrics.

This makes the model **robust to class imbalance** without changing the loss function — different tickers naturally end up with different θ, reflecting their different positive-class rates. Storing θ in `results_vn_buy[t]['threshold']` keeps every downstream cell self-contained: any plot or metric re-derivation uses the same θ that was tuned during training.

### 3.1.e Metrics

For a buy signal, **precision** matters most — false positives = buying into a flat or losing trade — so a model that runs lean on `pred+` and high on precision is preferred even at the cost of some recall. The report table includes all four classification metrics plus ROC-AUC (a threshold-independent ranking quality metric), the test-set positive rate (`pos`), the fraction the model decides to buy (`pred+`), and the tuned θ.

VCB gets two diagnostic plots — confusion matrix at the tuned θ, and ROC curve with the operating point marked. Other tickers stay in the table (same "no redundant per-ticker plots" rule as Task 1 / 2).

---

## 3.2 Sell signal

Mirror of 3.1 with the label flipped: positive iff `Close[t + 5] / Close[t] − 1 < −0.02`. Same `h = 5`, same `τ = 0.02`, same model architecture, same threshold-tuning protocol.

### 3.2.a Why two independent models, not one 3-class softmax

The buy and sell models are kept **separate** (rather than a softmax over `{sell, hold, buy}`) for two reasons:

1. The spec explicitly asks for two subtasks with two scores — one per direction.
2. Buy and sell aren't strictly mutually exclusive in a noisy market. There are days where the next 5 days neither rise > 2 % nor fall < −2 % *and* days where neither model fires — the implicit "hold" regime. A 3-class softmax would force the model to commit to one of three labels; two independent sigmoids let the asset stay in a non-decision regime when no signal is strong.

### 3.2.b Positive-class survey

Sell positive rate is generally lower than buy — Vietnamese equities historically drift upward (long-run positive return), so down-moves of ≥ 2 % over 5 days are rarer than up-moves of the same magnitude. Numbers from the full series:

| Ticker | Buy + % | Sell + % |
|---|---:|---:|
| VCB | 30.7 % | 26.8 % |
| HPG | 35.7 % | 30.5 % |
| FPT | 29.6 % | 25.5 % |
| VNM | 27.0 % | 21.8 % |
| MSN | 30.7 % | 28.5 % |
| MWG | 35.2 % | 25.5 % |

### 3.2.c Metric emphasis differs vs 3.1

For a sell signal, **recall matters more than in 3.1** — missing a sell signal means holding through a drawdown, which is asymmetrically costly in a long-only portfolio (you keep paying for the position to fall). The threshold-tuner still maximises F1 (a recall/precision balance), but when reading the table, prefer tickers with high *recall* over those with high *precision* if both reach similar F1. Same plot artefacts as 3.1 (VCB CM + ROC, red palette).

---

## 4.1 Profitability score

Task 4 trains **no new model**. The deep-learning core is in Task 2.3 and Task 3.2; Task 4 is an allocation + walk-forward backtest layer on top.

### 4.1.a Expected-return definition

For each `(ticker t, decision day d)` in the test window:

`E[return]_t,d  =  pred_close_t(d + h) / close_t(d) − 1`,  with `h = 5`.

`pred_close_t(d + h)` is the **5ᵗʰ step** of the Task 2.3 k=7 multistep LSTM for ticker `t`. We re-use `results_vn_kday[(t, 7)]` rather than training a dedicated h=5 head because:

1. Step 5 lies strictly inside k=7's training horizon — no extrapolation past the loss-optimised range.
2. Per-step RMSE tables in §2.3 already validated each step's error envelope, so we know what we're paying for.
3. Avoiding a third LSTM keeps the deep-learning stack identical across Task 2 / Task 3 / Task 4 — one fewer thing to justify in the report.

The denormalisation re-uses `denormalize_label_multistep` (per-window MinMax stored on `test.label_min` / `test.label_max`).

### 4.1.b Aggregation columns

The §4.1 table reports four numbers per ticker:

| Column | Definition | What it tells you |
|---|---|---|
| `mean E[return]` | mean of `E[return]_t,d` over the test window | "Would I buy this on average?" — a signed level |
| `mean realised` | mean of realised 5-day returns over the same days | ground truth; compares to the column above |
| `hit rate` | fraction of days where `sign(E[return]) == sign(realised)` | directional accuracy — agnostic to magnitude |
| `rank ρ` | Spearman correlation between `E[return]` and realised | does the model **rank** good days vs bad? |

We rely on the **ranking** columns (hit rate, ρ) in §4.3, not the levels — Task 2's regression errors compound at h=5 enough that levels are noisier than the rank.

### 4.1.c What we deliberately don't do

- **No reranking with Task 3.1's P(buy).** The buy classifier already shares the same underlying input pipeline; adding it as a second profitability signal would double-count the same input variation. We keep them complementary: E[return] for §4.1 / §4.3 scoring, P(sell) for §4.2 / §4.3 risk.
- **No per-ticker bias correction.** Some tickers' Task 2.3 model has a slight optimistic / pessimistic bias on the test window. We expose it in the table (`mean E[return]` vs `mean realised`) but don't centre it — doing so would bake in test-set knowledge.

---

## 4.2 Risk score

Two complementary signals, blended into a single `[0, 1]` cross-sectional risk score per `(date, ticker)`:

1. **Trailing 20-day realised volatility** of daily Close returns — classical "how much has this been moving lately?" risk. Computed on the full Close series; sliced at each decision day so it's trailing (no look-ahead).
2. **P(sell) from the Task 3.2 LSTM** — a *learned* bearish-regime probability. Catches setups the rolling vol won't: small std but a steady downtrend, for example.

### 4.2.a Why both signals

A single risk axis would either over-penalise high-vol-but-flat names (vol-only) or miss broad-market drawdowns where every name has elevated vol but only some have a bearish setup (P(sell)-only). The two are correlated enough (~0.3–0.5 typically) to reinforce in clear cases but disagree often enough to be useful as independent dimensions.

### 4.2.b Normalisation

For each day `d` independently:

1. Z-score each signal **across tickers**: `z_vol = (vol − vol.mean()) / vol.std()`, ditto for P(sell). Cross-sectional rather than across-time so the score is relative — at least one ticker is the safest and one the riskiest on every day.
2. Average: `raw = 0.5·z_vol + 0.5·z_psell`.
3. Min-max per row to `[0, 1]`: `risk = (raw − raw.min()) / (raw.max() − raw.min())`.

The `0.5 / 0.5` blend is the default; this is the place to tune if a future experiment shows one signal dominates. We didn't sweep it — the qualitative story is robust to the blend and we wanted to keep the methodology simple to defend.

### 4.2.c What the table shows

Per ticker:

- `mean vol` — average trailing 20-day vol on test days (in raw daily-return units, so values like `0.020` ≈ 2 % daily std)
- `mean P(sell)` — average sell probability on test days
- `mean risk` — the final `[0, 1]` score, averaged over days
- `#riskiest` / `%riskiest` — how often the ticker was the **top-1** by daily risk score (this is the "would be excluded under K=1" frequency that §4.3's prudent profile cares about)

---

## 4.3 Portfolio composition & backtest

### 4.3.a Single combined score

Per `(date, ticker)`: `s = E[return] − λ · risk`.

`λ` is the only "investor preference" knob. The score is then water-filled into weights (see 4.3.b). Negative scores are clipped to zero before normalisation — this is a **long-only** portfolio, so under-water names just get zero weight rather than going short.

### 4.3.b Water-filling weights (`_row_weights`)

For one day's score vector:

1. Zero out excluded names (if any).
2. Clip negatives to 0, normalise the remainder to sum to 1.
3. **Water-fill the cap:** if any weight would exceed `weight_cap`, set those to `weight_cap` and re-distribute the leftover mass across the rest proportionally to score. Repeat until no remaining name exceeds the cap.

Naive "clip & renormalise" loses mass when the redistribution can't proceed (e.g., remaining names already at cap). Water-filling always sums to 1 as long as `n_active · weight_cap ≥ 1`. With 6 tickers and `cap ∈ {0.25, 0.5}`, that holds with room to spare.

### 4.3.c Prudent vs aggressive

| Profile      | `λ` | Weight cap | Exclude top-K riskiest |
|--------------|----:|-----------:|----------------------:|
| Prudent      | 2.0 | 25 %       | 1                     |
| Aggressive   | 0.5 | 50 %       | 0                     |

`λ` is the dominant knob — it directly weighs return against risk inside the score. The weight cap is a secondary safety: prudent caps at 25 % so no single name dominates even if it scores well; aggressive allows up to 50 %. The hard exclusion (`K = 1`) on the prudent profile takes the day's riskiest name out of the universe entirely, on top of the soft `λ` penalty — belt-and-braces.

Defaults are chosen, not learned. We did not sweep `(λ, cap, K)` on validation because:

- The test universe is small (6 tickers), so a sweep would overfit fast.
- Task 4's grading hinges on *methodology*, not on hitting a specific Sharpe target — fixed, defensible defaults beat a noisy "best on val" pick.

### 4.3.d Backtest mechanics

- Walk-forward on the **test** window only — train/val are never touched.
- Rebalance every `h = 5` trading days; weights stay fixed in between (no continuous drift / no daily rebalancing — closer to a realistic strategy that costs trade execution).
- NAV uses **raw daily** Close-to-Close returns, so Sharpe is annualised against true day-to-day vol. The `active` weights are forward-filled from rebalance days.
- Benchmark: equal-weight portfolio on the same calendar / rebalance schedule, so differences come from scoring, not universe choice.

### 4.3.e Metrics

- **Annualised return** — `(NAV_final / NAV_start)^(252/n_days) − 1`
- **Sharpe** (rf=0) — `√252 · mean(daily_returns) / std(daily_returns)`
- **Max drawdown** — `min((NAV − running_max) / running_max)`

The NAV plot is a single panel (prudent / aggressive / equal-weight) with final-value annotations. We deliberately don't plot one ticker at a time — the same "comparison-table-not-per-ticker-viz" rule from Task 2 / Task 3.

### 4.3.f Look-ahead audit

Everything is computed using only information available at the decision day:

| Quantity | Source | Look-ahead? |
|---|---|---|
| `E[return]` | Task 2.3 LSTM (best epoch on train, by val_loss) | No — model has no knowledge of test |
| `P(sell)` | Task 3.2 LSTM (best epoch on train, by val_loss) | No |
| `rolling_vol` | trailing 20-day std of daily returns *up to and including* decision day | No |
| `realized_h_return` | future close at `d+h` | **Used only to compute NAV during backtest**, never in scoring |
| Daily returns for NAV | raw Close-to-Close | Used after weights are decided at rebalance day |

---

## Reusable helpers — quick reference

| Function | Defined in cell | Used by |
|---|---|---|
| `add_technical_features(df)` | 1.1 (feature cell) | 1.1 / 1.2 / 1.3 pipelines, 2.1 / 2.2 / 2.3 pipelines, plotting cells |
| `build_pipeline(df, name, window_size=30, horizon=1, val_ratio=0.15, test_ratio=0.15, feature_cols=None, label_col=None)` | 1.1 (pipeline cell) | 1.1 (3× Nasdaq), 1.2 (per (ticker, k)), 2.1 / 2.2 (Vietnam — passes `VN_FEATURE_COLS`/`VN_LABEL_COL`) |
| `denormalize_label(y_norm, label_min, label_max)` | 1.1 (pipeline cell) | 1.1 / 2.1 evaluation, `evaluate_test` |
| `checkpoint_path(task, name)` | checkpoint utilities | every training cell — builds `models/<task>/<name>.keras` |
| `train_with_checkpoint(model, train, val, save_path, ...)` | checkpoint utilities | every training cell — wraps `model.fit` with `ModelCheckpoint` |
| `build_lstm(input_shape)` | 1.1 extension | 1.1 MSFT/NVDA training, 1.2 training loop, 2.1 / 2.2 training loops |
| `evaluate_test(model, test)` | 1.1 extension | 1.1 / 1.2 / 2.1 / 2.2 comparison tables & VCB / AAPL plots |
| `build_pipeline_multistep(df, name, k, ..., feature_cols=None, label_col=None)` | 1.3 helpers | 1.3 training loop, 2.3 training loop |
| `build_lstm_multistep(input_shape, k)` | 1.3 helpers | 1.3 / 2.3 training loops |
| `denormalize_label_multistep(...)` | 1.3 helpers | `evaluate_test_multistep` |
| `evaluate_test_multistep(model, test)` | 1.3 helpers | 1.3 / 2.3 per-step tables & AAPL / VCB trajectory plots |
| `load_vietnam(ticker)` | 2.1 (Vietnam loading) | builds the 6 entries of `VN_RAW_FRAMES`; drops the leading unnamed index column, renames `TradingDate → Date`, parses dates |
| `ClfSplit` (namedtuple) | 3.1 helpers | classification analogue of `Split` — `.X`, `.y` only |
| `make_signal_labels(close, horizon, threshold, direction)` | 3.1 helpers | 1 if `Close[t+h]/Close[t]-1 > +τ` (buy) or `< -τ` (sell); NaN-padded last `h` rows |
| `build_pipeline_classification(df, name, horizon=5, threshold=0.02, direction='buy', ..., feature_cols=None, label_col=None)` | 3.1 helpers | Task 2 pipeline with 0/1 labels — used by 3.1 and 3.2 training loops |
| `build_lstm_classifier(input_shape)` | 3.1 helpers | `LSTM(64) → Dropout(0.2) → Dense(1, sigmoid)`, BCE loss, Adam |
| `tune_decision_threshold(model, val_split, grid=None)` | 3.1 helpers | sweeps θ on val, returns the θ maximising F1 (ties broken by precision) |
| `evaluate_test_clf(model, test_split, decision_threshold)` | 3.1 helpers | predict → apply θ → return `(y_true, y_prob, y_pred, metrics)` with AUC + CM |
| `predict_h_step_close(model, test_split, h)` | 4 helpers | one-pass predict + denormalise for the multistep LSTM, returns the h-step-ahead Close as a 1-D array |
| `ticker_test_frame(ticker, h=5)` | 4 helpers | per-decision-day DataFrame on the test window: `Date, Close, e_return, realized_h_return, rolling_vol, p_sell` |
| `build_test_panel()` | 4 helpers | `{ticker → ticker_test_frame(ticker)}` for every Vietnam ticker — used everywhere in §4 |
| `compute_risk_panel(panel)` | 4 helpers | wide DataFrame `[date × ticker]` of risk scores in `[0, 1]`; cross-sectional z-blend of vol + P(sell), then row min-max |
| `compute_score(e_return_wide, risk_wide, lam)` | 4 helpers | `E[return] − λ · risk`, indices preserved |
| `_row_weights(score_row, exclude_set, weight_cap)` | 4 helpers | **water-filling** one day's weights — clips negatives, applies the cap, redistributes leftover mass; always sums to 1 if `cap·n_active ≥ 1` |
| `build_weights(score_wide, risk_wide, weight_cap, exclude_top_k)` | 4 helpers | calls `_row_weights` per day; excludes the top-K riskiest names on each day if `exclude_top_k > 0` |
| `equal_weights_like(score_wide)` | 4 helpers | equal-weight benchmark DataFrame, same index/columns |
| `daily_returns_calendar(panel)` | 4 helpers | wide DataFrame `[date × ticker]` of raw daily Close-to-Close returns — used for NAV computation |
| `backtest(weights, daily_returns, rebalance_every=5)` | 4 helpers | walk-forward backtest; rebalance every `h` decision days and keep weights fixed in between; returns `(nav, port_daily, active_weights)` |
| `portfolio_metrics(nav, port_daily)` | 4 helpers | dict with `Annualised Return`, `Sharpe` (rf=0, 252 trading days), `Max Drawdown` |

For the full list of variable names (split shapes, dict keys, helper signatures, every checkpoint path) see [Local variable.yml](Local%20variable.yml).

---

## Things to remember when extending the notebook

- **Never shuffle.** Both `train_test_split` calls in `build_pipeline` pass `shuffle=False`. Any new pipeline must do the same.
- **Normalise per window, never globally.** A `StandardScaler` fit on the train set and applied to val/test would technically work but breaks compatibility with the sample notebook's reporting convention and makes denormalisation harder.
- **Always denormalise before reporting metrics in dollars.** Normalised MSE is fine for cross-ticker / cross-task comparison; for human-readable plots and tables use dollars.
- **Compare across tickers in normalised units, not dollars.** NVDA's price range is much tighter than AAPL's, so equal-skill models will produce very different dollar errors. The `MSE (norm)` column is the right axis for cross-ticker comparison.
- **Reload after checkpointing.** Every training cell calls `train_with_checkpoint(...)` and then `model = load_model(...)`. The reload is **not optional** — without it, `model` is the **last-epoch** weights, but the checkpoint on disk is the **best-by-val-loss** weights. Skipping the reload means downstream evaluation cells use a different model than the one persisted to disk.
- **Dict-key conventions:**
  - `results_k1[ticker]` — string key, holds the Nasdaq k=1 model and its train/val/test.
  - `results_kth[(ticker, k)]` — tuple key, holds Nasdaq 1.2's single-day-ahead models. `.y` is scalar.
  - `results_kday[(ticker, k)]` — tuple key, holds Nasdaq 1.3's multi-step models. `.y` is a vector of length k.
  - `results_vn_k1[ticker]` / `results_vn_kth[(ticker, k)]` / `results_vn_kday[(ticker, k)]` — Vietnam mirrors of the above. The six dicts are kept separate (rather than merged) so each task stays self-contained — a Vietnam ticker name colliding with a Nasdaq one (e.g., none do today, but they could) wouldn't be ambiguous.
  - `results_vn_buy[ticker]` / `results_vn_sell[ticker]` — Task 3 buy / sell signal classifiers. Each value has the standard `{model, history, train, val, test}` keys **plus a `threshold`** field — the val-tuned decision θ used to convert sigmoid output to a 0 / 1 buy/sell call. Whenever you re-evaluate one of these models downstream, **use the stored θ**, not 0.5.
  - `test_panel[ticker]` — Task 4's per-decision-day DataFrame on the test window (`Date, Close, e_return, realized_h_return, rolling_vol, p_sell`). Built by `build_test_panel()`; reused across §4.1 / §4.2 / §4.3.
  - `risk_wide` — wide DataFrame `[date × ticker]` of cross-sectional risk scores in `[0, 1]`. Built once in §4.2 and re-used by §4.3's weight builder.
  - `portfolio_results['prudent' | 'aggressive' | 'equal']` — Task 4 backtest outputs. Each entry has `{nav, returns, weights, metrics}`. NAV is a pandas Series indexed by daily dates; metrics has `Annualised Return`, `Sharpe`, `Max Drawdown`.
- **1.3 plot offset.** The 1.3 trajectory plot's date offset is `test_start_idx + window_size` (the **first** of the k predicted days). Do **not** add `+ k − 1` — that's the offset for 1.2's plot, which targets only the **kᵗʰ** day.
- **Don't rename load-bearing variables.** Task 1: `AAPL_train`, `AAPL_val`, `AAPL_test`, `AAPL_LSTM_model`, `MSFT_LSTM_model`, `NVDA_LSTM_model`, `feat_aapl`, `K_VALUES`, `TICKERS`, `SPLITS_K1`, `RAW_FRAMES`, `results_k1`, `results_kth`, `results_kday`, `MODELS_DIR`. Task 2: `VN_TICKERS`, `VN_DATA_DIR`, `VN_FEATURE_COLS`, `VN_LABEL_COL`, `VN_RAW_FRAMES`, `VN_SPLITS_K1`, `feat_vcb`, `results_vn_k1`, `results_vn_kth`, `results_vn_kday`. All are referenced by later cells. See [Local variable.yml](Local%20variable.yml) for the full list.
- **Date offset for plotting test labels:** `test_start_idx + window_size + k − 1`, where `test_start_idx = N_train + N_val`. Use the **per-(ticker, k) pipeline's own** train/val sizes (from `results_kth[(ticker, k)]['train']` / `results_vn_kth[(ticker, k)]['train']` and `['val']`), not the 2.1 / 1.1 split sizes — total windows shrink slightly as k grows.
- **No redundant per-ticker plots.** Task 1 plots predicted-vs-real for AAPL only; Task 2 for VCB only. The other tickers appear in the comparison tables. Tables convey "how well did each ticker do"; mirroring the same plot N times pads the report without adding insight.
- **Vietnam vs Nasdaq pipeline parity.** The pipeline helpers `build_pipeline` and `build_pipeline_multistep` accept optional `feature_cols=` / `label_col=` kwargs. When `None` (Task 1 call sites), they default to the Nasdaq globals (`FEATURE_COLS` / `LABEL_COL`). Task 2 call sites pass `VN_FEATURE_COLS` / `VN_LABEL_COL`. Don't drop these kwargs from Vietnam calls — silently using the Nasdaq globals would try to read an `Adjusted Close` column that doesn't exist on Vietnam frames and raise a KeyError mid-pipeline.
- **VND vs $ metric labels.** `evaluate_test` / `evaluate_test_multistep` return dicts with keys named `'MSE ($²)'`, `'RMSE ($)'`, `'MAE ($)'`. These names are historical (from Task 1); the *values* are in whatever unit the label column uses (USD for Nasdaq, VND for Vietnam). The Task 2 print cells relabel them as VND in the table headers; don't try to "fix" the dict keys downstream — it would break Task 1 cells.
- **Task 3 windowing has 4 fewer windows per ticker than Task 2.1.** The classification pipeline uses `last_valid = len(arr) − window_size − horizon + 1` with `horizon = 5`; Task 2.1's regression pipeline uses `horizon = 1`. So per ticker we lose `5 − 1 = 4` windows total (after the 70/15/15 split this shows up as ~2-3 fewer in each of train/val/test vs Task 2.1). Don't try to reuse the 2.1 `VN_SPLITS_K1` shapes for Task 3 — call `build_pipeline_classification` and read shapes from its return.
- **Task 3 stores a per-ticker decision threshold.** `results_vn_buy[t]['threshold']` and `results_vn_sell[t]['threshold']` are the val-tuned θ. Always use them when calling `evaluate_test_clf` — `decision_threshold=0.5` will give different (usually worse) metrics, and any inconsistency between table and plot would be confusing.
- **Different tickers get different θ.** That's by design — tickers with lower positive-class rates get lower θ. Don't try to enforce a global θ; the tuner per ticker is what makes the pipeline robust to class imbalance without adding a `class_weight` argument.
- **Task 3 ROC-AUC may be `nan` if a class is missing in a test set.** `evaluate_test_clf` guards `roc_auc_score` against this `ValueError` — won't happen on our six tickers at h=5, τ=0.02 (positive rates 22–36 %), but keep the guard if you later sweep larger τ values where a small ticker's test set could end up all-zero.
- **Task 4 trains no new models.** It reuses `results_vn_kday[(t, 7)]` (Task 2.3) and `results_vn_sell[t]` (Task 3.2). If you re-run the notebook from scratch, **Task 4 cells will fail until 2.3 and 3.2 have populated their result dicts.** No new `checkpoint_path` entries — there's nothing to save.
- **Task 4 uses the 5ᵗʰ step of the k=7 multistep model**, not a dedicated h=5 single-output model. `predict_h_step_close` takes `h=5` and indexes `y_pred[:, 4]`. Don't re-train an h=5 head — the per-step RMSE table in §2.3 already validated step 5's behaviour.
- **Task 4 test-window alignment loses 2 days per ticker** because Task 2.3's k=7 and Task 3.2's h=5 windowing produce slightly different test ranges. `ticker_test_frame` merges them on `Date` (inner join), so the resulting panel has ≤ both. For VCB that's 502 days out of 504 each side — a non-issue but expected.
- **Cap is water-filled, not clipped.** `_row_weights` uses water-filling (cap a name, redistribute the leftover, repeat) — not "clip then renormalise". The latter loses mass when the redistribution can't continue (e.g. only one name has positive score after exclusion). Keep the water-filling logic; the smoke test failed without it.
- **Weights stay fixed between rebalances.** `backtest` forward-fills the rebalance-day weights across daily returns until the next rebalance. This is deliberate — letting weights drift with returns is more realistic but adds another dimension to defend in the report, and it would not change the Sharpe ranking we care about.
- **Risk score is cross-sectional, not absolute.** `compute_risk_panel` z-scores per **day** across tickers, then min-max per day. So `risk = 0` means "safest of today's six tickers" and `risk = 1` means "riskiest of today's six tickers" — not "objectively safe / risky". Don't interpret `risk = 1.0` as a vol bound; it's a relative rank.
