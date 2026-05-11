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
└── task1.2/
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

## Reusable helpers — quick reference

| Function | Defined in cell | Used by |
|---|---|---|
| `add_technical_features(df)` | 1.1 (feature cell) | 1.1 pipeline, 1.2 pipeline, plotting cells |
| `build_pipeline(df, name, window_size=30, horizon=1, val_ratio=0.15, test_ratio=0.15)` | 1.1 (pipeline cell) | 1.1 (called 3× for AAPL/MSFT/NVDA), 1.2 (called per (ticker, k)) |
| `denormalize_label(y_norm, label_min, label_max)` | 1.1 (pipeline cell) | 1.1 evaluation, `evaluate_test` |
| `checkpoint_path(task, name)` | checkpoint utilities | every training cell — builds `models/<task>/<name>.keras` |
| `train_with_checkpoint(model, train, val, save_path, ...)` | checkpoint utilities | every training cell — wraps `model.fit` with `ModelCheckpoint` |
| `build_lstm(input_shape)` | 1.1 extension | 1.1 MSFT/NVDA training, 1.2 training loop |
| `evaluate_test(model, test)` | 1.1 extension | 1.1 cross-ticker table, 1.2 comparison table & AAPL plot |

For the full list of variable names (split shapes, dict keys, helper signatures, every checkpoint path) see [Local variable.yml](Local%20variable.yml).

---

## Things to remember when extending the notebook

- **Never shuffle.** Both `train_test_split` calls in `build_pipeline` pass `shuffle=False`. Any new pipeline must do the same.
- **Normalise per window, never globally.** A `StandardScaler` fit on the train set and applied to val/test would technically work but breaks compatibility with the sample notebook's reporting convention and makes denormalisation harder.
- **Always denormalise before reporting metrics in dollars.** Normalised MSE is fine for cross-ticker / cross-task comparison; for human-readable plots and tables use dollars.
- **Compare across tickers in normalised units, not dollars.** NVDA's price range is much tighter than AAPL's, so equal-skill models will produce very different dollar errors. The `MSE (norm)` column is the right axis for cross-ticker comparison.
- **Reload after checkpointing.** Every training cell calls `train_with_checkpoint(...)` and then `model = load_model(...)`. The reload is **not optional** — without it, `model` is the **last-epoch** weights, but the checkpoint on disk is the **best-by-val-loss** weights. Skipping the reload means downstream evaluation cells use a different model than the one persisted to disk.
- **Dict-key conventions:**
  - `results_k1[ticker]` — string key, holds the k=1 model and its train/val/test.
  - `results_kth[(ticker, k)]` — tuple key, holds the 1.2 models. The two dicts are kept separate (rather than merged into one giant `results[(ticker, k)]` with k ∈ {1, 3, 7}) so that 1.1 stays self-contained.
- **Don't rename load-bearing variables.** `AAPL_train`, `AAPL_val`, `AAPL_test`, `AAPL_LSTM_model`, `MSFT_LSTM_model`, `NVDA_LSTM_model`, `feat_aapl`, `K_VALUES`, `TICKERS`, `SPLITS_K1`, `RAW_FRAMES`, `results_k1`, `results_kth`, `MODELS_DIR` are all referenced by later cells. Renaming them silently will break downstream evaluation/plotting cells. See [Local variable.yml](Local%20variable.yml) for the full list.
- **Date offset for plotting test labels:** `test_start_idx + window_size + k − 1`, where `test_start_idx = N_train + N_val`. Use the **per-(ticker, k) pipeline's own** train/val sizes (from `results_kth[(ticker, k)]['train']` and `['val']`), not the 1.1 split sizes — total windows shrink slightly as k grows.
- **No redundant per-ticker plots.** The notebook only plots predicted-vs-real for AAPL — MSFT and NVDA appear only in the comparison tables. This is deliberate: tables convey "how well did each ticker do", and adding the same plot N times only pads the report without adding insight.
