We considered time-series cross-validation (rolling/expanding window). Given the large dataset sizes (6,000–10,000 rows per ticker) and the need for comparable test sets across Tasks 1–4, we opted for a single chronological 70/15/15 holdout split. This avoids the computational overhead of refitting across multiple folds while still preserving temporal order and preventing leakage.



