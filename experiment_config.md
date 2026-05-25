# Experiment Configuration

## 1. Experiment Identity

- Project Title: Incremental Feature Selection Framework for Real-Time Stock Price Forecasting Using Deep Time Series Models
- Experiment Type: Multi-symbol, multi-horizon online learning experiment
- Experiment Status: Final thesis rerun
- Implementation Language: Python
- Main Framework: PyTorch
- Repository Purpose: Reproducible thesis modelling pipeline
- Data Source Layer: CSV snapshot stored in the project `Data/` folder
- Source File: `Data/canonical_1m_rth_FULL.csv`
- Original Source Table: `staging.canonical_1m_rth`
- Market Session: Regular Trading Hours (RTH)
- Data Frequency: 1-minute bars
- Code Version / Git Commit: latest pushed commit on main branch
- Final Run Date: TBD after final full rerun

---

## 2. Final Data Snapshot

- Snapshot ID: final_snapshot_01
- Snapshot Policy: fixed cutoff with full RTH trading days only
- Symbols: AAPL, MSFT, NVDA, TSLA
- Snapshot Start: 2025-01-02 09:30:00 America/New_York
- Snapshot End: 2026-02-13 15:59:00 America/New_York
- Source Data End Before Cutoff: 2026-05-01 15:59:00 America/New_York
- Excluded Gap Period: 2026-02-14 to 2026-03-29
- Full-Day Requirement: 390 one-minute RTH bars per symbol per trading day
- Half-Days Excluded: enabled
- Incomplete Intraday Days Excluded: enabled
- Final Row Counts: TBD after final snapshot count

All baselines, no-IFS experiments, and IFS experiments use this exact same filtered snapshot.

---

## 3. Research Alignment

### Main Research Question

Does an Incremental Feature Selection framework enhance the adaptability, interpretability, and predictive performance of deep time-series models for real-time stock price forecasting?

### RQ1

To what extent does integrating an Incremental Feature Selection mechanism based on dynamic feature weighting improve short-term forecast accuracy and robustness in real-time financial data?

### RQ2

Which features have the most significant influence on model performance, and how does their relative importance change over time as new data arrive?

### RQ3

To what extent does the real-time implementation of the proposed Incremental Feature Selection framework provide predictive advantages over traditional historical forecasting models?

---

## 4. Data Source and Filtering

### Required Raw Columns

The CSV snapshot must contain at least the following columns:

- symbol
- datetime
- open
- high
- low
- close

Volume may exist in the raw source data, but it is not used in the final modelling feature set.

### Derived Columns

The following variables are constructed inside the modelling scripts:

- change_ratio
- macd
- minute_of_day
- day_of_week
- month_of_year

### Timezone Handling

- Raw timestamps are interpreted as UTC.
- Timestamps are converted to America/New_York for RTH filtering and interpretation.
- Final snapshot boundaries are defined in America/New_York time.

### Snapshot Filter

```text
datetime >= 2025-01-02 09:30:00 America/New_York
datetime <= 2026-02-13 15:59:00 America/New_York
symbol IN (AAPL, MSFT, NVDA, TSLA)
```

### Full-Day Filtering

Only complete RTH trading days are retained. A valid trading day must contain exactly 390 one-minute bars per symbol. Any day with fewer or more than 390 bars is excluded before model training and evaluation.

Removed or incomplete days are recorded as `removed_days` CSV files inside the `Results/` directory when each script is executed.

---

## 5. Model Inventory

The experiment includes seven model families. Each model is evaluated across all four symbols and all forecast horizons.

- M1: Naive baseline
- M2: ARIMA baseline using `auto_arima` model selection
- M3: Offline LSTM
- M4: Online LSTM
- M5: Incremental Transformer, close-only
- M6: IL-ETransformer without IFS
- M7: IL-ETransformer with Incremental Feature Selection

Each model is implemented in two target variants where applicable:

- P variant: price-level forecasting
- D variant: delta forecasting

Model scripts are stored in:

```text
Models/Baselines/
Models/IFS_Models/
```

---

## 6. Forecasting Setup

- Forecast Horizons: 1, 30, 60, 240, 390 bars
- Warmup Split: 20%
- Validation Split: 10%
- Online Evaluation Split: 70%
- LOOKBACK: 60 bars
- Evaluation Start: first 60 online steps skipped for consistency across models
- Main Evaluation Mode: chronological walk-forward evaluation

The same chronological split policy is applied across all models to maintain comparability.

---

## 7. Target Definitions

### Price Variant

The price variant predicts the future closing price:

```text
price_target[t] = close[t + horizon]
```

Direction is derived from the predicted price movement relative to the current close:

```text
pred_direction = 1 if pred_price > close[t] else 0
```

### Delta Variant

The delta variant predicts the future price change:

```text
delta_target[t] = close[t + horizon] - close[t]
```

For delta models, regression metrics are computed on the predicted delta. Directional metrics are included only where implemented.

---

## 8. Feature Sets

### Close-Only Models

The following models use only close-price history:

- M1
- M2
- M3
- M4
- M5

### Six-Feature IL-ETransformer Models

The following models use six market features:

- M6
- M7

Feature list:

- open
- high
- low
- close
- change_ratio
- macd

Time embeddings are also used in M6 and M7:

- minute_of_day
- day_of_week
- month_of_year

---

## 9. Reproducibility

- Global Random Seed: 42
- All stochastic models use `RANDOM_SEED = 42`.
- The seed is applied to Python `random`, NumPy, PyTorch, and CUDA where applicable.
- Dependencies are documented in `requirements.txt`.
- The data snapshot is fixed as `Data/canonical_1m_rth_FULL.csv`.
- The execution entry point is `run_all.py`.
- All outputs are written to `Results/`.
- All model scripts use relative project paths, not server-specific absolute paths.

### ARIMA Configuration

- ARIMA implementation: `auto_arima`
- Model selection: automatic ARIMA order selection using `auto_arima`
- Refit policy: periodic refit during walk-forward evaluation where implemented
- REFIT_INTERVAL: 500 bars
- ARIMA is deterministic for a fixed dataset and configuration, so no stochastic seed is required for ARIMA fitting.

### Random Search Configuration

For stochastic deep learning models:

- Random search budget: 15 trials
- Random seed: 42
- Search objective: validation MAE
- The same search budget is used across comparable models.

---

## 10. Output Structure

All generated outputs are written to the `Results/` directory.

Expected output locations:

```text
Results/
Results/logs/
Results/Gate_P/
Results/Gate_D/
```

### Main Outputs

Model-level summary files and prediction files are saved directly inside:

```text
Results/
```

### Logs

Execution logs from `run_all.py` are saved inside:

```text
Results/logs/
```

### Gate History Outputs

For M7 only:

```text
Results/Gate_P/
Results/Gate_D/
```

- `Results/Gate_P/`: gate histories for the price variant
- `Results/Gate_D/`: gate histories for the delta variant

Gate history files contain feature-gate values recorded during online evaluation.

---

## 11. Pipeline Execution

The entire modelling pipeline can be executed from the project root using:

```bash
python3 run_all.py
```

The script runs all model files in sequence and stores logs in:

```text
Results/logs/
```

If a model fails, `run_all.py` stops and reports the failed script.

---

## 12. Repository Structure

Expected project structure:

```text
Project/
├── Data/
│   ├── canonical_1m_rth_FULL.csv
│   └── README.md
├── Models/
│   ├── Baselines/
│   │   ├── M1_P.py
│   │   ├── M1_D.py
│   │   ├── M2_P.py
│   │   ├── M2_D.py
│   │   ├── M3_P.py
│   │   ├── M3_D.py
│   │   ├── M4_P.py
│   │   ├── M4_D.py
│   │   ├── M5_P.py
│   │   ├── M5_D.py
│   │   ├── M6_P.py
│   │   └── M6_D.py
│   └── IFS_Models/
│       ├── M7_P.py
│       └── M7_D.py
├── Results/
│   ├── logs/
│   ├── Gate_P/
│   └── Gate_D/
├── experiment_config.md
├── requirements.txt
├── run_all.py
├── .gitignore
└── README.md
```

---

## 13. Environment and Dependencies

The Python environment is documented using:

```text
requirements.txt
```

To reproduce the environment, create a virtual environment and install dependencies using:

```bash
pip install -r requirements.txt
```

The main libraries used include:

- pandas
- numpy
- scikit-learn
- statsmodels
- pmdarima
- torch
- matplotlib
- psycopg2-binary
- PyYAML

Exact versions are listed in `requirements.txt`.

---

## 14. Version Control

The project is tracked using Git and hosted on GitHub.

Repository:

```text
https://github.com/amirarzy/incremental-feature-selection-for-real-time-stock-forecasting
```

Version control is used to track major changes to:

- model scripts
- reproducibility configuration
- pipeline execution files
- documentation files
- dependency specification

The final Git commit hash should be added to this file after the final push.

---

## 15. Notes for External Reproduction

An external user should be able to reproduce the experiment by following these steps:

1. Clone the GitHub repository.
2. Place `canonical_1m_rth_FULL.csv` inside the `Data/` folder if it is not already included.
3. Create a Python environment.
4. Install dependencies from `requirements.txt`.
5. Run `python3 run_all.py` from the project root.
6. Inspect generated outputs in `Results/`.

The code uses relative project paths, so it should not depend on server-specific absolute paths.

---

## 16. Reproducibility Checklist Coverage

This configuration supports the Sprint 6 reproducibility checklist as follows:

- Clean and structured codebase: model scripts are separated by model family and target variant.
- External understandability: model inventory, data snapshot, features, targets, and output paths are documented.
- End-to-end execution: `run_all.py` executes all model scripts sequentially.
- Result reproduction: the fixed CSV snapshot, fixed split policy, fixed horizons, and fixed seeds support reproducible reruns.
- Dependencies and environment: `requirements.txt` records the Python package environment.
- Dataset accessibility: the expected data file and location are documented.
- Version control: the GitHub repository tracks the final codebase.
- Code walkthrough readiness: each model script is named by model number and target variant, and this file documents the design choices.