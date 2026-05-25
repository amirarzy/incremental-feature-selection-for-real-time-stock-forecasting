# 📈 Incremental Feature Selection for Real-Time Stock Forecasting

This repository contains the reproducible modelling pipeline for the thesis project:

**Incremental Feature Selection Framework for Real-Time Stock Price Forecasting Using Deep Time Series Models**

The project evaluates baseline, online learning, transformer-based, and Incremental Feature Selection models on 1-minute NASDAQ stock data.

---

## 🎯 Project Purpose

The goal of this repository is to make the full modelling pipeline:

- clean and well-structured
- documented and understandable
- reproducible from a fixed data snapshot
- executable by an external user
- suitable for Sprint 6 reproducibility evaluation

---

## 📁 Project Structure

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
│   ├── Gate_D/
│   └── README.md
├── experiment_config.md
├── requirements.txt
├── run_all.py
├── .gitignore
└── README.md
```

---

## 📊 Data

The required input file is:

```text
Data/canonical_1m_rth_FULL.csv
```

The dataset contains 1-minute Regular Trading Hours OHLC data for four NASDAQ symbols:

- AAPL
- MSFT
- NVDA
- TSLA

The fixed final snapshot covers:

```text
2025-01-02 09:30:00 America/New_York
to
2026-02-13 15:59:00 America/New_York
```

Only complete Regular Trading Hours trading days are retained.

A valid trading day must contain exactly:

```text
390 one-minute bars per symbol
```

More details are documented in:

```text
Data/README.md
```

---

## 🤖 Models

The repository contains seven model families:

| Model | Description |
|---|---|
| M1 | Naive baseline |
| M2 | ARIMA baseline using `auto_arima` |
| M3 | Offline LSTM |
| M4 | Online LSTM |
| M5 | Incremental Transformer close-only |
| M6 | IL-ETransformer without IFS |
| M7 | IL-ETransformer with Incremental Feature Selection |

Each model is implemented in two target variants:

| Variant | Meaning |
|---|---|
| `_P` | price-level forecasting |
| `_D` | delta forecasting |

---

## ⏱️ Forecasting Setup

The experiment uses:

- Forecast horizons: 1, 30, 60, 240, 390 bars
- Warmup split: 20%
- Validation split: 10%
- Online evaluation split: 70%
- LOOKBACK: 60 bars
- Main evaluation mode: chronological walk-forward evaluation
- Global random seed: 42

---

## 🧩 Feature Sets

Close-only models use close-price history:

```text
M1, M2, M3, M4, M5
```

The IL-ETransformer models use six market features:

- open
- high
- low
- close
- change_ratio
- macd

M6 and M7 also use time embeddings:

- minute_of_day
- day_of_week
- month_of_year

---

## 🔁 Reproducibility

The project uses:

- fixed CSV snapshot
- relative project paths
- fixed random seed
- fixed forecast horizons
- fixed chronological split policy
- documented dependencies
- documented experiment configuration
- single execution entry point

Full configuration details are provided in:

```text
experiment_config.md
```

---

## ⚙️ Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Full Pipeline

From the project root, run:

```bash
python3 run_all.py
```

The script executes all model files sequentially.

If a model fails, the pipeline stops and reports the failed script.

---

## 📦 Outputs

All generated outputs are written to:

```text
Results/
```

Execution logs are written to:

```text
Results/logs/
```

M7 gate-history files are written to:

```text
Results/Gate_P/
Results/Gate_D/
```

Generated result files are excluded from Git using `.gitignore`.

More details are documented in:

```text
Results/README.md
```

---

## 🗂️ Main Files

| File | Purpose |
|---|---|
| `experiment_config.md` | full experiment setup and reproducibility configuration |
| `requirements.txt` | Python dependencies |
| `run_all.py` | full pipeline execution script |
| `.gitignore` | excluded generated files and environment folders |
| `Data/README.md` | dataset documentation |
| `Results/README.md` | output documentation |

---

## 🧪 Syntax Check

A lightweight syntax check can be run with:

```bash
python3 -m py_compile run_all.py Models/Baselines/*.py Models/IFS_Models/*.py
```

If the command finishes without error, all Python files are syntactically valid.

---

## 🧭 External Reproduction Steps

An external user can reproduce the experiment by following these steps:

1. Clone the repository.
2. Place `canonical_1m_rth_FULL.csv` inside the `Data/` folder.
3. Create a Python environment.
4. Install dependencies using `pip install -r requirements.txt`.
5. Run `python3 run_all.py` from the project root.
6. Inspect generated outputs in the `Results/` folder.

---

## 🔗 Repository

```text
https://github.com/amirarzy/incremental-feature-selection-for-real-time-stock-forecasting
```

---

## 📝 Notes

The dataset file is required for full reproduction.

The repository uses relative paths, so the code does not depend on server-specific absolute paths.

The project is prepared as the final reproducible thesis modelling pipeline for Sprint 6.
