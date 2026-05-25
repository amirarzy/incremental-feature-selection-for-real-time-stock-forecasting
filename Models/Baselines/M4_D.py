#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online LSTM Baseline  |  next_delta target
==========================================
- Target: close[t + horizon] - close[t]

Differences from offline LSTM delta:
  1. Dynamic scaler: mean/std recomputed from rolling window each step
  2. Online update: after each prediction, one gradient step on actual delta
  3. Extra hyperparameter: online_learning_rate (Qian, 2025)

Same 15 random trials as offline model (seed=42) — fair comparison.

Resume capability:
    Completed (symbol, horizon) pairs are skipped automatically.
    Results are appended row by row so a partial run is never lost.

Evaluation starts at row LOOKBACK=60 to match IL-ETransformer.

Output columns (summary):
    symbol, horizon, model,
    best_seq_len, best_hidden_size, best_num_layers,
    best_dropout, best_lr, best_online_lr,
    best_batch_size, best_epochs,
    delta_mae, delta_rmse,
    delta_relative_mae_pct,
    implied_delta_relative_mae_pct

FIX (v2):
    evaluate_online previously used delta_tgt[t] = close[t+h] - close[t]
    as the online update target, which introduces data leakage at long
    horizons. Fixed to use current_delta = close[t] - close[t-1],
    consistent with run_trial. Hyperparameters are unchanged.
"""

import os
import math
import random
import yaml
import numpy as np
import pandas as pd
import psycopg2
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

TABLE_NAME = "staging.canonical_1m_rth"

SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA"]

NY_TZ = "America/New_York"

SNAPSHOT_START_NY = "2025-01-02 09:30:00"
SNAPSHOT_END_NY   = "2026-02-13 15:59:00"

REQUIRE_FULL_RTH_DAYS = True
EXPECTED_BARS_PER_DAY = 390

TARGET_HORIZONS_BARS = [1, 30, 60, 240, 390]

WARMUP_RATIO = 0.20
VAL_RATIO    = 0.10
ONLINE_RATIO = 0.70

LOOKBACK     = 60
SCALE_WINDOW = 390  # one trading day

N_TRIALS    = 15
RANDOM_SEED = 42

DATA_SOURCE = "csv"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "canonical_1m_rth_FULL.csv"

OUTPUT_SUMMARY     = RESULTS_DIR /"baseline_lstm_online_delta_summary.csv"
OUTPUT_PREDICTIONS = RESULTS_DIR /"baseline_lstm_online_delta_predictions.csv"
OUTPUT_REMOVED_DAYS = RESULTS_DIR / "lstm_online_delta_removed_days.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# HYPERPARAMETER SPACE
# =============================================================================

HP_SPACE = {
    "seq_len":              [16, 32, 64],
    "hidden_size":          [32, 64, 128],
    "num_layers":           [1, 2],
    "dropout":              [0.0, 0.1, 0.2, 0.4],
    "learning_rate":        [1e-4, 5e-4, 1e-3],
    "batch_size":           [64, 128],
    "epochs":               [10, 20],
    "online_learning_rate": [1e-5, 5e-5, 1e-4],
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# RESUME
# =============================================================================

def get_completed_runs(path: str) -> set:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        return set(zip(df["symbol"].astype(str), df["horizon"].astype(int)))
    except Exception:
        return set()


def append_row(path: str, row: dict) -> None:
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_header, index=False)


def append_predictions(path: str, pred_df: pd.DataFrame) -> None:
    write_header = not os.path.exists(path)
    pred_df.to_csv(path, mode="a", header=write_header, index=False)


# =============================================================================
# DATABASE
# =============================================================================

def load_dbt_profile() -> dict:
    profile_path = os.path.expanduser("~/.dbt/profiles.yml")
    with open(profile_path, "r") as f:
        profiles = yaml.safe_load(f)
    profile_name = list(profiles.keys())[0]
    profile      = profiles[profile_name]
    target       = profile["target"]
    return profile["outputs"][target]


def create_connection():
    cfg = load_dbt_profile()
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg.get("port", 5432),
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def _filter_full_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    daily_counts = df.groupby(["symbol", "date"]).size().reset_index(name="bars")
    full_days    = daily_counts[daily_counts["bars"] == EXPECTED_BARS_PER_DAY][["symbol", "date"]]
    removed      = daily_counts[daily_counts["bars"] != EXPECTED_BARS_PER_DAY].copy()
    removed.to_csv(OUTPUT_REMOVED_DAYS, index=False)
    df = df.merge(full_days, on=["symbol", "date"], how="inner").drop(columns=["date"])
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_from_csv() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(NY_TZ)
    df = df[df["symbol"].isin(SYMBOLS)].copy()
    start = pd.Timestamp(SNAPSHOT_START_NY, tz=NY_TZ)
    end   = pd.Timestamp(SNAPSHOT_END_NY,   tz=NY_TZ)
    df = df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()
    return df[["symbol", "datetime", "close"]].sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_from_db() -> pd.DataFrame:
    conn  = create_connection()
    query = f"""
        SELECT symbol, datetime, close FROM {TABLE_NAME}
        WHERE symbol = ANY(%s)
          AND datetime >= TIMESTAMP WITH TIME ZONE %s
          AND datetime <= TIMESTAMP WITH TIME ZONE %s
        ORDER BY symbol, datetime;
    """
    params = [SYMBOLS, f"{SNAPSHOT_START_NY} America/New_York", f"{SNAPSHOT_END_NY} America/New_York"]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(NY_TZ)
    return df[["symbol", "datetime", "close"]].sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_all_data() -> pd.DataFrame:
    print(f"[DATA] source={DATA_SOURCE.upper()}", end="  ")
    if DATA_SOURCE.lower() == "csv":
        df = load_from_csv()
    elif DATA_SOURCE.lower() == "db":
        df = load_from_db()
    else:
        raise ValueError("DATA_SOURCE must be 'csv' or 'db'.")
    if REQUIRE_FULL_RTH_DAYS:
        df = _filter_full_days(df)
    print(f"rows={len(df):,} | {df['datetime'].min()} → {df['datetime'].max()}")
    return df.reset_index(drop=True)


# =============================================================================
# TARGET AND SPLIT
# =============================================================================

def make_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    d           = df.copy()
    d["target"] = d["close"].shift(-horizon) - d["close"]
    return d.dropna(subset=["target"]).reset_index(drop=True)


def split_warmup_val_online(df: pd.DataFrame):
    n          = len(df)
    warmup_end = int(n * WARMUP_RATIO)
    val_end    = int(n * (WARMUP_RATIO + VAL_RATIO))
    return df.iloc[:warmup_end].copy(), df.iloc[warmup_end:val_end].copy(), df.iloc[val_end:].copy()


# =============================================================================
# DYNAMIC SCALER
# =============================================================================

def dynamic_scale(history: list, scale_window: int):
    window = history[-scale_window:] if len(history) >= scale_window else history
    mean   = float(np.mean(window))
    std    = float(np.std(window)) + 1e-8
    return mean, std


def normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std


def denormalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return values * std + mean


# =============================================================================
# DATASET  (warmup training only)
# =============================================================================

class DeltaSequenceDataset(Dataset):
    def __init__(
        self,
        close: np.ndarray,
        delta_target: np.ndarray,
        seq_len: int,
        close_mean: float,
        close_std: float,
        delta_mean: float,
        delta_std: float,
    ):
        close_norm  = normalize(close, close_mean, close_std)
        target_norm = normalize(delta_target, delta_mean, delta_std)

        X, y = [], []
        for i in range(seq_len, len(close_norm)):
            X.append(close_norm[i - seq_len: i])
            y.append(target_norm[i])

        self.X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# MODEL
# =============================================================================

class LSTMModel(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


# =============================================================================
# RANDOM SEARCH
# =============================================================================

def sample_hyperparams(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in HP_SPACE.items()}


def run_trial(
    hp: dict,
    warmup: pd.DataFrame,
    val: pd.DataFrame,
    warmup_close_mean: float,
    warmup_close_std: float,
    warmup_delta_mean: float,
    warmup_delta_std: float,
) -> float:
    set_seed(RANDOM_SEED)

    train_ds = DeltaSequenceDataset(
        warmup["close"].values, warmup["target"].values, hp["seq_len"],
        warmup_close_mean, warmup_close_std, warmup_delta_mean, warmup_delta_std,
    )

    if len(train_ds) == 0:
        return float("inf")

    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    model        = LSTMModel(hp["hidden_size"], hp["num_layers"], hp["dropout"]).to(DEVICE)
    optimizer    = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion    = nn.MSELoss()

    for _ in range(hp["epochs"]):
        train_one_epoch(model, train_loader, optimizer, criterion)

    # Evaluate on val with dynamic scaler and delayed online updates
    online_optimizer = torch.optim.Adam(model.parameters(), lr=hp["online_learning_rate"])
    history          = list(warmup["close"].values)
    close_vals       = val["close"].values
    delta_tgt        = val["target"].values
    preds, actuals   = [], []

    def _infer_horizon(frame: pd.DataFrame) -> int:
        close_arr = frame["close"].values.astype(float)
        tgt_arr   = frame["target"].values.astype(float)

        best_h = None
        best_score = float("inf")

        for h in TARGET_HORIZONS_BARS:
            if len(frame) <= h + 5:
                continue
            expected_delta = close_arr[h:] - close_arr[:-h]
            score = float(np.nanmedian(np.abs(tgt_arr[:-h] - expected_delta)))
            if score < best_score:
                best_score = score
                best_h = h

        if best_h is None:
            raise ValueError("Could not infer horizon from validation data.")

        return int(best_h)

    horizon = _infer_horizon(val)
    pending_updates = []

    model.eval()
    for t in range(len(val)):
        close_mean, close_std = dynamic_scale(history, SCALE_WINDOW)

        if len(history) >= SCALE_WINDOW + 1:
            diffs      = np.diff(history[-SCALE_WINDOW:])
            delta_mean = float(np.mean(diffs))
            delta_std  = float(np.std(diffs)) + 1e-8
        else:
            delta_mean = warmup_delta_mean
            delta_std  = warmup_delta_std

        window      = np.array(history[-hp["seq_len"]:], dtype=np.float32)
        window_norm = normalize(window, close_mean, close_std)
        x           = torch.tensor(window_norm).unsqueeze(0).unsqueeze(-1).to(DEVICE)

        with torch.no_grad():
            pred_norm  = model(x).item()
            pred_delta = denormalize(np.array([pred_norm]), delta_mean, delta_std)[0]

        preds.append(float(pred_delta))
        actuals.append(float(delta_tgt[t]))

        # Delayed update: store current h-step target, but do not use it until t+h.
        pending_updates.append({
            "release_t": t + horizon,
            "x": x.detach().cpu(),
            "target": float(delta_tgt[t]),
            "delta_mean": float(delta_mean),
            "delta_std": float(delta_std),
        })

        # Current close becomes observable only after the prediction step.
        history.append(float(close_vals[t]))

        # Update only samples whose horizon-ahead target would now be observable.
        matured = [u for u in pending_updates if u["release_t"] <= t]
        pending_updates = [u for u in pending_updates if u["release_t"] > t]

        for u in matured:
            x_old = u["x"].to(DEVICE)
            actual_norm = torch.tensor(
                [(u["target"] - u["delta_mean"]) / u["delta_std"]],
                dtype=torch.float32
            ).to(DEVICE)

            model.train()
            online_optimizer.zero_grad()
            loss = criterion(model(x_old), actual_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            online_optimizer.step()
            model.eval()

    return float(mean_absolute_error(actuals, preds))


def random_search(
    warmup: pd.DataFrame,
    val: pd.DataFrame,
    warmup_close_mean: float,
    warmup_close_std: float,
    warmup_delta_mean: float,
    warmup_delta_std: float,
    n_trials: int,
) -> dict:
    rng      = random.Random(RANDOM_SEED)
    best_hp  = None
    best_mae = float("inf")

    print(f"  [random_search] {n_trials} trials — Bergstra & Bengio (2012)")

    for t in range(n_trials):
        hp  = sample_hyperparams(rng)
        mae = run_trial(hp, warmup, val, warmup_close_mean, warmup_close_std, warmup_delta_mean, warmup_delta_std)
        flag = " *" if mae < best_mae else ""
        print(f"    trial {t+1:02d}/{n_trials} | val_mae={mae:.6f}{flag} | {hp}")
        if mae < best_mae:
            best_mae = mae
            best_hp  = hp

    print(f"  => best_val_mae={best_mae:.6f} | best_hp={best_hp}")
    return best_hp


# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

def train_final_model(
    hp: dict,
    warmup: pd.DataFrame,
    warmup_close_mean: float,
    warmup_close_std: float,
    warmup_delta_mean: float,
    warmup_delta_std: float,
) -> nn.Module:
    set_seed(RANDOM_SEED)

    train_ds = DeltaSequenceDataset(
        warmup["close"].values, warmup["target"].values, hp["seq_len"],
        warmup_close_mean, warmup_close_std, warmup_delta_mean, warmup_delta_std,
    )

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty.")

    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    model        = LSTMModel(hp["hidden_size"], hp["num_layers"], hp["dropout"]).to(DEVICE)
    optimizer    = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion    = nn.MSELoss()

    for epoch in range(hp["epochs"]):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{hp['epochs']} | train_loss={loss:.6f}")

    model.eval()
    return model


# =============================================================================
# ONLINE EVALUATION
# =============================================================================

def evaluate_online(
    model: nn.Module,
    warmup: pd.DataFrame,
    online: pd.DataFrame,
    hp: dict,
    warmup_delta_mean: float,
    warmup_delta_std: float,
    lookback: int,
) -> dict:
    """
    Walk-forward online evaluation with:
      1. Dynamic scaler for close (rolling window)
      2. Dynamic delta scaler (rolling window of 1-step diffs)
      3. Delayed online gradient update using the h-step delta target only
         after it would be observable.
    """
    seq_len          = hp["seq_len"]
    online_lr        = hp["online_learning_rate"]
    criterion        = nn.MSELoss()
    online_optimizer = torch.optim.Adam(model.parameters(), lr=online_lr)

    close_vals = online["close"].values
    delta_tgt  = online["target"].values
    history    = list(warmup["close"].values)

    pred_delta_all = []

    def _infer_horizon(frame: pd.DataFrame) -> int:
        close_arr = frame["close"].values.astype(float)
        tgt_arr   = frame["target"].values.astype(float)

        best_h = None
        best_score = float("inf")

        for h in TARGET_HORIZONS_BARS:
            if len(frame) <= h + 5:
                continue
            expected_delta = close_arr[h:] - close_arr[:-h]
            score = float(np.nanmedian(np.abs(tgt_arr[:-h] - expected_delta)))
            if score < best_score:
                best_score = score
                best_h = h

        if best_h is None:
            raise ValueError("Could not infer horizon from online data.")

        return int(best_h)

    horizon = _infer_horizon(online)
    pending_updates = []

    for t in range(len(online)):
        close_mean, close_std = dynamic_scale(history, SCALE_WINDOW)

        if len(history) >= SCALE_WINDOW + 1:
            diffs      = np.diff(history[-SCALE_WINDOW:])
            delta_mean = float(np.mean(diffs))
            delta_std  = float(np.std(diffs)) + 1e-8
        else:
            delta_mean = warmup_delta_mean
            delta_std  = warmup_delta_std

        window      = np.array(history[-seq_len:], dtype=np.float32)
        window_norm = normalize(window, close_mean, close_std)
        x           = torch.tensor(window_norm).unsqueeze(0).unsqueeze(-1).to(DEVICE)

        model.eval()
        with torch.no_grad():
            pred_norm  = model(x).item()
            pred_delta = denormalize(np.array([pred_norm]), delta_mean, delta_std)[0]

        pred_delta_all.append(float(pred_delta))

        # Delayed update: store current h-step delta target, but do not use it until t+h.
        pending_updates.append({
            "release_t": t + horizon,
            "x": x.detach().cpu(),
            "target": float(delta_tgt[t]),
            "delta_mean": float(delta_mean),
            "delta_std": float(delta_std),
        })

        # Current close becomes observable only after the prediction step.
        history.append(float(close_vals[t]))

        # Update only samples whose horizon-ahead target would now be observable.
        matured = [u for u in pending_updates if u["release_t"] <= t]
        pending_updates = [u for u in pending_updates if u["release_t"] > t]

        for u in matured:
            x_old = u["x"].to(DEVICE)
            actual_norm = torch.tensor(
                [(u["target"] - u["delta_mean"]) / u["delta_std"]],
                dtype=torch.float32
            ).to(DEVICE)

            model.train()
            online_optimizer.zero_grad()
            loss = criterion(model(x_old), actual_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            online_optimizer.step()
            model.eval()

    pred_delta_all = np.array(pred_delta_all, dtype=np.float64)

    return {
        "pred":         pred_delta_all[lookback:],
        "actual":       delta_tgt[lookback:],
        "close_prices": close_vals[lookback:],
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(out: dict) -> dict:
    pred, actual, close_prices = out["pred"], out["actual"], out["close_prices"]

    mae  = float(mean_absolute_error(actual, pred))
    rmse = float(math.sqrt(mean_squared_error(actual, pred)))

    delta_relative_mae_pct = float(
        mae / (np.mean(np.abs(actual)) + 1e-8) * 100.0
    )
    implied_delta_relative_mae_pct = float(
        mae / (np.mean(np.abs(close_prices)) + 1e-8) * 100.0
    )

    return {
        "delta_mae":                      mae,
        "delta_rmse":                     rmse,
        "delta_relative_mae_pct":         delta_relative_mae_pct,
        "implied_delta_relative_mae_pct": implied_delta_relative_mae_pct,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(RANDOM_SEED)
    df_raw = load_all_data()

    completed = get_completed_runs(OUTPUT_SUMMARY)
    if completed:
        print(f"[RESUME] {len(completed)} pair(s) already done: {completed}")

    print("\n" + "=" * 100)
    print("ONLINE LSTM BASELINE  |  next_delta  |  random search  |  v2 (leakage fixed)")
    print("=" * 100)
    print(f"symbols      = {SYMBOLS}")
    print(f"horizons     = {TARGET_HORIZONS_BARS}")
    print(f"snapshot     = {SNAPSHOT_START_NY} → {SNAPSHOT_END_NY}")
    print(f"split        = {WARMUP_RATIO}:{VAL_RATIO}:{ONLINE_RATIO}")
    print(f"lookback     = {LOOKBACK}")
    print(f"scale_window = {SCALE_WINDOW} bars (dynamic scaler)")
    print(f"n_trials     = {N_TRIALS}  (Bergstra & Bengio, 2012)")
    print(f"device       = {DEVICE}")
    print(f"seed         = {RANDOM_SEED}")
    print(f"output       = {OUTPUT_SUMMARY}")

    for symbol in SYMBOLS:
        df_sym = (
            df_raw[df_raw["symbol"] == symbol]
            .copy()
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        if len(df_sym) < 5000:
            print(f"\n[SKIP] {symbol}: only {len(df_sym):,} rows.")
            continue

        for horizon in TARGET_HORIZONS_BARS:

            if (symbol, horizon) in completed:
                print(f"\n[SKIP] {symbol} h={horizon} — already in output.")
                continue

            print("\n" + "-" * 100)
            print(f"symbol={symbol} | horizon={horizon}")

            d = make_target(df_sym, horizon)
            warmup, val, online = split_warmup_val_online(d)

            print(
                f"rows={len(d):,} | warmup={len(warmup):,} | "
                f"val={len(val):,} | online={len(online):,} | "
                f"eval_rows={len(online) - LOOKBACK:,}"
            )

            warmup_close_mean = float(np.mean(warmup["close"].values))
            warmup_close_std  = float(np.std(warmup["close"].values))  + 1e-8
            warmup_delta_mean = float(np.mean(warmup["target"].values))
            warmup_delta_std  = float(np.std(warmup["target"].values)) + 1e-8

            best_hp = random_search(
                warmup, val,
                warmup_close_mean, warmup_close_std,
                warmup_delta_mean, warmup_delta_std,
                N_TRIALS,
            )

            print("  [train] final model on warmup ...")
            try:
                model = train_final_model(
                    hp=best_hp,
                    warmup=warmup,
                    warmup_close_mean=warmup_close_mean,
                    warmup_close_std=warmup_close_std,
                    warmup_delta_mean=warmup_delta_mean,
                    warmup_delta_std=warmup_delta_std,
                )
            except Exception as e:
                print(f"[ERROR] final training failed: {e}")
                continue

            print("  [eval] online phase (dynamic scaler + online updates) ...")
            try:
                out = evaluate_online(
                    model=model,
                    warmup=warmup,
                    online=online,
                    hp=best_hp,
                    warmup_delta_mean=warmup_delta_mean,
                    warmup_delta_std=warmup_delta_std,
                    lookback=LOOKBACK,
                )
            except Exception as e:
                print(f"[ERROR] evaluation failed: {e}")
                continue

            m = compute_metrics(out)

            print(
                f"delta_mae={m['delta_mae']:.6f} | "
                f"delta_rmse={m['delta_rmse']:.6f} | "
                f"delta_rel_mae={m['delta_relative_mae_pct']:.4f}% | "
                f"implied_rel_mae={m['implied_delta_relative_mae_pct']:.4f}%"
            )

            result_row = {
                "symbol":                          symbol,
                "horizon":                         horizon,
                "model":                           "lstm_online",
                "best_seq_len":                    best_hp["seq_len"],
                "best_hidden_size":                best_hp["hidden_size"],
                "best_num_layers":                 best_hp["num_layers"],
                "best_dropout":                    best_hp["dropout"],
                "best_lr":                         best_hp["learning_rate"],
                "best_online_lr":                  best_hp["online_learning_rate"],
                "best_batch_size":                 best_hp["batch_size"],
                "best_epochs":                     best_hp["epochs"],
                "delta_mae":                       m["delta_mae"],
                "delta_rmse":                      m["delta_rmse"],
                "delta_relative_mae_pct":          m["delta_relative_mae_pct"],
                "implied_delta_relative_mae_pct":  m["implied_delta_relative_mae_pct"],
            }
            append_row(OUTPUT_SUMMARY, result_row)
            print(f"  => saved to {OUTPUT_SUMMARY}")

            pred_df = online.iloc[LOOKBACK:].reset_index(drop=True)[
                ["symbol", "datetime", "close", "target"]
            ].copy()
            pred_df["horizon"]      = horizon
            pred_df["model"]        = "lstm_online"
            pred_df["pred_delta"]   = out["pred"]
            pred_df["actual_delta"] = out["actual"]
            pred_df["error"]        = pred_df["actual_delta"] - pred_df["pred_delta"]
            pred_df["abs_error"]    = pred_df["error"].abs()

            append_predictions(OUTPUT_PREDICTIONS, pred_df)
            print(f"  => predictions saved to {OUTPUT_PREDICTIONS}")

    print("\n" + "=" * 100)
    print("DONE")
    if os.path.exists(OUTPUT_SUMMARY):
        summary = pd.read_csv(OUTPUT_SUMMARY)
        cols = [
            "symbol", "horizon", "model",
            "delta_mae", "delta_rmse",
            "delta_relative_mae_pct",
            "implied_delta_relative_mae_pct",
        ]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()