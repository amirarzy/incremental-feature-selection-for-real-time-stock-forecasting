#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Incremental Transformer Baseline  |  close-only  |  price + direction targets
==============================================================================
Architecture: Transformer encoder (no IFS, no EWC) — close price only.
Purpose     : Isolates architecture effect from feature and IFS effects.
              Paired with Online LSTM (same close-only input) and
              IL-ETransformer ifs_ewc (6 features + IFS) for ablation.

Scaler      : Dynamic — rolling mean/std over SCALE_WINDOW bars (1 session).
              Matches Online LSTM scaling strategy.

Search      : Random search, N_TRIALS=15 — Bergstra & Bengio (2012).
              Same seed and budget as LSTM baselines for fair comparison.

Targets     :
  price     : next_close = close[t + h]
              metrics: MAE, RMSE, Relative MAE %
  direction : sign(close[t+h] - close[t])  → 1 (up) / 0 (down)
              metrics: Accuracy, F1 weighted, Precision, Recall

Evaluation  : Online phase (70%) — walk-forward, no lookahead.
              First LOOKBACK=60 steps skipped to match IL-ETransformer.
              One gradient step after each prediction (online_lr).
"""

import os
import math
import random
import warnings
import yaml

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

TABLE_NAME = "staging.canonical_1m_rth"

SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA"]

NY_TZ             = "America/New_York"
SNAPSHOT_START_NY = "2025-01-02 09:30:00"
SNAPSHOT_END_NY   = "2026-02-13 15:59:00"

REQUIRE_FULL_RTH_DAYS = True
EXPECTED_BARS_PER_DAY = 390

TARGET_HORIZONS_BARS = [1, 30, 60, 240, 390]

WARMUP_RATIO = 0.20
VAL_RATIO    = 0.10
ONLINE_RATIO = 0.70

# Must match IL-ETransformer — first LOOKBACK steps of online phase skipped
LOOKBACK = 60

# Dynamic scaler window — one trading session (matches Online LSTM)
SCALE_WINDOW = 390

# Random search — Bergstra & Bengio (2012)
N_TRIALS    = 15
RANDOM_SEED = 42

DATA_SOURCE = "csv"   # "db" or "csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "canonical_1m_rth_FULL.csv"

OUTPUT_SUMMARY      = RESULTS_DIR /"baseline_transformer_incremental_summary.csv"
OUTPUT_PREDICTIONS  = RESULTS_DIR /"baseline_transformer_incremental_predictions.csv"
OUTPUT_REMOVED_DAYS = RESULTS_DIR / "transformer_incremental_removed_days.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# HYPERPARAMETER SPACE
# Justified per Online LSTM space (Bergstra & Bengio, 2012;
# Smith, 2017; Keskar et al., 2017; Qian, 2025)
# =============================================================================

HP_SPACE = {
    "seq_len":       [16, 32, 64],        # = seq_len in LSTM baselines
    "d_model":       [32, 64, 128],       # equivalent to hidden_size in LSTM
    "nhead":         [4],                 # fixed: d_model must be divisible
    "enc_layers":    [1, 2],              # = num_layers in LSTM
    "ff_dim":        [64, 128, 256],      # 2-4x d_model — standard Transformer
    "dropout":       [0.0, 0.1, 0.2],    # = LSTM dropout range
    "learning_rate": [1e-4, 5e-4, 1e-3], # = LSTM lr range — Smith (2017)
    "batch_size":    [64, 128],           # = LSTM batch range — Keskar (2017)
    "epochs":        [10, 20],            # = LSTM epochs
    "online_lr":     [1e-5, 5e-5, 1e-4], # = Online LSTM online_lr — Qian (2025)
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
    path = os.path.expanduser("~/.dbt/profiles.yml")
    with open(path) as f:
        profiles = yaml.safe_load(f)
    profile = profiles[list(profiles.keys())[0]]
    return profile["outputs"][profile["target"]]


def create_connection():
    cfg = load_dbt_profile()
    return psycopg2.connect(
        host=cfg["host"], port=cfg.get("port", 5432),
        dbname=cfg["dbname"], user=cfg["user"], password=cfg["password"],
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def _filter_full_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    daily = df.groupby(["symbol", "date"]).size().reset_index(name="bars")
    full  = daily[daily["bars"] == EXPECTED_BARS_PER_DAY][["symbol", "date"]]
    removed = daily[daily["bars"] != EXPECTED_BARS_PER_DAY].copy()
    removed.to_csv(OUTPUT_REMOVED_DAYS, index=False)
    df = df.merge(full, on=["symbol", "date"], how="inner").drop(columns=["date"])
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_from_db() -> pd.DataFrame:
    conn  = create_connection()
    query = f"""
        SELECT symbol, datetime, close
        FROM   {TABLE_NAME}
        WHERE  symbol = ANY(%s)
          AND  datetime >= TIMESTAMP WITH TIME ZONE %s
          AND  datetime <= TIMESTAMP WITH TIME ZONE %s
        ORDER BY symbol, datetime
    """
    params = [
        SYMBOLS,
        f"{SNAPSHOT_START_NY} America/New_York",
        f"{SNAPSHOT_END_NY} America/New_York",
    ]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(NY_TZ)
    return df[["symbol", "datetime", "close"]].sort_values(
        ["symbol", "datetime"]).reset_index(drop=True)


def load_from_csv() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(NY_TZ)
    df = df[df["symbol"].isin(SYMBOLS)].copy()
    start = pd.Timestamp(SNAPSHOT_START_NY, tz=NY_TZ)
    end   = pd.Timestamp(SNAPSHOT_END_NY,   tz=NY_TZ)
    df    = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
    return df[["symbol", "datetime", "close"]].sort_values(
        ["symbol", "datetime"]).reset_index(drop=True)


def load_all_data() -> pd.DataFrame:
    print(f"[DATA] source={DATA_SOURCE.upper()}", end="  ")
    df = load_from_db() if DATA_SOURCE == "db" else load_from_csv()
    if REQUIRE_FULL_RTH_DAYS:
        df = _filter_full_days(df)
    print(f"rows={len(df):,} | {df['datetime'].min()} → {df['datetime'].max()}")
    return df.reset_index(drop=True)


# =============================================================================
# TARGETS & SPLIT
# =============================================================================

def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    price_target[t]     = close[t + horizon]
    direction_target[t] = 1 if close[t+horizon] > close[t] else 0
    """
    d = df.copy()
    future               = d["close"].shift(-horizon)
    d["price_target"]    = future
    d["direction_target"]= (future > d["close"]).astype(int)
    return d.dropna(subset=["price_target"]).reset_index(drop=True)


def split_warmup_val_online(df: pd.DataFrame):
    n = len(df)
    we = int(n * WARMUP_RATIO)
    ve = int(n * (WARMUP_RATIO + VAL_RATIO))
    return df.iloc[:we].copy(), df.iloc[we:ve].copy(), df.iloc[ve:].copy()


# =============================================================================
# DYNAMIC SCALER  (matches Online LSTM)
# =============================================================================

def dynamic_scale(history: list, scale_window: int = SCALE_WINDOW):
    """
    Compute mean and std from the most recent scale_window observations.
    Applied at each online step to normalize inputs and targets.
    """
    w    = history[-scale_window:] if len(history) >= scale_window else history
    mean = float(np.mean(w))
    std  = float(np.std(w)) + 1e-8
    return mean, std


def normalize(value, mean: float, std: float):
    if isinstance(value, np.ndarray):
        return (value - mean) / std
    return (float(value) - mean) / std


def denormalize(value, mean: float, std: float):
    if isinstance(value, np.ndarray):
        return value * std + mean
    return float(value) * std + mean


# =============================================================================
# DATASET  (warmup training — static scaler fitted on warmup close)
# =============================================================================

class WarmupDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset for warmup training.
    Uses static mean/std computed from warmup close prices (no leakage).
    """
    def __init__(self, close: np.ndarray, price_target: np.ndarray,
                 seq_len: int, mean: float, std: float):
        c_norm = normalize(close, mean, std)
        t_norm = normalize(price_target, mean, std)
        X, y = [], []
        for i in range(seq_len, len(c_norm)):
            X.append(c_norm[i - seq_len: i])
            y.append(t_norm[i])
        self.X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# =============================================================================
# MODEL
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerCloseOnly(nn.Module):
    """
    Transformer encoder regressor — close price only (input_size=1).
    No IFS, no EWC, no additional features.
    """
    def __init__(self, d_model: int, nhead: int, enc_layers: int,
                 ff_dim: int, dropout: float):
        super().__init__()
        self.proj    = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


# =============================================================================
# TRAINING HELPERS
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


def evaluate_val_mae(model, warmup_close, val, hp) -> float:
    """
    Walk-forward validation with dynamic scaler + delayed online updates.
    Mirrors Online LSTM val evaluation for fair hyperparameter comparison.

    Causal delayed update:
      - At step t, predict price_target[t] = close[t+h].
      - Store the input window, scaler stats, and realised target.
      - Update only after h steps, when close[t+h] would be observable.
    """
    seq_len    = hp["seq_len"]
    criterion  = nn.MSELoss()
    online_opt = torch.optim.Adam(model.parameters(), lr=hp["online_lr"])

    history     = list(warmup_close)
    close_vals  = val["close"].values
    price_tgt   = val["price_target"].values
    preds, acts = [], []

    def _infer_horizon(frame: pd.DataFrame) -> int:
        if "horizon" in frame.columns:
            return int(frame["horizon"].iloc[0])

        close_arr = frame["close"].values.astype(float)
        tgt_arr   = frame["price_target"].values.astype(float)

        best_h = None
        best_score = float("inf")

        for h in TARGET_HORIZONS_BARS:
            if len(frame) <= h + 5:
                continue
            score = float(np.nanmedian(np.abs(tgt_arr[:-h] - close_arr[h:])))
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
        mean, std = dynamic_scale(history)
        window    = np.array(history[-seq_len:], dtype=np.float32)
        x = torch.tensor(normalize(window, mean, std)
                         ).unsqueeze(0).unsqueeze(-1).to(DEVICE)

        with torch.no_grad():
            pred = denormalize(model(x).item(), mean, std)

        preds.append(pred)
        acts.append(float(price_tgt[t]))

        # Store this sample for delayed update. Do not update immediately
        # with price_tgt[t], because price_tgt[t] = close[t+h].
        pending_updates.append({
            "release_t": t + horizon,
            "x": x.detach().cpu(),
            "target": float(price_tgt[t]),
            "mean": float(mean),
            "std": float(std),
        })

        # Current close becomes observable after the prediction.
        history.append(float(close_vals[t]))

        # Update only samples whose horizon-ahead target would now be observable.
        matured = [u for u in pending_updates if u["release_t"] <= t]
        pending_updates = [u for u in pending_updates if u["release_t"] > t]

        for u in matured:
            x_old = u["x"].to(DEVICE)
            y_norm = torch.tensor(
                [normalize(float(u["target"]), u["mean"], u["std"])],
                dtype=torch.float32
            ).to(DEVICE)

            model.train()
            online_opt.zero_grad()
            criterion(model(x_old), y_norm).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            online_opt.step()
            model.eval()

    return float(mean_absolute_error(acts, preds))


# =============================================================================
# RANDOM SEARCH  —  Bergstra & Bengio (2012)
# =============================================================================

def sample_hp(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in HP_SPACE.items()}


def run_trial(hp: dict, warmup: pd.DataFrame, val: pd.DataFrame,
              warmup_mean: float, warmup_std: float) -> float:
    """Train on warmup with static scaler; evaluate on val with dynamic scaler."""
    if hp["d_model"] % hp["nhead"] != 0:
        return float("inf")

    set_seed(RANDOM_SEED)

    ds = WarmupDataset(warmup["close"].values, warmup["price_target"].values,
                       hp["seq_len"], warmup_mean, warmup_std)
    if len(ds) == 0:
        return float("inf")

    loader    = torch.utils.data.DataLoader(ds, hp["batch_size"], shuffle=True)
    model     = TransformerCloseOnly(
        hp["d_model"], hp["nhead"], hp["enc_layers"],
        hp["ff_dim"],  hp["dropout"]
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion = nn.MSELoss()

    for _ in range(hp["epochs"]):
        train_one_epoch(model, loader, optimizer, criterion)

    return evaluate_val_mae(model, warmup["close"].values, val, hp)


def random_search(warmup: pd.DataFrame, val: pd.DataFrame,
                  warmup_mean: float, warmup_std: float) -> dict:
    rng      = random.Random(RANDOM_SEED)
    best_hp  = None
    best_mae = float("inf")

    print(f"  [random_search] {N_TRIALS} trials — Bergstra & Bengio (2012)")

    for t in range(N_TRIALS):
        hp  = sample_hp(rng)
        mae = run_trial(hp, warmup, val, warmup_mean, warmup_std)
        flag = " *" if mae < best_mae else ""
        print(f"    trial {t+1:02d}/{N_TRIALS} | val_mae={mae:.6f}{flag} | {hp}")
        if mae < best_mae:
            best_mae = mae
            best_hp  = hp

    print(f"  => best_val_mae={best_mae:.6f} | best_hp={best_hp}")
    return best_hp


# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

def train_final(hp: dict, warmup: pd.DataFrame,
                warmup_mean: float, warmup_std: float) -> nn.Module:
    set_seed(RANDOM_SEED)
    ds = WarmupDataset(warmup["close"].values, warmup["price_target"].values,
                       hp["seq_len"], warmup_mean, warmup_std)
    if len(ds) == 0:
        raise ValueError("Empty warmup dataset.")

    loader    = torch.utils.data.DataLoader(ds, hp["batch_size"], shuffle=True)
    model     = TransformerCloseOnly(
        hp["d_model"], hp["nhead"], hp["enc_layers"],
        hp["ff_dim"],  hp["dropout"]
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion = nn.MSELoss()

    for epoch in range(hp["epochs"]):
        loss = train_one_epoch(model, loader, optimizer, criterion)
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{hp['epochs']} | train_loss={loss:.6f}")

    model.eval()
    return model


# =============================================================================
# ONLINE EVALUATION
# =============================================================================

def evaluate_online(model: nn.Module, warmup_close: np.ndarray,
                    online: pd.DataFrame, hp: dict, lookback: int) -> dict:
    """
    Walk-forward evaluation on the online phase with:
      1. Dynamic scaler  — rolling SCALE_WINDOW
      2. Delayed online update — one gradient step only after target availability

    Direction is derived as sign(pred_price - close[t]),
    matching IL-ETransformer evaluation convention.

    First `lookback` steps are skipped to match IL-ETransformer.
    """
    seq_len    = hp["seq_len"]
    online_opt = torch.optim.Adam(model.parameters(), lr=hp["online_lr"])
    criterion  = nn.MSELoss()

    close_vals = online["close"].values
    price_tgt  = online["price_target"].values
    dir_tgt    = online["direction_target"].values

    history         = list(warmup_close)
    pred_price_all  = []

    def _infer_horizon(frame: pd.DataFrame) -> int:
        if "horizon" in frame.columns:
            return int(frame["horizon"].iloc[0])

        close_arr = frame["close"].values.astype(float)
        tgt_arr   = frame["price_target"].values.astype(float)

        best_h = None
        best_score = float("inf")

        for h in TARGET_HORIZONS_BARS:
            if len(frame) <= h + 5:
                continue
            score = float(np.nanmedian(np.abs(tgt_arr[:-h] - close_arr[h:])))
            if score < best_score:
                best_score = score
                best_h = h

        if best_h is None:
            raise ValueError("Could not infer horizon from online data.")

        return int(best_h)

    horizon = _infer_horizon(online)
    pending_updates = []

    for t in range(len(online)):
        mean, std = dynamic_scale(history)
        window    = np.array(history[-seq_len:], dtype=np.float32)
        x = torch.tensor(normalize(window, mean, std)
                         ).unsqueeze(0).unsqueeze(-1).to(DEVICE)

        # Predict
        model.eval()
        with torch.no_grad():
            pred_price = denormalize(model(x).item(), mean, std)

        pred_price_all.append(float(pred_price))

        # Store this sample for delayed update. Do not update immediately
        # with price_tgt[t], because price_tgt[t] = close[t+h].
        pending_updates.append({
            "release_t": t + horizon,
            "x": x.detach().cpu(),
            "target": float(price_tgt[t]),
            "mean": float(mean),
            "std": float(std),
        })

        # Current close becomes observable after the prediction.
        history.append(float(close_vals[t]))

        # Update only samples whose horizon-ahead target would now be observable.
        matured = [u for u in pending_updates if u["release_t"] <= t]
        pending_updates = [u for u in pending_updates if u["release_t"] > t]

        for u in matured:
            x_old = u["x"].to(DEVICE)
            y_norm = torch.tensor(
                [normalize(float(u["target"]), u["mean"], u["std"])],
                dtype=torch.float32
            ).to(DEVICE)

            model.train()
            online_opt.zero_grad()
            criterion(model(x_old), y_norm).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            online_opt.step()
            model.eval()

    pred_price_all = np.array(pred_price_all, dtype=np.float64)
    pred_dir_all   = (pred_price_all > close_vals).astype(int)

    return {
        "pred_price":       pred_price_all[lookback:],
        "actual_price":     price_tgt[lookback:],
        "pred_direction":   pred_dir_all[lookback:],
        "actual_direction": dir_tgt[lookback:],
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(out: dict) -> dict:
    pred_price    = out["pred_price"]
    actual_price  = out["actual_price"]
    pred_dir      = out["pred_direction"]
    actual_dir    = out["actual_direction"]

    mae  = float(mean_absolute_error(actual_price, pred_price))
    rmse = float(math.sqrt(mean_squared_error(actual_price, pred_price)))
    rel  = float(mae / (np.mean(np.abs(actual_price)) + 1e-8) * 100.0)

    acc  = float(accuracy_score(actual_dir, pred_dir))
    f1   = float(f1_score(actual_dir, pred_dir, zero_division=0))
    prec = float(precision_score(actual_dir, pred_dir, zero_division=0))
    rec  = float(recall_score(actual_dir, pred_dir, zero_division=0))

    return {
        "price_mae":              mae,
        "price_rmse":             rmse,
        "price_relative_mae_pct": rel,
        "direction_accuracy":     acc,
        "direction_f1":           f1,
        "direction_precision":    prec,
        "direction_recall":       rec,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(RANDOM_SEED)
    df_raw = load_all_data()

    completed = get_completed_runs(OUTPUT_SUMMARY)
    if completed:
        print(f"[RESUME] {len(completed)} pair(s) done: {completed}")

    print("\n" + "=" * 100)
    print("INCREMENTAL TRANSFORMER BASELINE  |  close-only  |  price + direction")
    print("=" * 100)
    print(f"symbols      = {SYMBOLS}")
    print(f"horizons     = {TARGET_HORIZONS_BARS}")
    print(f"snapshot     = {SNAPSHOT_START_NY} → {SNAPSHOT_END_NY}")
    print(f"split        = {WARMUP_RATIO}:{VAL_RATIO}:{ONLINE_RATIO}")
    print(f"lookback     = {LOOKBACK}")
    print(f"scale_window = {SCALE_WINDOW}  (dynamic — 1 trading session)")
    print(f"n_trials     = {N_TRIALS}  (Bergstra & Bengio, 2012)")
    print(f"device       = {DEVICE}")
    print(f"seed         = {RANDOM_SEED}")

    for symbol in SYMBOLS:
        df_sym = (df_raw[df_raw["symbol"] == symbol]
                  .copy().sort_values("datetime").reset_index(drop=True))

        if len(df_sym) < 5000:
            print(f"\n[SKIP] {symbol}: only {len(df_sym):,} rows.")
            continue

        for horizon in TARGET_HORIZONS_BARS:

            if (symbol, horizon) in completed:
                print(f"\n[SKIP] {symbol} h={horizon} — already done.")
                continue

            print("\n" + "-" * 100)
            print(f"symbol={symbol} | horizon={horizon}")

            d = make_targets(df_sym, horizon)
            warmup, val, online = split_warmup_val_online(d)

            print(f"rows={len(d):,} | warmup={len(warmup):,} | "
                  f"val={len(val):,} | online={len(online):,} | "
                  f"eval_rows={len(online) - LOOKBACK:,}")

            # Static scaler for warmup training (no data leakage)
            warmup_mean = float(np.mean(warmup["close"].values))
            warmup_std  = float(np.std(warmup["close"].values)) + 1e-8

            # Random search on val (dynamic scaler + online updates)
            best_hp = random_search(warmup, val, warmup_mean, warmup_std)

            # Train final model on warmup
            print("  [train] final model on warmup ...")
            try:
                model = train_final(best_hp, warmup, warmup_mean, warmup_std)
            except Exception as e:
                print(f"[ERROR] training failed: {e}")
                continue

            # Evaluate on online phase
            print("  [eval] online phase ...")
            try:
                out = evaluate_online(
                    model=model,
                    warmup_close=warmup["close"].values,
                    online=online,
                    hp=best_hp,
                    lookback=LOOKBACK,
                )
            except Exception as e:
                print(f"[ERROR] evaluation failed: {e}")
                continue

            m = compute_metrics(out)

            print(
                f"price_mae={m['price_mae']:.6f} | "
                f"price_rmse={m['price_rmse']:.6f} | "
                f"price_rel_mae={m['price_relative_mae_pct']:.4f}% | "
                f"dir_acc={m['direction_accuracy']:.4f} | "
                f"dir_f1={m['direction_f1']:.4f} | "
                f"precision={m['direction_precision']:.4f} | "
                f"recall={m['direction_recall']:.4f}"
            )

            # Save summary row
            result_row = {
                "symbol":                   symbol,
                "horizon":                  horizon,
                "model":                    "transformer_incremental",
                "best_seq_len":             best_hp["seq_len"],
                "best_d_model":             best_hp["d_model"],
                "best_nhead":               best_hp["nhead"],
                "best_enc_layers":          best_hp["enc_layers"],
                "best_ff_dim":              best_hp["ff_dim"],
                "best_dropout":             best_hp["dropout"],
                "best_lr":                  best_hp["learning_rate"],
                "best_online_lr":           best_hp["online_lr"],
                "best_batch_size":          best_hp["batch_size"],
                "best_epochs":              best_hp["epochs"],
                "price_mae":                m["price_mae"],
                "price_rmse":               m["price_rmse"],
                "price_relative_mae_pct":   m["price_relative_mae_pct"],
                "direction_accuracy":       m["direction_accuracy"],
                "direction_f1":             m["direction_f1"],
                "direction_precision":      m["direction_precision"],
                "direction_recall":         m["direction_recall"],
            }
            append_row(OUTPUT_SUMMARY, result_row)
            print(f"  => saved to {OUTPUT_SUMMARY}")

            # Save prediction rows
            pred_df = online.iloc[LOOKBACK:].reset_index(drop=True)[
                ["symbol", "datetime", "close", "price_target", "direction_target"]
            ].copy()
            pred_df["horizon"]           = horizon
            pred_df["model"]             = "transformer_incremental"
            pred_df["pred_price"]        = out["pred_price"]
            pred_df["actual_price"]      = out["actual_price"]
            pred_df["pred_direction"]    = out["pred_direction"]
            pred_df["actual_direction"]  = out["actual_direction"]
            pred_df["price_error"]       = (pred_df["actual_price"]
                                            - pred_df["pred_price"])
            pred_df["price_abs_error"]   = pred_df["price_error"].abs()
            pred_df["direction_correct"] = (
                pred_df["pred_direction"] == pred_df["actual_direction"]
            ).astype(int)

            append_predictions(OUTPUT_PREDICTIONS, pred_df)
            print(f"  => predictions saved to {OUTPUT_PREDICTIONS}")

    print("\n" + "=" * 100)
    print("DONE")
    if os.path.exists(OUTPUT_SUMMARY):
        summary = pd.read_csv(OUTPUT_SUMMARY)
        cols = ["symbol", "horizon", "model",
                "price_mae", "price_rmse", "price_relative_mae_pct",
                "direction_accuracy", "direction_f1",
                "direction_precision", "direction_recall"]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()