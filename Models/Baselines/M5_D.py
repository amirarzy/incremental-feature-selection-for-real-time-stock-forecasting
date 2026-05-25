#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Incremental Transformer Baseline  |  close-only  |  delta target
=================================================================
Architecture: Transformer encoder (no IFS, no EWC) — close price only.
Purpose     : Delta target baseline — predicts price change directly.

Scaler      : Dynamic — two separate rolling scalers:
                close_scaler : normalizes input window  (SCALE_WINDOW bars)
                delta_scaler : normalizes delta target  (SCALE_WINDOW bars)

Search      : Random search, N_TRIALS=15 — Bergstra & Bengio (2012).

Target      : delta = close[t+h] - close[t]
              metrics: MAE, RMSE, Relative MAE %

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

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

TABLE_NAME = "staging.canonical_1m_rth"
SYMBOLS    = ["AAPL", "MSFT", "NVDA", "TSLA"]

NY_TZ             = "America/New_York"
SNAPSHOT_START_NY = "2025-01-02 09:30:00"
SNAPSHOT_END_NY   = "2026-02-13 15:59:00"

REQUIRE_FULL_RTH_DAYS = True
EXPECTED_BARS_PER_DAY = 390
TARGET_HORIZONS_BARS  = [1, 30, 60, 240, 390]

WARMUP_RATIO = 0.20
VAL_RATIO    = 0.10
ONLINE_RATIO = 0.70

LOOKBACK     = 60
SCALE_WINDOW = 390
N_TRIALS     = 15
RANDOM_SEED  = 42

DATA_SOURCE = "csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "canonical_1m_rth_FULL.csv"

OUTPUT_SUMMARY     = RESULTS_DIR /"baseline_transformer_incremental_delta_summary.csv"
OUTPUT_PREDICTIONS = RESULTS_DIR /"baseline_transformer_incremental_delta_predictions.csv"
OUTPUT_REMOVED_DAYS = RESULTS_DIR / "transformer_incremental_delta_removed_days.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# HYPERPARAMETER SPACE
# Bergstra & Bengio (2012); Smith (2017); Keskar et al. (2017); Qian (2025)
# =============================================================================

HP_SPACE = {
    "seq_len":       [16, 32, 64],
    "d_model":       [32, 64, 128],
    "nhead":         [4],
    "enc_layers":    [1, 2],
    "ff_dim":        [64, 128, 256],
    "dropout":       [0.0, 0.1, 0.2],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size":    [64, 128],
    "epochs":        [10, 20],
    "online_lr":     [1e-5, 5e-5, 1e-4],
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
# DATABASE / CSV
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


def _filter_full_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["datetime"].dt.date
    daily   = df.groupby(["symbol", "date"]).size().reset_index(name="bars")
    full    = daily[daily["bars"] == EXPECTED_BARS_PER_DAY][["symbol", "date"]]
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
    params = [SYMBOLS,
              f"{SNAPSHOT_START_NY} America/New_York",
              f"{SNAPSHOT_END_NY} America/New_York"]
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
# TARGET & SPLIT
# =============================================================================

def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """delta_target[t] = close[t + horizon] - close[t]"""
    d = df.copy()
    d["delta_target"] = d["close"].shift(-horizon) - d["close"]
    return d.dropna(subset=["delta_target"]).reset_index(drop=True)


def split_warmup_val_online(df: pd.DataFrame):
    n  = len(df)
    we = int(n * WARMUP_RATIO)
    ve = int(n * (WARMUP_RATIO + VAL_RATIO))
    return df.iloc[:we].copy(), df.iloc[we:ve].copy(), df.iloc[ve:].copy()


# =============================================================================
# DYNAMIC SCALERS — separate for close input and delta target
# =============================================================================

def dynamic_scale(history: list, scale_window: int = SCALE_WINDOW):
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
# DATASET  (warmup — static scalers, no leakage)
# =============================================================================

class WarmupDataset(torch.utils.data.Dataset):
    def __init__(self, close, delta_target, seq_len,
                 close_mean, close_std, delta_mean, delta_std):
        c_norm = normalize(close,        close_mean, close_std)
        t_norm = normalize(delta_target, delta_mean, delta_std)
        X, y = [], []
        for i in range(seq_len, len(c_norm)):
            X.append(c_norm[i - seq_len: i])
            y.append(t_norm[i])
        self.X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):        return len(self.X)
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
    def __init__(self, d_model, nhead, enc_layers, ff_dim, dropout):
        super().__init__()
        self.proj    = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
            activation="relu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.fc      = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


# =============================================================================
# TRAINING HELPER
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


# =============================================================================
# VAL EVALUATION
# =============================================================================

def evaluate_val_mae(model, warmup_close, warmup_delta, val, hp) -> float:
    """
    Walk-forward validation with dynamic scalers + delayed online updates.

    Causal delayed update:
      - At step t, predict delta_target[t] = close[t+h] - close[t].
      - Store the input window, scaler stats, and h-step delta target.
      - Update only after h steps, when the corresponding target would be observable.
      - delta_hist is updated only with matured h-step targets, not future targets.
    """
    seq_len    = hp["seq_len"]
    criterion  = nn.MSELoss()
    online_opt = torch.optim.Adam(model.parameters(), lr=hp["online_lr"])

    close_hist = list(warmup_close)
    delta_hist = list(warmup_delta)

    close_vals = val["close"].values
    delta_tgt  = val["delta_target"].values
    preds, acts = [], []

    def _infer_horizon(frame: pd.DataFrame) -> int:
        if "horizon" in frame.columns:
            return int(frame["horizon"].iloc[0])

        close_arr = frame["close"].values.astype(float)
        tgt_arr   = frame["delta_target"].values.astype(float)

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
        c_mean, c_std = dynamic_scale(close_hist)
        d_mean, d_std = dynamic_scale(delta_hist)

        window = np.array(close_hist[-seq_len:], dtype=np.float32)
        x = torch.tensor(normalize(window, c_mean, c_std)
                         ).unsqueeze(0).unsqueeze(-1).to(DEVICE)

        with torch.no_grad():
            pred = denormalize(model(x).item(), d_mean, d_std)

        preds.append(float(pred))
        acts.append(float(delta_tgt[t]))

        # Store sample for delayed update. Do not update immediately with future h-step delta.
        pending_updates.append({
            "release_t": t + horizon,
            "x": x.detach().cpu(),
            "target": float(delta_tgt[t]),
            "d_mean": float(d_mean),
            "d_std": float(d_std),
        })

        # Current close becomes observable after this step.
        close_hist.append(float(close_vals[t]))

        # Update only samples whose horizon-ahead delta is now observable.
        matured = [u for u in pending_updates if u["release_t"] <= t]
        pending_updates = [u for u in pending_updates if u["release_t"] > t]

        for u in matured:
            x_old = u["x"].to(DEVICE)
            y_norm = torch.tensor(
                [normalize(u["target"], u["d_mean"], u["d_std"])],
                dtype=torch.float32
            ).to(DEVICE)

            model.train()
            online_opt.zero_grad()
            loss = criterion(model(x_old), y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            online_opt.step()
            model.eval()

            # The h-step delta target is now observed and can enter the delta scaler history.
            delta_hist.append(float(u["target"]))

    return float(mean_absolute_error(acts, preds))


# =============================================================================
# RANDOM SEARCH
# =============================================================================

def sample_hp(rng): return {k: rng.choice(v) for k, v in HP_SPACE.items()}


def run_trial(hp, warmup, val, wc_mean, wc_std, wd_mean, wd_std) -> float:
    if hp["d_model"] % hp["nhead"] != 0:
        return float("inf")
    set_seed(RANDOM_SEED)
    ds = WarmupDataset(warmup["close"].values, warmup["delta_target"].values,
                       hp["seq_len"], wc_mean, wc_std, wd_mean, wd_std)
    if len(ds) == 0:
        return float("inf")
    loader    = torch.utils.data.DataLoader(ds, hp["batch_size"], shuffle=True)
    model     = TransformerCloseOnly(hp["d_model"], hp["nhead"],
                                     hp["enc_layers"], hp["ff_dim"],
                                     hp["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion = nn.MSELoss()
    for _ in range(hp["epochs"]):
        train_one_epoch(model, loader, optimizer, criterion)
    return evaluate_val_mae(model, warmup["close"].values,
                            warmup["delta_target"].values, val, hp)


def random_search(warmup, val, wc_mean, wc_std, wd_mean, wd_std) -> dict:
    rng, best_hp, best_mae = random.Random(RANDOM_SEED), None, float("inf")
    print(f"  [random_search] {N_TRIALS} trials — Bergstra & Bengio (2012)")
    for t in range(N_TRIALS):
        hp  = sample_hp(rng)
        mae = run_trial(hp, warmup, val, wc_mean, wc_std, wd_mean, wd_std)
        flag = " *" if mae < best_mae else ""
        print(f"    trial {t+1:02d}/{N_TRIALS} | val_mae={mae:.6f}{flag} | {hp}")
        if mae < best_mae:
            best_mae, best_hp = mae, hp
    print(f"  => best_val_mae={best_mae:.6f} | best_hp={best_hp}")
    return best_hp


# =============================================================================
# TRAIN FINAL
# =============================================================================

def train_final(hp, warmup, wc_mean, wc_std, wd_mean, wd_std) -> nn.Module:
    set_seed(RANDOM_SEED)
    ds = WarmupDataset(warmup["close"].values, warmup["delta_target"].values,
                       hp["seq_len"], wc_mean, wc_std, wd_mean, wd_std)
    if len(ds) == 0:
        raise ValueError("Empty warmup dataset.")
    loader    = torch.utils.data.DataLoader(ds, hp["batch_size"], shuffle=True)
    model     = TransformerCloseOnly(hp["d_model"], hp["nhead"],
                                     hp["enc_layers"], hp["ff_dim"],
                                     hp["dropout"]).to(DEVICE)
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

def evaluate_online(model, warmup_close, warmup_delta,
                    online, hp, lookback) -> dict:
    """
    Walk-forward online evaluation with dynamic scalers + delayed online updates.

    Causal delayed update:
      - At step t, predict delta_target[t] = close[t+h] - close[t].
      - Store the input window, scaler stats, and h-step delta target.
      - Update only after h steps, when the corresponding target would be observable.
      - delta_hist is updated only with matured h-step targets, not future targets.
    """
    seq_len    = hp["seq_len"]
    online_opt = torch.optim.Adam(model.parameters(), lr=hp["online_lr"])
    criterion  = nn.MSELoss()

    close_vals = online["close"].values
    delta_tgt  = online["delta_target"].values
    close_hist = list(warmup_close)
    delta_hist = list(warmup_delta)
    pred_all   = []

    def _infer_horizon(frame: pd.DataFrame) -> int:
        if "horizon" in frame.columns:
            return int(frame["horizon"].iloc[0])

        close_arr = frame["close"].values.astype(float)
        tgt_arr   = frame["delta_target"].values.astype(float)

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
        c_mean, c_std = dynamic_scale(close_hist)
        d_mean, d_std = dynamic_scale(delta_hist)

        window = np.array(close_hist[-seq_len:], dtype=np.float32)
        x = torch.tensor(normalize(window, c_mean, c_std)
                         ).unsqueeze(0).unsqueeze(-1).to(DEVICE)

        model.eval()
        with torch.no_grad():
            pred = denormalize(model(x).item(), d_mean, d_std)
        pred_all.append(float(pred))

        # Store sample for delayed update. Do not update immediately with future h-step delta.
        pending_updates.append({
            "release_t": t + horizon,
            "x": x.detach().cpu(),
            "target": float(delta_tgt[t]),
            "d_mean": float(d_mean),
            "d_std": float(d_std),
        })

        # Current close becomes observable after this step.
        close_hist.append(float(close_vals[t]))

        # Update only matured samples.
        matured = [u for u in pending_updates if u["release_t"] <= t]
        pending_updates = [u for u in pending_updates if u["release_t"] > t]

        for u in matured:
            x_old = u["x"].to(DEVICE)
            y_norm = torch.tensor(
                [normalize(u["target"], u["d_mean"], u["d_std"])],
                dtype=torch.float32
            ).to(DEVICE)

            model.train()
            online_opt.zero_grad()
            loss = criterion(model(x_old), y_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            online_opt.step()
            model.eval()

            # The h-step delta target is now observed and can enter the delta scaler history.
            delta_hist.append(float(u["target"]))

    pred_all = np.array(pred_all, dtype=np.float64)
    return {
        "pred_delta":   pred_all[lookback:],
        "actual_delta": delta_tgt[lookback:],
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(out: dict) -> dict:
    pred   = out["pred_delta"]
    actual = out["actual_delta"]
    mae    = float(mean_absolute_error(actual, pred))
    rmse   = float(math.sqrt(mean_squared_error(actual, pred)))
    rel    = float(mae / (np.mean(np.abs(actual)) + 1e-8) * 100.0)
    return {
        "delta_mae":              mae,
        "delta_rmse":             rmse,
        "delta_relative_mae_pct": rel,
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
    print("INCREMENTAL TRANSFORMER BASELINE  |  close-only  |  delta target")
    print("=" * 100)
    print(f"symbols      = {SYMBOLS}")
    print(f"horizons     = {TARGET_HORIZONS_BARS}")
    print(f"snapshot     = {SNAPSHOT_START_NY} → {SNAPSHOT_END_NY}")
    print(f"split        = {WARMUP_RATIO}:{VAL_RATIO}:{ONLINE_RATIO}")
    print(f"lookback     = {LOOKBACK}")
    print(f"scale_window = {SCALE_WINDOW}")
    print(f"n_trials     = {N_TRIALS}")
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

            wc_mean = float(np.mean(warmup["close"].values))
            wc_std  = float(np.std(warmup["close"].values))  + 1e-8
            wd_mean = float(np.mean(warmup["delta_target"].values))
            wd_std  = float(np.std(warmup["delta_target"].values)) + 1e-8

            best_hp = random_search(warmup, val,
                                    wc_mean, wc_std, wd_mean, wd_std)

            print("  [train] final model on warmup ...")
            try:
                model = train_final(best_hp, warmup,
                                    wc_mean, wc_std, wd_mean, wd_std)
            except Exception as e:
                print(f"[ERROR] training failed: {e}")
                continue

            print("  [eval] online phase ...")
            try:
                out = evaluate_online(
                    model=model,
                    warmup_close=warmup["close"].values,
                    warmup_delta=warmup["delta_target"].values,
                    online=online,
                    hp=best_hp,
                    lookback=LOOKBACK,
                )
            except Exception as e:
                print(f"[ERROR] evaluation failed: {e}")
                continue

            m = compute_metrics(out)

            print(f"delta_mae={m['delta_mae']:.6f} | "
                  f"delta_rmse={m['delta_rmse']:.6f} | "
                  f"delta_rel_mae={m['delta_relative_mae_pct']:.4f}%")

            append_row(OUTPUT_SUMMARY, {
                "symbol":                  symbol,
                "horizon":                 horizon,
                "model":                   "transformer_incremental_delta",
                "best_seq_len":            best_hp["seq_len"],
                "best_d_model":            best_hp["d_model"],
                "best_nhead":              best_hp["nhead"],
                "best_enc_layers":         best_hp["enc_layers"],
                "best_ff_dim":             best_hp["ff_dim"],
                "best_dropout":            best_hp["dropout"],
                "best_lr":                 best_hp["learning_rate"],
                "best_online_lr":          best_hp["online_lr"],
                "best_batch_size":         best_hp["batch_size"],
                "best_epochs":             best_hp["epochs"],
                "delta_mae":               m["delta_mae"],
                "delta_rmse":              m["delta_rmse"],
                "delta_relative_mae_pct":  m["delta_relative_mae_pct"],
            })
            print(f"  => saved to {OUTPUT_SUMMARY}")

            pred_df = online.iloc[LOOKBACK:].reset_index(drop=True)[
                ["symbol", "datetime", "close", "delta_target"]
            ].copy()
            pred_df["horizon"]         = horizon
            pred_df["model"]           = "transformer_incremental_delta"
            pred_df["pred_delta"]      = out["pred_delta"]
            pred_df["actual_delta"]    = out["actual_delta"]
            pred_df["delta_error"]     = pred_df["actual_delta"] - pred_df["pred_delta"]
            pred_df["delta_abs_error"] = pred_df["delta_error"].abs()

            append_predictions(OUTPUT_PREDICTIONS, pred_df)
            print(f"  => predictions saved to {OUTPUT_PREDICTIONS}")

    print("\n" + "=" * 100)
    print("DONE")
    if os.path.exists(OUTPUT_SUMMARY):
        summary = pd.read_csv(OUTPUT_SUMMARY)
        cols = ["symbol", "horizon", "model",
                "delta_mae", "delta_rmse", "delta_relative_mae_pct"]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()