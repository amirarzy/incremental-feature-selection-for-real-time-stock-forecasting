#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M1-D — Naive Baseline | next_delta target
=========================================
Baseline:
  M1-D: naive_zero_delta
        pred_delta[t] = 0

This is the delta-space equivalent of the price persistence baseline:
  pred_price[t] = close[t]
  therefore pred_delta[t] = pred_price[t] - close[t] = 0

Evaluation:
  Online phase only.
  First LOOKBACK=60 rows skipped to match IL-ETransformer.

Outputs:
  baseline_m1_delta_summary.csv
  baseline_m1_delta_predictions.csv
"""

import os
import math
import yaml
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error


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

LOOKBACK = 60

DATA_SOURCE = "csv"   # "db" or "csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "canonical_1m_rth_FULL.csv"

OUTPUT_SUMMARY      = RESULTS_DIR / "baseline_m1_delta_summary.csv"
OUTPUT_PREDICTIONS  = RESULTS_DIR / "baseline_m1_delta_predictions.csv"
OUTPUT_REMOVED_DAYS = RESULTS_DIR / "m1_delta_removed_days.csv"


# =============================================================================
# DATABASE
# =============================================================================

def load_dbt_profile() -> dict:
    path = os.path.expanduser("~/.dbt/profiles.yml")
    with open(path, "r") as f:
        profiles = yaml.safe_load(f)
    profile = profiles[list(profiles.keys())[0]]
    return profile["outputs"][profile["target"]]


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

    daily = (
        df.groupby(["symbol", "date"])
        .size()
        .reset_index(name="bars")
    )

    full_days = daily[daily["bars"] == EXPECTED_BARS_PER_DAY][["symbol", "date"]]

    removed = daily[daily["bars"] != EXPECTED_BARS_PER_DAY].copy()
    removed.to_csv(OUTPUT_REMOVED_DAYS, index=False)

    df = df.merge(full_days, on=["symbol", "date"], how="inner")
    df = df.drop(columns=["date"])

    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_from_db() -> pd.DataFrame:
    conn = create_connection()

    query = f"""
        SELECT symbol, datetime, close
        FROM {TABLE_NAME}
        WHERE symbol = ANY(%s)
          AND datetime >= TIMESTAMP WITH TIME ZONE %s
          AND datetime <= TIMESTAMP WITH TIME ZONE %s
        ORDER BY symbol, datetime;
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
        ["symbol", "datetime"]
    ).reset_index(drop=True)


def load_from_csv() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(NY_TZ)

    df = df[df["symbol"].isin(SYMBOLS)].copy()

    start = pd.Timestamp(SNAPSHOT_START_NY, tz=NY_TZ)
    end   = pd.Timestamp(SNAPSHOT_END_NY,   tz=NY_TZ)

    df = df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()

    return df[["symbol", "datetime", "close"]].sort_values(
        ["symbol", "datetime"]
    ).reset_index(drop=True)


def load_all_data() -> pd.DataFrame:
    print(f"[DATA] source={DATA_SOURCE.upper()}", end="  ")

    if DATA_SOURCE.lower() == "db":
        df = load_from_db()
    elif DATA_SOURCE.lower() == "csv":
        df = load_from_csv()
    else:
        raise ValueError("DATA_SOURCE must be 'db' or 'csv'.")

    if REQUIRE_FULL_RTH_DAYS:
        df = _filter_full_days(df)

    print(f"rows={len(df):,} | {df['datetime'].min()} → {df['datetime'].max()}")

    return df.reset_index(drop=True)


# =============================================================================
# TARGET AND SPLIT
# =============================================================================

def make_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    delta_target[t] = close[t + horizon] - close[t]
    """
    d = df.copy()

    d["delta_target"] = d["close"].shift(-horizon) - d["close"]

    return d.dropna(subset=["delta_target"]).reset_index(drop=True)


def split_warmup_val_online(df: pd.DataFrame):
    n = len(df)

    warmup_end = int(n * WARMUP_RATIO)
    val_end    = int(n * (WARMUP_RATIO + VAL_RATIO))

    warmup = df.iloc[:warmup_end].copy()
    val    = df.iloc[warmup_end:val_end].copy()
    online = df.iloc[val_end:].copy()

    return warmup, val, online


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    actual_delta: np.ndarray,
    pred_delta: np.ndarray,
    close_prices: np.ndarray,
) -> dict:
    actual_delta = np.asarray(actual_delta, dtype=np.float64)
    pred_delta   = np.asarray(pred_delta, dtype=np.float64)

    mae  = float(mean_absolute_error(actual_delta, pred_delta))
    rmse = float(math.sqrt(mean_squared_error(actual_delta, pred_delta)))

    delta_relative_mae_pct = float(
        mae / (np.mean(np.abs(actual_delta)) + 1e-8) * 100.0
    )

    implied_delta_relative_mae_pct = float(
        mae / (np.mean(np.abs(close_prices)) + 1e-8) * 100.0
    )

    return {
        "delta_mae": mae,
        "delta_rmse": rmse,
        "delta_relative_mae_pct": delta_relative_mae_pct,
        "implied_delta_relative_mae_pct": implied_delta_relative_mae_pct,
    }


# =============================================================================
# BASELINE
# =============================================================================

def run_m1_delta(online_df: pd.DataFrame, lookback: int) -> tuple[dict, pd.DataFrame]:
    """
    M1-D: naive_zero_delta
    pred_delta[t] = 0

    By construction, delta_relative_mae_pct equals 100% for this
    zero-delta baseline because MAE = mean(|actual_delta|).
    """
    eval_df = online_df.iloc[lookback:].copy().reset_index(drop=True)

    eval_df["pred_delta"] = 0.0
    eval_df["actual_delta"] = eval_df["delta_target"]

    eval_df["implied_pred_price"] = eval_df["close"] + eval_df["pred_delta"]
    eval_df["implied_actual_price"] = eval_df["close"] + eval_df["actual_delta"]

    eval_df["delta_error"] = eval_df["actual_delta"] - eval_df["pred_delta"]
    eval_df["delta_abs_error"] = eval_df["delta_error"].abs()

    metrics = compute_metrics(
        actual_delta=eval_df["actual_delta"].values,
        pred_delta=eval_df["pred_delta"].values,
        close_prices=eval_df["close"].values,
    )

    return metrics, eval_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_raw = load_all_data()

    print("\n" + "=" * 100)
    print("M1-D — NAIVE ZERO DELTA | next_delta")
    print("=" * 100)
    print(f"symbols      = {SYMBOLS}")
    print(f"horizons     = {TARGET_HORIZONS_BARS}")
    print(f"snapshot     = {SNAPSHOT_START_NY} → {SNAPSHOT_END_NY}")
    print(f"split        = {WARMUP_RATIO}:{VAL_RATIO}:{ONLINE_RATIO}")
    print(f"lookback     = {LOOKBACK}")
    print(f"full_days    = {REQUIRE_FULL_RTH_DAYS} ({EXPECTED_BARS_PER_DAY} bars/day)")
    print(f"output       = {OUTPUT_SUMMARY}")

    summary_rows = []
    prediction_rows = []

    for symbol in SYMBOLS:
        df_sym = (
            df_raw[df_raw["symbol"] == symbol]
            .copy()
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        if len(df_sym) < 5000:
            print(f"\n[SKIP] {symbol}: insufficient rows ({len(df_sym):,}).")
            continue

        for horizon in TARGET_HORIZONS_BARS:
            print("\n" + "-" * 100)
            print(f"symbol={symbol} | horizon={horizon}")

            d = make_target(df_sym, horizon)
            warmup, val, online = split_warmup_val_online(d)

            print(
                f"rows={len(d):,} | warmup={len(warmup):,} | "
                f"val={len(val):,} | online={len(online):,} | "
                f"eval_rows={len(online) - LOOKBACK:,}"
            )

            metrics, pred_df = run_m1_delta(online, LOOKBACK)

            print(
                f"delta_mae={metrics['delta_mae']:.6f} | "
                f"delta_rmse={metrics['delta_rmse']:.6f} | "
                f"delta_rel_mae={metrics['delta_relative_mae_pct']:.4f}% | "
                f"implied_rel_mae={metrics['implied_delta_relative_mae_pct']:.4f}%"
            )

            summary_rows.append({
                "symbol": symbol,
                "horizon": horizon,
                "model": "naive_zero_delta",
                "delta_mae": metrics["delta_mae"],
                "delta_rmse": metrics["delta_rmse"],
                "delta_relative_mae_pct": metrics["delta_relative_mae_pct"],
                "implied_delta_relative_mae_pct": metrics["implied_delta_relative_mae_pct"],
            })

            keep_cols = [
                "symbol", "datetime", "close",
                "delta_target",
                "pred_delta", "actual_delta",
                "implied_pred_price", "implied_actual_price",
                "delta_error", "delta_abs_error",
            ]

            out_pred = pred_df[keep_cols].copy()
            out_pred["horizon"] = horizon
            out_pred["model"] = "naive_zero_delta"

            prediction_rows.append(out_pred)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    if prediction_rows:
        predictions = pd.concat(prediction_rows, ignore_index=True)
        predictions.to_csv(OUTPUT_PREDICTIONS, index=False)

    print("\n" + "=" * 100)
    print("DONE")
    print(f"Saved summary:     {OUTPUT_SUMMARY}")
    print(f"Saved predictions: {OUTPUT_PREDICTIONS}")

    if len(summary) > 0:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()