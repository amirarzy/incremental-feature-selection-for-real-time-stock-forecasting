#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
M1-P — Naive Baseline | next_close target + derived direction
=============================================================
Baseline:
  M1-P: naive_last_close
        pred_price[t] = close[t]

Direction:
  Direction is evaluated only for the price variant.
  It is derived from the predicted price:
        pred_direction[t] = 1[pred_price[t] > close[t]]

  For naive_last_close, pred_price[t] = close[t], so:
        pred_direction[t] = 0
  This means the naive price baseline always predicts no upward move.

Evaluation:
  Online phase only.
  First LOOKBACK=60 rows skipped to match IL-ETransformer.

Outputs:
  baseline_m1_price_summary.csv
  baseline_m1_price_predictions.csv
"""

import os
import math
import yaml
import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


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

OUTPUT_SUMMARY      = RESULTS_DIR / "baseline_m1_price_summary.csv"
OUTPUT_PREDICTIONS  = RESULTS_DIR / "baseline_m1_price_predictions.csv"
OUTPUT_REMOVED_DAYS = RESULTS_DIR / "m1_price_removed_days.csv"


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
    price_target[t] = close[t + horizon]
    direction_target[t] = 1 if close[t+horizon] > close[t], else 0
    """
    d = df.copy()

    future = d["close"].shift(-horizon)

    d["price_target"] = future
    d["direction_target"] = (future > d["close"]).astype(int)

    return d.dropna(subset=["price_target"]).reset_index(drop=True)


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
    actual_price: np.ndarray,
    pred_price: np.ndarray,
    close_prices: np.ndarray,
    actual_direction: np.ndarray,
    pred_direction: np.ndarray,
) -> dict:
    actual_price = np.asarray(actual_price, dtype=np.float64)
    pred_price   = np.asarray(pred_price, dtype=np.float64)

    mae  = float(mean_absolute_error(actual_price, pred_price))
    rmse = float(math.sqrt(mean_squared_error(actual_price, pred_price)))

    price_relative_mae_pct = float(
        mae / (np.mean(np.abs(actual_price)) + 1e-8) * 100.0
    )

    implied_price_relative_mae_pct = float(
        mae / (np.mean(np.abs(close_prices)) + 1e-8) * 100.0
    )

    acc  = float(accuracy_score(actual_direction, pred_direction))
    f1   = float(f1_score(actual_direction, pred_direction, zero_division=0))
    prec = float(precision_score(actual_direction, pred_direction, zero_division=0))
    rec  = float(recall_score(actual_direction, pred_direction, zero_division=0))

    return {
        "price_mae": mae,
        "price_rmse": rmse,
        "price_relative_mae_pct": price_relative_mae_pct,
        "implied_price_relative_mae_pct": implied_price_relative_mae_pct,
        "direction_accuracy": acc,
        "direction_f1": f1,
        "direction_precision": prec,
        "direction_recall": rec,
    }


# =============================================================================
# BASELINE
# =============================================================================

def run_m1_price(online_df: pd.DataFrame, lookback: int) -> tuple[dict, pd.DataFrame]:
    """
    M1-P: naive_last_close
    pred_price[t] = close[t]
    """
    eval_df = online_df.iloc[lookback:].copy().reset_index(drop=True)

    eval_df["pred_price"] = eval_df["close"]
    eval_df["actual_price"] = eval_df["price_target"]

    eval_df["pred_direction"] = (
        eval_df["pred_price"] > eval_df["close"]
    ).astype(int)

    eval_df["actual_direction"] = eval_df["direction_target"].astype(int)

    eval_df["price_error"] = eval_df["actual_price"] - eval_df["pred_price"]
    eval_df["price_abs_error"] = eval_df["price_error"].abs()
    eval_df["direction_correct"] = (
        eval_df["pred_direction"] == eval_df["actual_direction"]
    ).astype(int)

    metrics = compute_metrics(
        actual_price=eval_df["actual_price"].values,
        pred_price=eval_df["pred_price"].values,
        close_prices=eval_df["close"].values,
        actual_direction=eval_df["actual_direction"].values,
        pred_direction=eval_df["pred_direction"].values,
    )

    return metrics, eval_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_raw = load_all_data()

    print("\n" + "=" * 100)
    print("M1-P — NAIVE LAST CLOSE | next_close + derived direction")
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

            metrics, pred_df = run_m1_price(online, LOOKBACK)

            print(
                f"price_mae={metrics['price_mae']:.6f} | "
                f"price_rmse={metrics['price_rmse']:.6f} | "
                f"price_rel_mae={metrics['price_relative_mae_pct']:.4f}% | "
                f"dir_acc={metrics['direction_accuracy']:.4f} | "
                f"dir_f1={metrics['direction_f1']:.4f}"
            )

            summary_rows.append({
                "symbol": symbol,
                "horizon": horizon,
                "model": "naive_last_close",
                "price_mae": metrics["price_mae"],
                "price_rmse": metrics["price_rmse"],
                "price_relative_mae_pct": metrics["price_relative_mae_pct"],
                "implied_price_relative_mae_pct": metrics["implied_price_relative_mae_pct"],
                "direction_accuracy": metrics["direction_accuracy"],
                "direction_f1": metrics["direction_f1"],
                "direction_precision": metrics["direction_precision"],
                "direction_recall": metrics["direction_recall"],
            })

            keep_cols = [
                "symbol", "datetime", "close",
                "price_target", "direction_target",
                "pred_price", "actual_price",
                "pred_direction", "actual_direction",
                "price_error", "price_abs_error",
                "direction_correct",
            ]

            out_pred = pred_df[keep_cols].copy()
            out_pred["horizon"] = horizon
            out_pred["model"] = "naive_last_close"

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