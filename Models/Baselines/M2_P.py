#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARIMA Baseline  |  next_close (price level) + direction targets
===============================================================
- Target 1 (price):     close[t + horizon]
- Target 2 (direction): 1 if close[t+horizon] > close[t] else 0

ARIMA order is determined ONCE on warmup data using auto_arima.
The same order is then used for the entire walk-forward evaluation.

Resume capability:
    If the output CSV already exists, completed (symbol, horizon) pairs
    are skipped automatically. Results are appended row by row.

Evaluation starts at row LOOKBACK=60 to match IL-ETransformer.
"""

import os
import math
import yaml
import numpy as np
import pandas as pd
import psycopg2
import warnings
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
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

NY_TZ = "America/New_York"

SNAPSHOT_START_NY = "2025-01-02 09:30:00"
SNAPSHOT_END_NY   = "2026-02-13 15:59:00"

REQUIRE_FULL_RTH_DAYS = True
EXPECTED_BARS_PER_DAY = 390

TARGET_HORIZONS_BARS = [1, 30, 60, 240, 390]

WARMUP_RATIO = 0.20
VAL_RATIO    = 0.10
ONLINE_RATIO = 0.70

# Must match IL-ETransformer — first LOOKBACK steps of online phase are skipped
LOOKBACK = 60

# Refit ARIMA every N steps during online phase (same order, new data window)
REFIT_INTERVAL = 500

DATA_SOURCE = "csv"   # "csv" or "db"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "canonical_1m_rth_FULL.csv"

OUTPUT_SUMMARY      = RESULTS_DIR /"baseline_arima_price_direction_summary.csv"
OUTPUT_PREDICTIONS  = RESULTS_DIR /"baseline_arima_price_direction_predictions.csv"
OUTPUT_REMOVED_DAYS = RESULTS_DIR / "arima_price_direction_removed_days.csv"


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

    daily_counts = (
        df.groupby(["symbol", "date"])
        .size()
        .reset_index(name="bars")
    )

    full_days = daily_counts[
        daily_counts["bars"] == EXPECTED_BARS_PER_DAY
    ][["symbol", "date"]]

    removed = daily_counts[daily_counts["bars"] != EXPECTED_BARS_PER_DAY].copy()
    removed.to_csv(OUTPUT_REMOVED_DAYS, index=False)

    df = df.merge(full_days, on=["symbol", "date"], how="inner")
    df = df.drop(columns=["date"])
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
    return df[["symbol", "datetime", "close"]].sort_values(
        ["symbol", "datetime"]
    ).reset_index(drop=True)


def load_from_db() -> pd.DataFrame:
    conn  = create_connection()
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

    print(
        f"rows={len(df):,} | "
        f"{df['datetime'].min()} → {df['datetime'].max()}"
    )
    return df.reset_index(drop=True)


# =============================================================================
# TARGET AND SPLIT
# =============================================================================

def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    price_target[t]     = close[t + horizon]
    direction_target[t] = 1 if close[t+horizon] > close[t] else 0
    delta[t]            = close[t+horizon] - close[t]
    """
    d                     = df.copy()
    future_close          = d["close"].shift(-horizon)
    d["price_target"]     = future_close
    d["delta"]            = future_close - d["close"]
    d["direction_target"] = (d["delta"] > 0).astype(int)
    return d.dropna(subset=["price_target"]).reset_index(drop=True)


def split_warmup_val_online(df: pd.DataFrame):
    n          = len(df)
    warmup_end = int(n * WARMUP_RATIO)
    val_end    = int(n * (WARMUP_RATIO + VAL_RATIO))
    return (
        df.iloc[:warmup_end].copy(),
        df.iloc[warmup_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


# =============================================================================
# AUTO_ARIMA ORDER SELECTION
# =============================================================================

def select_order_auto_arima(warmup_close: np.ndarray) -> tuple:
    """
    Fit auto_arima on warmup close prices to determine best (p, d, q).
    The identified order is held fixed for the entire walk-forward phase.

    Citation: Hyndman & Khandakar (2008) — automatic ARIMA order selection
    via stepwise search minimising AIC.
    """
    print("  [auto_arima] fitting on warmup prices ...", end=" ", flush=True)
    am = auto_arima(
        warmup_close,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=5,
        max_q=5,
        max_d=2,
    )
    order = am.order
    print(f"best order = {order}")
    return order


# =============================================================================
# WALK-FORWARD ON PRICE LEVELS
# =============================================================================

def arima_walk_forward_price(
    warmup_close: np.ndarray,
    eval_df: pd.DataFrame,
    horizon: int,
    order: tuple,
    lookback: int,
) -> dict:
    """
    Walk-forward ARIMA on price level series.

    At each step t:
      - History = warmup_close + close values seen so far
      - Forecast 'horizon' steps ahead
      - Predicted price = forecast[horizon - 1]  (the h-th step)
      - Direction = 1 if predicted_price > close[t] else 0
      - Append actual close[t] to history
    """
    if len(warmup_close) < 100:
        raise ValueError("Warmup data is too short for ARIMA.")

    history    = list(warmup_close.astype(np.float64))
    close_vals = eval_df["close"].values
    price_tgt  = eval_df["price_target"].values
    dir_tgt    = eval_df["direction_target"].values

    pred_price_all = []
    pred_dir_all   = []

    try:
        model = ARIMA(history, order=order).fit()
    except Exception as e:
        raise RuntimeError(f"Initial ARIMA fit failed: {e}")

    n = len(eval_df)

    for t in range(n):
        if t % REFIT_INTERVAL == 0 and t != 0:
            try:
                model = ARIMA(history, order=order).fit()
            except Exception:
                pass

        try:
            forecast     = model.forecast(steps=horizon)
            pred_price_t = float(
                forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]
            )
        except Exception:
            pred_price_t = float(history[-1])

        pred_dir_t = 1 if pred_price_t > float(close_vals[t]) else 0

        pred_price_all.append(pred_price_t)
        pred_dir_all.append(pred_dir_t)

        history.append(float(close_vals[t]))

    pred_price_all = np.array(pred_price_all, dtype=np.float64)
    pred_dir_all   = np.array(pred_dir_all,   dtype=np.int32)

    return {
        "pred_price":       pred_price_all[lookback:],
        "actual_price":     price_tgt[lookback:],
        "pred_direction":   pred_dir_all[lookback:],
        "actual_direction": dir_tgt[lookback:],
        "close_prices":     close_vals[lookback:],
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(out: dict) -> dict:
    pred_price       = out["pred_price"]
    actual_price     = out["actual_price"]
    pred_direction   = out["pred_direction"]
    actual_direction = out["actual_direction"]

    mae  = float(mean_absolute_error(actual_price, pred_price))
    rmse = float(math.sqrt(mean_squared_error(actual_price, pred_price)))

    price_relative_mae_pct = float(
        mae / (np.mean(np.abs(actual_price)) + 1e-8) * 100.0
    )

    acc       = float(accuracy_score(actual_direction, pred_direction))
    f1        = float(f1_score(actual_direction, pred_direction, zero_division=0))
    precision = float(precision_score(actual_direction, pred_direction, zero_division=0))
    recall    = float(recall_score(actual_direction, pred_direction, zero_division=0))

    return {
        "price_mae":                  mae,
        "price_rmse":                 rmse,
        "price_relative_mae_pct":     price_relative_mae_pct,
        "direction_accuracy":         acc,
        "direction_f1":               f1,
        "direction_precision":        precision,
        "direction_recall":           recall,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_raw = load_all_data()

    completed = get_completed_runs(OUTPUT_SUMMARY)
    if completed:
        print(f"[RESUME] {len(completed)} pair(s) already done: {completed}")

    print("\n" + "=" * 100)
    print("ARIMA BASELINE  |  next_close (price) + direction  |  auto_arima order selection")
    print("=" * 100)
    print(f"symbols    = {SYMBOLS}")
    print(f"horizons   = {TARGET_HORIZONS_BARS}")
    print(f"snapshot   = {SNAPSHOT_START_NY} → {SNAPSHOT_END_NY}")
    print(f"split      = {WARMUP_RATIO}:{VAL_RATIO}:{ONLINE_RATIO}")
    print(f"lookback   = {LOOKBACK}  (first {LOOKBACK} online steps skipped)")
    print(f"refit_int  = {REFIT_INTERVAL}")
    print(f"output     = {OUTPUT_SUMMARY}")

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

        # Determine order once per symbol on warmup close prices
        warmup_for_order, _, _ = split_warmup_val_online(
            make_targets(df_sym, horizon=1)
        )
        try:
            order = select_order_auto_arima(warmup_for_order["close"].values)
        except Exception as e:
            print(f"[ERROR] auto_arima failed for {symbol}: {e}")
            continue

        for horizon in TARGET_HORIZONS_BARS:

            if (symbol, horizon) in completed:
                print(f"\n[SKIP] {symbol} h={horizon} — already in output.")
                continue

            print("\n" + "-" * 100)
            print(f"symbol={symbol} | horizon={horizon} | order={order}")

            d = make_targets(df_sym, horizon)
            warmup, val, online = split_warmup_val_online(d)

            print(
                f"rows={len(d):,} | warmup={len(warmup):,} | "
                f"val={len(val):,} | online={len(online):,} | "
                f"eval_rows={len(online) - LOOKBACK:,}"
            )

            try:
                out = arima_walk_forward_price(
                    warmup_close=warmup["close"].values,
                    eval_df=online,
                    horizon=horizon,
                    order=order,
                    lookback=LOOKBACK,
                )
            except Exception as e:
                print(f"[ERROR] walk-forward failed: {e}")
                continue

            m = compute_metrics(out)

            model_label = f"arima_{order[0]}_{order[1]}_{order[2]}"

            print(
                f"price_mae={m['price_mae']:.6f} | "
                f"price_rmse={m['price_rmse']:.6f} | "
                f"price_rel_mae={m['price_relative_mae_pct']:.4f}% | "
                f"acc={m['direction_accuracy']:.4f} | "
                f"f1={m['direction_f1']:.4f} | "
                f"precision={m['direction_precision']:.4f} | "
                f"recall={m['direction_recall']:.4f}"
            )

            result_row = {
                "symbol":                  symbol,
                "horizon":                 horizon,
                "model":                   model_label,
                "arima_order":             str(order),
                "warmup_rows":             len(warmup),
                "val_rows":                len(val),
                "online_rows":             len(online),
                "eval_rows":               len(out["pred_price"]),
                "price_mae":               m["price_mae"],
                "price_rmse":              m["price_rmse"],
                "price_relative_mae_pct":  m["price_relative_mae_pct"],
                "direction_accuracy":      m["direction_accuracy"],
                "direction_f1":            m["direction_f1"],
                "direction_precision":     m["direction_precision"],
                "direction_recall":        m["direction_recall"],
            }
            append_row(OUTPUT_SUMMARY, result_row)
            print(f"  => saved to {OUTPUT_SUMMARY}")

            pred_df = online.iloc[LOOKBACK:].reset_index(drop=True)[
                ["symbol", "datetime", "close", "price_target", "direction_target"]
            ].copy()
            pred_df["horizon"]           = horizon
            pred_df["model"]             = model_label
            pred_df["pred_price"]        = out["pred_price"]
            pred_df["actual_price"]      = out["actual_price"]
            pred_df["pred_direction"]    = out["pred_direction"]
            pred_df["actual_direction"]  = out["actual_direction"]
            pred_df["price_error"]       = pred_df["actual_price"] - pred_df["pred_price"]
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
        cols = [
            "symbol", "horizon", "model",
            "price_mae", "price_rmse", "price_relative_mae_pct",
            "direction_accuracy", "direction_f1",
            "direction_precision", "direction_recall",
            "eval_rows",
        ]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()