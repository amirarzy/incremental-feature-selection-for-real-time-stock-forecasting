#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARIMA Baseline  |  next_delta target
=====================================
- Target: close[t + horizon] - close[t]

ARIMA is fitted on the DIFFERENCE series (delta series), so d is forced to 0.
The optimal (p, 0, q) order is determined ONCE per symbol on warmup deltas
using auto_arima, then held fixed for the entire walk-forward evaluation.

Resume capability:
    If the output CSV already exists, completed (symbol, horizon) pairs
    are skipped automatically. Results are appended row by row.

Evaluation starts at row LOOKBACK=60 to match IL-ETransformer.

Output columns:
    symbol, horizon, model,
    delta_mae, delta_rmse,
    delta_relative_mae_pct,
    implied_delta_relative_mae_pct
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

OUTPUT_SUMMARY      = RESULTS_DIR / "baseline_arima_delta_summary.csv"
OUTPUT_PREDICTIONS  = RESULTS_DIR / "baseline_arima_delta_predictions.csv"
OUTPUT_REMOVED_DAYS = RESULTS_DIR / "arima_delta_removed_days.csv"


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

def make_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    target[t] = close[t + horizon] - close[t]
    """
    d           = df.copy()
    d["target"] = d["close"].shift(-horizon) - d["close"]
    return d.dropna(subset=["target"]).reset_index(drop=True)


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
    Fit auto_arima on the DELTA (diff) series of warmup close prices.
    d is fixed to 0 because we are already working on the differenced series.

    The identified (p, 0, q) order is held fixed for the entire walk-forward
    phase and reused across all horizons for this symbol.

    Citation: Hyndman & Khandakar (2008) — automatic ARIMA order selection
    via stepwise search minimising AIC.
    """
    warmup_delta = np.diff(warmup_close).astype(np.float64)

    print("  [auto_arima] fitting on warmup delta series ...", end=" ", flush=True)
    am = auto_arima(
        warmup_delta,
        d=0,
        D=0,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=5,
        max_q=5,
    )
    order = am.order
    print(f"best order = {order}")
    return order


# =============================================================================
# WALK-FORWARD ON DELTA SERIES
# =============================================================================

def arima_walk_forward_delta(
    warmup_close: np.ndarray,
    eval_df: pd.DataFrame,
    horizon: int,
    order: tuple,
    lookback: int,
) -> dict:
    """
    Walk-forward ARIMA on the delta (difference) series.

    At each step t:
      - history_delta = diff(warmup_close) + observed deltas so far
      - ARIMA forecasts 'horizon' future 1-bar deltas
      - predicted cumulative delta = sum of forecasted deltas
      - actual cumulative delta    = close[t + horizon] - close[t]
      - append current bar's delta to history
    """
    if len(warmup_close) < 100:
        raise ValueError("Warmup data is too short for ARIMA.")

    history_delta = list(np.diff(warmup_close).astype(np.float64))

    close_vals  = eval_df["close"].values
    target_vals = eval_df["target"].values
    prev_close  = float(warmup_close[-1])

    pred_delta_all = []

    try:
        model = ARIMA(history_delta, order=order).fit()
    except Exception as e:
        raise RuntimeError(f"Initial ARIMA fit failed: {e}")

    n = len(eval_df)

    for t in range(n):
        if t % REFIT_INTERVAL == 0 and t != 0:
            try:
                model = ARIMA(history_delta, order=order).fit()
            except Exception:
                pass

        try:
            forecasted_deltas = model.forecast(steps=horizon)
            if hasattr(forecasted_deltas, "values"):
                pred_delta_t = float(np.sum(forecasted_deltas.values))
            else:
                pred_delta_t = float(np.sum(forecasted_deltas))
        except Exception:
            pred_delta_t = 0.0

        pred_delta_all.append(pred_delta_t)

        current_close = float(close_vals[t])
        history_delta.append(current_close - prev_close)
        prev_close = current_close

    pred_delta_all = np.array(pred_delta_all, dtype=np.float64)

    return {
        "pred":         pred_delta_all[lookback:],
        "actual":       target_vals[lookback:],
        "close_prices": close_vals[lookback:],
    }


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    close_prices: np.ndarray,
) -> dict:
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
    df_raw = load_all_data()

    completed = get_completed_runs(OUTPUT_SUMMARY)
    if completed:
        print(f"[RESUME] {len(completed)} pair(s) already done: {completed}")

    print("\n" + "=" * 100)
    print("ARIMA BASELINE  |  next_delta  |  auto_arima order selection on delta series")
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

        # Determine order once per symbol — reused across all horizons
        warmup_for_order, _, _ = split_warmup_val_online(
            make_target(df_sym, horizon=1)
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

            d = make_target(df_sym, horizon)
            warmup, val, online = split_warmup_val_online(d)

            print(
                f"rows={len(d):,} | warmup={len(warmup):,} | "
                f"val={len(val):,} | online={len(online):,} | "
                f"eval_rows={len(online) - LOOKBACK:,}"
            )

            try:
                out = arima_walk_forward_delta(
                    warmup_close=warmup["close"].values,
                    eval_df=online,
                    horizon=horizon,
                    order=order,
                    lookback=LOOKBACK,
                )
            except Exception as e:
                print(f"[ERROR] walk-forward failed: {e}")
                continue

            m = compute_metrics(
                actual=out["actual"],
                pred=out["pred"],
                close_prices=out["close_prices"],
            )

            model_label = f"arima_{order[0]}_{order[1]}_{order[2]}"

            print(
                f"delta_mae={m['delta_mae']:.6f} | "
                f"delta_rmse={m['delta_rmse']:.6f} | "
                f"delta_rel_mae={m['delta_relative_mae_pct']:.4f}% | "
                f"implied_rel_mae={m['implied_delta_relative_mae_pct']:.4f}%"
            )

            # Save summary row immediately — only required columns
            result_row = {
                "symbol":                          symbol,
                "horizon":                         horizon,
                "model":                           model_label,
                "delta_mae":                       m["delta_mae"],
                "delta_rmse":                      m["delta_rmse"],
                "delta_relative_mae_pct":          m["delta_relative_mae_pct"],
                "implied_delta_relative_mae_pct":  m["implied_delta_relative_mae_pct"],
            }
            append_row(OUTPUT_SUMMARY, result_row)
            print(f"  => saved to {OUTPUT_SUMMARY}")

            # Save prediction-level rows immediately
            pred_df = online.iloc[LOOKBACK:].reset_index(drop=True)[
                ["symbol", "datetime", "close", "target"]
            ].copy()
            pred_df["horizon"]      = horizon
            pred_df["model"]        = model_label
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
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()