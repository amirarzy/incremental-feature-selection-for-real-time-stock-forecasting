#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model #7b — IL-ETransformer + IFS  |  6 features  |  delta + direction
========================================================================
Qian (2025) replication WITH Incremental Feature Selection (IFS) gate.
TARGET VARIANT: delta = close[t+h] − close[t]  (price change, not level)

Identical to Model #6 (IL-ETransformer no_ifs) in all architectural
details except for one addition:

  IFS Gate (added before ConvTokenEmbedding):
    g ∈ R^{N_FEATURES}  — learnable parameter, sigmoid-activated
    X_gated = X * sigmoid(g)         applied at each forward/online step
    Gates are updated jointly with all model parameters during TS-EWC steps.
    Gate values recorded every EWC_UPDATE_FREQ bars → gate_history.csv (RQ2).

Features  : open, high, low, close, change_ratio, macd
Targets   : delta     → close[t + h] − close[t]   (price change)
            direction → sign(delta)  →  1 (up) / 0 (down)

Architecture (Qian, 2025):
  Embedding:
    IFSGate                  — learnable sigmoid gate per feature (MODEL #7 only)
    CausalConvTokenEmbedding — Conv1D kernel=3, left-pad=2 (causal, no future leakage)
    PositionalEncoding       — sinusoidal
    TimeEmbedding            — minute_of_day / day_of_week / month_of_year → learnable
    X_emb = data_emb + pos_emb + time_emb

  Encoder: N enc_layers × (ConMHA → ContinualNorm → FFN → ContinualNorm)
  Decoder: M=1 layer    × (Masked MHA → CN → Cross MHA → CN → FFN → CN) → Linear
           X_de = Concat(X_token, X_0)
           X_token = last encoder embedding token
           X_0     = zero vector (prediction slot)

  ConMHA (Eq. 9-13):
    Train  : standard full-sequence MHA — O(L²·d_k) per forward
    Online : FIFO cache for K and V (size = seq_len)
             Each step: oldest K/V evicted, new K/V appended.
             Attention computed only for current query → O(L·d_k) per step.

  ContinualNorm (Eq. 20-21):
    GroupNorm → BatchNorm with EMA momentum η=EWC_ETA.
    Eval mode: BN uses running stats — safe at batch_size=1.

  TS-EWC (Eq. 22):
    Novelty buffer  : update if MSE > dynamic threshold (1.5× rolling mean)
    Familiarity buf : circular buffer for Fisher estimation
    Fisher matrix   : diagonal, from FISHER_N_SAMPLES familiarity samples
    Loss = L_task + λ Σ F_i(θ_i − θ*_i)²
    After each EWC step: FIFO caches refreshed (parameters changed → stale K/V)

  IFS Gate tracking (RQ2):
    Every EWC_UPDATE_FREQ bars, sigmoid(g) values saved:
    gate_history_{symbol}_h{horizon}.csv
    Columns: bar_index, datetime, symbol, horizon,
             gate_open, gate_high, gate_low, gate_close,
             gate_change_ratio, gate_macd

Training  : Warmup + Val → Adam, MSELoss, batch from HP_SPACE
            Online       → TS-EWC loss every EWC_UPDATE_FREQ=25 bars (novelty-gated)

HP Search : random search, N_TRIALS=15, RANDOM_SEED=42 — Bergstra & Bengio (2012)
            Same seed, same budget as all other baselines in this study.

Evaluation: Online phase (70%) walk-forward, no lookahead.
            First LOOKBACK=60 steps skipped — matches all other baselines.
"""

import os
import math
import random
import warnings
import yaml
from collections import deque

import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
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

TABLE_NAME        = "staging.canonical_1m_rth"
SYMBOLS           = ["AAPL", "MSFT", "NVDA", "TSLA"]
NY_TZ             = "America/New_York"
SNAPSHOT_START_NY = "2025-01-02 09:30:00"
SNAPSHOT_END_NY   = "2026-02-13 15:59:00"

REQUIRE_FULL_RTH_DAYS = True
EXPECTED_BARS_PER_DAY = 390

TARGET_HORIZONS_BARS = [1, 30, 60, 240, 390]

WARMUP_RATIO = 0.20
VAL_RATIO    = 0.10
ONLINE_RATIO = 0.70

LOOKBACK     = 60    # First N online steps skipped — matches all baselines
SCALE_WINDOW = 390   # Dynamic scaler window (one trading session)

# TS-EWC
EWC_UPDATE_FREQ  = 25    # Gradient update every N bars (novelty-gated)
EWC_ETA          = 0.1   # ContinualNorm BN EMA rate η (Eq. 20-21)
FISHER_N_SAMPLES = 50    # Samples used for Fisher matrix estimation
FAM_BUF_SIZE     = 500   # Familiarity buffer capacity

# Architecture
DEC_TOKEN_LEN = 1        # Length of X_token in decoder input
CONV_KERNEL   = 3        # ConvTokenEmbedding kernel size (Eq. 5)
CONV_PAD      = CONV_KERNEL - 1  # Left-pad for causality

# Features
FEATURE_COLS = ["open", "high", "low", "close", "change_ratio", "macd"]
TIME_COLS    = ["minute_of_day", "day_of_week", "month_of_year"]
N_FEATURES   = len(FEATURE_COLS)
CLOSE_IDX    = FEATURE_COLS.index("close")   # index 3

# Random search — Bergstra & Bengio (2012)
N_TRIALS    = 15
RANDOM_SEED = 42

DATA_SOURCE = "csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "canonical_1m_rth_FULL.csv"

OUTPUT_SUMMARY      = RESULTS_DIR / "baseline_iltransformer_ifs_delta_summary.csv"
OUTPUT_PREDICTIONS  = RESULTS_DIR / "baseline_iltransformer_ifs_delta_predictions.csv"
OUTPUT_REMOVED_DAYS = RESULTS_DIR / "iltransformer_ifs_removed_days.csv"

GATE_DIR = RESULTS_DIR / "Gate_D"
GATE_DIR.mkdir(parents=True, exist_ok=True)

# Gate history written per (symbol, horizon) — see append_gate_history()
GATE_HISTORY_PREFIX = "gate_history_delta"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# HYPERPARAMETER SPACE
# Same budget (N=15, seed=42) as all other baselines — Bergstra & Bengio (2012)
# =============================================================================

HP_SPACE = {
    "seq_len":       [16, 32, 60],
    "d_model":       [32, 64, 128],
    "nhead":         [2, 4, 8],
    "enc_layers":    [1, 2],
    "ff_dim":        [64, 128, 256],
    "dropout":       [0.0, 0.1, 0.2],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "online_lr":     [1e-5, 5e-5, 1e-4],
    "batch_size":    [64, 128],
    "epochs":        [10, 20],
    "ewc_lambda":    [0.1, 1.0, 10.0],
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
    df_row    = pd.DataFrame([row])
    write_hdr = not os.path.exists(path)
    df_row.to_csv(path, mode="a", header=write_hdr, index=False)


def append_predictions(path: str, pred_df: pd.DataFrame) -> None:
    write_hdr = not os.path.exists(path)
    pred_df.to_csv(path, mode="a", header=write_hdr, index=False)


def append_gate_history(symbol: str, horizon: int,
                        gate_df: pd.DataFrame) -> None:
    """Write gate history for one (symbol, horizon) pair."""
    path      = GATE_DIR / f"{GATE_HISTORY_PREFIX}_{symbol}_h{horizon}.csv"
    write_hdr = not os.path.exists(path)
    gate_df.to_csv(path, mode="a", header=write_hdr, index=False)


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
# DATA LOADING & FEATURE ENGINEERING
# =============================================================================

def _filter_full_days(df: pd.DataFrame) -> pd.DataFrame:
    df         = df.copy()
    df["date"] = df["datetime"].dt.date
    daily      = df.groupby(["symbol", "date"]).size().reset_index(name="bars")
    full       = daily[daily["bars"] == EXPECTED_BARS_PER_DAY][["symbol", "date"]]
    removed    = daily[daily["bars"] != EXPECTED_BARS_PER_DAY].copy()
    removed.to_csv(OUTPUT_REMOVED_DAYS, index=False)
    df = df.merge(full, on=["symbol", "date"], how="inner").drop(columns=["date"])
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    change_ratio = (close − open) / (|open| + 1e-8)   [causal, no leakage]
    macd         = EMA(close,12) − EMA(close,26)       [causal EWM]
    """
    df = df.copy()
    df["change_ratio"] = (df["close"] - df["open"]) / (df["open"].abs() + 1e-8)
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g         = g.copy()
        g["macd"] = (g["close"].ewm(span=12, adjust=False).mean()
                     - g["close"].ewm(span=26, adjust=False).mean())
        parts.append(g)
    return pd.concat(parts).sort_values(["symbol", "datetime"]).reset_index(drop=True)


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    minute_of_day : 0-389  (9:30 → 0, 15:59 → 389)
    day_of_week   : 0-4    (Mon=0, Fri=4)
    month_of_year : 0-11   (Jan=0, Dec=11)
    """
    df = df.copy()
    df["minute_of_day"] = (
        (df["datetime"].dt.hour - 9) * 60 + df["datetime"].dt.minute - 30
    ).clip(0, 389).astype(int)
    df["day_of_week"]   = df["datetime"].dt.dayofweek.clip(0, 4).astype(int)
    df["month_of_year"] = (df["datetime"].dt.month - 1).clip(0, 11).astype(int)
    return df


def load_from_db() -> pd.DataFrame:
    conn  = create_connection()
    query = f"""
        SELECT symbol, datetime, open, high, low, close
        FROM   {TABLE_NAME}
        WHERE  symbol = ANY(%s)
          AND  datetime >= TIMESTAMP WITH TIME ZONE %s
          AND  datetime <= TIMESTAMP WITH TIME ZONE %s
        ORDER BY symbol, datetime
    """
    df = pd.read_sql_query(
        query, conn,
        params=[SYMBOLS,
                f"{SNAPSHOT_START_NY} America/New_York",
                f"{SNAPSHOT_END_NY} America/New_York"]
    )
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(NY_TZ)
    return df.sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_from_csv() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(NY_TZ)
    df  = df[df["symbol"].isin(SYMBOLS)].copy()
    df  = df[(df["datetime"] >= pd.Timestamp(SNAPSHOT_START_NY, tz=NY_TZ)) &
             (df["datetime"] <= pd.Timestamp(SNAPSHOT_END_NY,   tz=NY_TZ))]
    return df[["symbol", "datetime", "open", "high", "low", "close"]]\
           .sort_values(["symbol", "datetime"]).reset_index(drop=True)


def load_all_data() -> pd.DataFrame:
    print(f"[DATA] source={DATA_SOURCE.upper()}", end="  ")
    df = load_from_db() if DATA_SOURCE == "db" else load_from_csv()
    if REQUIRE_FULL_RTH_DAYS:
        df = _filter_full_days(df)
    df = compute_features(df)
    df = compute_time_features(df)
    print(f"rows={len(df):,} | {df['datetime'].min()} → {df['datetime'].max()}")
    return df.reset_index(drop=True)


# =============================================================================
# TARGETS & SPLIT
# =============================================================================

def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    d                 = df.copy()
    future            = d["close"].shift(-horizon)
    d["delta_target"] = future - d["close"]          # price change only
    return d.dropna(subset=["delta_target"]).reset_index(drop=True)


def split_warmup_val_online(df: pd.DataFrame):
    n  = len(df)
    we = int(n * WARMUP_RATIO)
    ve = int(n * (WARMUP_RATIO + VAL_RATIO))
    return df.iloc[:we].copy(), df.iloc[we:ve].copy(), df.iloc[ve:].copy()


# =============================================================================
# FEATURE SCALING
# =============================================================================

def compute_static_stats(df: pd.DataFrame):
    """Per-feature mean/std from fixed DataFrame. Used for warmup training only."""
    mean = df[FEATURE_COLS].mean().values.astype(np.float32)
    std  = (df[FEATURE_COLS].std().values + 1e-8).astype(np.float32)
    return mean, std


def dynamic_stats(hist_arr: np.ndarray):
    """Per-feature rolling mean/std from last SCALE_WINDOW rows of history."""
    w    = hist_arr[-SCALE_WINDOW:] if len(hist_arr) >= SCALE_WINDOW else hist_arr
    mean = w.mean(axis=0).astype(np.float32)
    std  = (w.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


def norm_feat(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean) / std


def norm_target(val: float, mean: np.ndarray, std: np.ndarray) -> float:
    """
    Normalize delta target by close std — no mean shift.
    Delta is centered near zero; subtracting close mean would be incorrect.
    """
    return val / float(std[CLOSE_IDX])


def denorm_target(val: float, mean: np.ndarray, std: np.ndarray) -> float:
    return val * float(std[CLOSE_IDX])


# =============================================================================
# DATASET  (warmup training — static scaler, no data leakage)
# =============================================================================

class WarmupDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int,
                 mean: np.ndarray, std: np.ndarray):
        feat_n = norm_feat(df[FEATURE_COLS].values.astype(np.float32), mean, std)
        tgt_n  = np.array([norm_target(float(v), mean, std)
                           for v in df["delta_target"].values], dtype=np.float32)
        tf     = df[TIME_COLS].values.astype(np.int64)

        X, T, y = [], [], []
        for i in range(seq_len, len(feat_n)):
            X.append(feat_n[i - seq_len: i])
            T.append(tf[i - seq_len: i])
            y.append(tgt_n[i])

        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.T = torch.tensor(np.array(T), dtype=torch.long)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.T[i], self.y[i]


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class IFSGate(nn.Module):
    """
    Incremental Feature Selection gate — MODEL #7 addition.

    One learnable scalar per feature: g ∈ R^{N_FEATURES}.
    Applied as element-wise multiplication after sigmoid activation:
        X_gated[b, t, i] = X[b, t, i] * sigmoid(g[i])

    Gates are updated jointly with all model parameters during TS-EWC steps.
    They are NOT frozen at any point — the online gradient flow naturally
    up- or down-weights features as their relevance evolves.

    Initialization: g = 0 → sigmoid(0) = 0.5 (all features equally weighted).
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, F]  →  [B, L, F]"""
        return x * torch.sigmoid(self.gate_logits)   # broadcast over B and L

    def gate_values(self) -> np.ndarray:
        """Return current sigmoid(g) as a numpy array for logging."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_logits).cpu().numpy()


class CausalConvTokenEmbedding(nn.Module):
    """
    Eq. 5 — Conv1D token embedding with explicit causality.
    kernel=3, left-pad=CONV_PAD=2 → output at position i depends only on
    inputs {i-2, i-1, i}. No right-pad → no future leakage.

    Online mode: maintains a FIFO context buffer of last CONV_PAD raw feature
    vectors. Each step: buffer + new token → conv → single output token.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model
        self.conv       = nn.Conv1d(n_features, d_model, kernel_size=CONV_KERNEL, padding=0)
        self._buf: torch.Tensor = None   # [B, CONV_PAD, n_features]

    # ── Training / full-sequence ──────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] → left-pad(CONV_PAD) → conv → [B, L, d_model]
        x_pad = F.pad(x.transpose(1, 2), (CONV_PAD, 0))
        return self.conv(x_pad).transpose(1, 2)

    # ── Online helpers ────────────────────────────────────────────────────────
    def init_buf(self, x_warmup: torch.Tensor) -> None:
        """Seed context buffer with last CONV_PAD tokens of warmup sequence."""
        self._buf = x_warmup[:, -CONV_PAD:, :].detach().clone()

    def forward_online(self, x_new: torch.Tensor) -> torch.Tensor:
        """
        x_new: [B, 1, F]
        Returns: [B, 1, d_model]
        FIFO: drop oldest context token, append x_new, run conv.
        """
        if self._buf is None:
            self._buf = torch.zeros(
                x_new.size(0), CONV_PAD, self.n_features, device=x_new.device
            )
        ctx = torch.cat([self._buf, x_new], dim=1)           # [B, CONV_PAD+1, F]
        out = self.conv(ctx.transpose(1, 2)).transpose(1, 2) # [B, 1, d_model]
        self._buf = ctx[:, 1:, :].detach().clone()           # keep last CONV_PAD tokens
        return out

    def reset_buf(self) -> None:
        self._buf = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full sequence: PE for positions 0..L-1."""
        return self.pe[:, :x.size(1), :]

    def at(self, pos: int, device: torch.device) -> torch.Tensor:
        """Single-position PE for online mode."""
        return self.pe[:, pos: pos + 1, :].to(device)


class TimeEmbedding(nn.Module):
    """
    Learnable time embeddings (Qian 2025).
    Three separate embedding tables summed:
      minute_of_day : 390 entries
      day_of_week   :   5 entries
      month_of_year :  12 entries
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.min_emb   = nn.Embedding(390, d_model)
        self.dow_emb   = nn.Embedding(5,   d_model)
        self.month_emb = nn.Embedding(12,  d_model)

    def forward(self, tf: torch.Tensor) -> torch.Tensor:
        # tf: [B, L, 3] → [B, L, d_model]
        return (self.min_emb(tf[:, :, 0])
                + self.dow_emb(tf[:, :, 1])
                + self.month_emb(tf[:, :, 2]))


class ContinualNorm(nn.Module):
    """
    Eq. 20-21 — ContinualNorm: GroupNorm → BatchNorm.
    BN momentum = η = EWC_ETA  →  running stats via EMA during training.
    Eval mode (online): BN uses running_mean / running_var → safe at B=1.
    """

    def __init__(self, d_model: int):
        super().__init__()
        num_groups = next((g for g in [8, 4, 2, 1] if d_model % g == 0), 1)
        self.gn    = nn.GroupNorm(num_groups, d_model)
        self.bn    = nn.BatchNorm1d(d_model, momentum=EWC_ETA, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        gn_out  = self.gn(x.transpose(1, 2)).transpose(1, 2)   # [B, L, D]
        bn_out  = self.bn(gn_out.reshape(B * L, D)).reshape(B, L, D)
        return bn_out


class ConMHA(nn.Module):
    """
    Continual Multi-Head Attention — Eq. 9-13, Qian (2025).

    Train mode:
        Standard full-sequence attention. O(L²·d_k) per forward.
        Input/output: [B, L, d_model].

    Online mode (forward_online):
        FIFO cache for K and V, size = seq_len.
        Step t:
          1. k_new = W_k(x_new),  v_new = W_v(x_new)
          2. FIFO: evict cache[:, :, 0, :], append k_new / v_new at tail
          3. q_new = W_q(x_new)
          4. scores = q_new @ cache_k.T / sqrt(d_k)  → [B, nhead, 1, seq_len]
          5. attn   = softmax(scores)
          6. out    = attn @ cache_v                  → [B, nhead, 1, d_k]
        O(L·d_k) per step — exact Eq. 9-13 implementation.
    """

    def __init__(self, d_model: int, nhead: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead   = nhead
        self.d_k     = d_model // nhead
        self.seq_len = seq_len

        self.W_q     = nn.Linear(d_model, d_model)
        self.W_k     = nn.Linear(d_model, d_model)
        self.W_v     = nn.Linear(d_model, d_model)
        self.W_o     = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # FIFO caches — runtime state, not model parameters
        self._cache_k: torch.Tensor = None  # [B, nhead, seq_len, d_k]
        self._cache_v: torch.Tensor = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, d_model] → [B, nhead, L, d_k]"""
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.d_k).transpose(1, 2)

    # ── Cache management ──────────────────────────────────────────────────────

    def reset_cache(self, B: int, device: torch.device) -> None:
        """Initialize empty FIFO caches (zero-filled)."""
        self._cache_k = torch.zeros(B, self.nhead, self.seq_len, self.d_k, device=device)
        self._cache_v = torch.zeros(B, self.nhead, self.seq_len, self.d_k, device=device)

    def init_cache_from(self, x: torch.Tensor) -> None:
        """
        Pre-fill FIFO cache from reference sequence x: [B, L_ref, d_model].
        Takes the last seq_len positions.
        Called once per layer before online evaluation begins.
        """
        with torch.no_grad():
            K = self._split_heads(self.W_k(x))[:, :, -self.seq_len:, :]
            V = self._split_heads(self.W_v(x))[:, :, -self.seq_len:, :]
        self._cache_k = K.detach().clone()
        self._cache_v = V.detach().clone()

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Full-sequence MHA. x: [B, L, d_model] → [B, L, d_model]"""
        B, L, _ = x.shape
        Q = self._split_heads(self.W_q(x))   # [B, nhead, L, d_k]
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out  = (torch.matmul(attn, V)
                .transpose(1, 2).contiguous()
                .view(B, L, self.d_model))
        return self.W_o(out)

    # ── Online forward (Eq. 9-13) ─────────────────────────────────────────────

    def forward_online(self, x_new: torch.Tensor) -> torch.Tensor:
        """
        Incremental step. x_new: [B, 1, d_model] → [B, 1, d_model].
        Implements FIFO cache exactly as described in Eq. 9-13.
        """
        B = x_new.size(0)
        if self._cache_k is None:
            self.reset_cache(B, x_new.device)

        k_new = self._split_heads(self.W_k(x_new))   # [B, nhead, 1, d_k]
        v_new = self._split_heads(self.W_v(x_new))
        q_new = self._split_heads(self.W_q(x_new))

        # FIFO update — evict oldest (position 0), append new at tail
        self._cache_k = torch.cat([self._cache_k[:, :, 1:, :], k_new], dim=2)
        self._cache_v = torch.cat([self._cache_v[:, :, 1:, :], v_new], dim=2)

        # Attention for current query only — O(L·d_k)
        scores = (torch.matmul(q_new, self._cache_k.transpose(-2, -1))
                  / math.sqrt(self.d_k))                              # [B, nhead, 1, L]
        attn   = self.dropout(torch.softmax(scores, dim=-1))
        out    = torch.matmul(attn, self._cache_v)                    # [B, nhead, 1, d_k]
        out    = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        return self.W_o(out)


# =============================================================================
# ENCODER LAYER
# =============================================================================

class EncoderLayer(nn.Module):
    """ConMHA → ContinualNorm → FFN → ContinualNorm  (post-norm, Qian 2025)."""

    def __init__(self, d_model: int, nhead: int, seq_len: int,
                 ff_dim: int, dropout: float):
        super().__init__()
        self.attn    = ConMHA(d_model, nhead, seq_len, dropout)
        self.norm1   = ContinualNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.norm2   = ContinualNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Training: [B, L, d_model] → [B, L, d_model]"""
        x = self.norm1(x + self.dropout(self.attn(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

    def forward_online(self, x_new: torch.Tensor) -> torch.Tensor:
        """Online: [B, 1, d_model] → [B, 1, d_model]"""
        x_new = self.norm1(x_new + self.dropout(self.attn.forward_online(x_new)))
        x_new = self.norm2(x_new + self.dropout(self.ffn(x_new)))
        return x_new


# =============================================================================
# DECODER LAYER
# Note: decoder self-attention uses standard MHA (not ConMHA) because
# decoder input is always length 2 (X_token + X_0). For L=2, FIFO ConMHA
# is identical to standard MHA — no approximation involved.
# =============================================================================

class DecoderLayer(nn.Module):
    """
    Masked MHA → ContinualNorm → Cross-Attention → ContinualNorm → FFN → ContinualNorm.
    Cross-attention query = decoder, key/value = encoder output cache.
    """

    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.norm1      = ContinualNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                                 batch_first=True)
        self.norm2      = ContinualNorm(d_model)
        self.ffn        = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.norm3   = ContinualNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        sa_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))
        ca_out, _ = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(x + self.dropout(ca_out))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


# =============================================================================
# FULL MODEL
# =============================================================================

class ILETransformerIFS(nn.Module):
    """
    IL-ETransformer with IFS gate (Model #7 — thesis contribution).

    Identical to ILETransformerNoIFS (Model #6) except for IFSGate:
      Before ConvTokenEmbedding: X_gated = X * sigmoid(g)
      g is a learnable parameter updated jointly via TS-EWC gradient steps.

    Two execution modes:
      Training : forward(x_enc, tf_enc) — batch, full sequence
      Online   : init_online(x_w, tf_w) once, then forward_online(x_new, tf_new)
    """

    def __init__(self, d_model: int, nhead: int, enc_layers: int,
                 dec_layers: int, ff_dim: int, dropout: float, seq_len: int):
        super().__init__()
        self.d_model  = d_model
        self.seq_len  = seq_len

        # IFS gate — MODEL #7 addition
        self.ifs_gate   = IFSGate(N_FEATURES)

        self.conv_embed = CausalConvTokenEmbedding(N_FEATURES, d_model)
        self.pos_enc    = PositionalEncoding(d_model)
        self.time_enc   = TimeEmbedding(d_model)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, nhead, seq_len, ff_dim, dropout)
            for _ in range(enc_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, nhead, ff_dim, dropout)
            for _ in range(dec_layers)
        ])
        self.fc = nn.Linear(d_model, 1)

        # Online state
        self._enc_out_cache: torch.Tensor = None  # [B, seq_len, d_model] FIFO
        self._online_pos:    int          = 0

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(self, x_enc: torch.Tensor, tf_enc: torch.Tensor) -> torch.Tensor:
        """
        x_enc  : [B, L, 6]
        tf_enc : [B, L, 3]
        returns: scalar prediction [B]
        """
        B = x_enc.size(0)

        # IFS gate applied before token embedding
        x_gated = self.ifs_gate(x_enc)                          # [B, L, 6]

        enc_emb = (self.conv_embed(x_gated)
                   + self.pos_enc(x_enc)
                   + self.time_enc(tf_enc))                      # [B, L, d_model]

        enc_out = enc_emb
        for layer in self.encoder:
            enc_out = layer(enc_out)                             # [B, L, d_model]

        # Decoder input: X_de = Concat(X_token, X_0)
        x_token  = enc_emb[:, -DEC_TOKEN_LEN:, :]                    # [B, 1, d_model]
        x_zero   = torch.zeros(B, 1, self.d_model, device=x_enc.device)
        dec_in   = torch.cat([x_token, x_zero], dim=1)               # [B, 2, d_model]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            dec_in.size(1), device=x_enc.device
        )
        dec_out = dec_in
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, tgt_mask)

        return self.fc(dec_out[:, -1, :]).squeeze(-1)                 # [B]

    # ── Online initialization ─────────────────────────────────────────────────

    def init_online(self, x_warmup: torch.Tensor,
                    tf_warmup: torch.Tensor) -> None:
        """
        Pre-fills all FIFO caches from the last seq_len steps of warmup.
        Only the last seq_len rows are needed because all online caches
        have length seq_len. This prevents positional-encoding length
        mismatch when the warmup partition is longer than max_len.
        x_warmup : [B, L_w, 6]  normalized
        tf_warmup: [B, L_w, 3]
        """
        with torch.no_grad():
            x_seed  = x_warmup[:, -self.seq_len:, :]
            tf_seed = tf_warmup[:, -self.seq_len:, :]

            x_gated = self.ifs_gate(x_seed)

            # Seed causal conv buffer (post-gate)
            self.conv_embed.init_buf(x_gated)

            # Seed sequence embedding
            enc_emb = (self.conv_embed(x_gated)
                       + self.pos_enc(x_seed)
                       + self.time_enc(tf_seed))       # [B, seq_len, d_model]

            # Per-layer: init ConMHA cache from layer input, then compute output
            x = enc_emb
            for layer in self.encoder:
                layer.attn.init_cache_from(x)          # cache from layer INPUT
                x = layer(x)                           # compute layer output

            # Encoder-output FIFO cache for decoder cross-attention
            self._enc_out_cache = x[:, -self.seq_len:, :].detach().clone()

        self._online_pos = x_seed.size(1)
        self.eval()

    # ── Online forward (one step) ─────────────────────────────────────────────

    def forward_online(self, x_new: torch.Tensor,
                       tf_new: torch.Tensor) -> torch.Tensor:
        """
        One incremental step.
        x_new  : [B, 1, 6]  normalized
        tf_new : [B, 1, 3]
        returns: scalar [B]
        """
        B = x_new.size(0)

        # IFS gate applied before token embedding
        x_gated = self.ifs_gate(x_new)                                # [B, 1, 6]

        # Embedding for new token
        conv_out  = self.conv_embed.forward_online(x_gated)           # [B, 1, d_model]
        pos_out   = self.pos_enc.at(self.seq_len - 1, x_new.device)  # [1, 1, d_model]
        time_out  = self.time_enc(tf_new)                             # [B, 1, d_model]
        x_emb_new = conv_out + pos_out + time_out                     # [B, 1, d_model]

        # Encoder: each layer uses FIFO ConMHA — O(L·d_k) per layer
        enc_out_new = x_emb_new
        for layer in self.encoder:
            enc_out_new = layer.forward_online(enc_out_new)           # [B, 1, d_model]

        # Update encoder-output FIFO cache for decoder cross-attention
        if self._enc_out_cache is None:
            self._enc_out_cache = enc_out_new.expand(
                B, self.seq_len, self.d_model
            ).detach().clone()
        else:
            self._enc_out_cache = torch.cat(
                [self._enc_out_cache[:, 1:, :], enc_out_new.detach()], dim=1
            )                                                          # [B, seq_len, d_model]

        # Decoder: X_de = [X_token, X_0]
        x_zero   = torch.zeros(B, 1, self.d_model, device=x_new.device)
        dec_in   = torch.cat([x_emb_new, x_zero], dim=1)             # [B, 2, d_model]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            2, device=x_new.device
        )
        dec_out = dec_in
        for layer in self.decoder:
            dec_out = layer(dec_out, self._enc_out_cache, tgt_mask)

        self._online_pos += 1
        return self.fc(dec_out[:, -1, :]).squeeze(-1)                 # [B]

    # ── Cache refresh after TS-EWC parameter update ───────────────────────────

    def refresh_caches(self, x_hist: torch.Tensor,
                       tf_hist: torch.Tensor) -> None:
        """
        After a TS-EWC gradient step, model weights AND gate values change
        → cached K/V are stale. Recompute from recent history.
        x_hist : [B, ≥seq_len, 6]  normalized
        tf_hist: [B, ≥seq_len, 3]
        """
        with torch.no_grad():
            x_gated = self.ifs_gate(x_hist)
            self.conv_embed.init_buf(x_gated)
            enc_emb = (self.conv_embed(x_gated)
                       + self.pos_enc(x_hist)
                       + self.time_enc(tf_hist))
            x = enc_emb
            for layer in self.encoder:
                layer.attn.init_cache_from(x)
                x = layer(x)
            self._enc_out_cache = x[:, -self.seq_len:, :].detach().clone()
        self.eval()

    def reset_online(self) -> None:
        """Reset all online state between symbol/horizon runs."""
        self._enc_out_cache = None
        self._online_pos    = 0
        self.conv_embed.reset_buf()
        for layer in self.encoder:
            layer.attn._cache_k = None
            layer.attn._cache_v = None


# =============================================================================
# TS-EWC  (Eq. 22, Qian 2025)
# =============================================================================

class TSEWC:
    """
    Temporal Streaming Elastic Weight Consolidation.

    novelty buffer  : gates updates — fires when MSE > dynamic threshold
    familiarity buf : circular buffer (FAM_BUF_SIZE) of (x, tf, y) tuples
    Fisher matrix   : diagonal, estimated from FISHER_N_SAMPLES buffer samples
    EWC loss        : L_task + λ Σ F_i(θ_i − θ*_i)²
    threshold       : 1.5× rolling mean of last 100 MSE errors (dynamic)
    cache refresh   : called after each parameter update via model.refresh_caches()

    Note: IFS gate logits are included in the Fisher / EWC regularization
    automatically because they are nn.Parameters of the model.
    """

    def __init__(self, model: ILETransformerIFS, ewc_lambda: float):
        self.model     = model
        self.lam       = ewc_lambda
        self.criterion = nn.MSELoss()

        self.params_star: dict = {}
        self.fisher:      dict = {}

        self.buf_x:  deque = deque(maxlen=FAM_BUF_SIZE)
        self.buf_tf: deque = deque(maxlen=FAM_BUF_SIZE)
        self.buf_y:  deque = deque(maxlen=FAM_BUF_SIZE)

        self.err_hist: deque  = deque(maxlen=100)
        self.threshold: float = None

    def add_sample(self, x: torch.Tensor, tf: torch.Tensor,
                   y: torch.Tensor) -> None:
        self.buf_x.append(x.detach().cpu())
        self.buf_tf.append(tf.detach().cpu())
        self.buf_y.append(y.detach().cpu())

    def update_threshold(self, mse: float) -> None:
        self.err_hist.append(mse)
        if len(self.err_hist) >= 10:
            self.threshold = float(np.mean(self.err_hist)) * 1.5

    def is_novel(self, mse: float) -> bool:
        return (self.threshold is None) or (mse > self.threshold)

    def consolidate(self) -> None:
        """Estimate diagonal Fisher matrix and save θ* anchor."""
        n = len(self.buf_x)
        if n < 5:
            return

        n_s  = min(FISHER_N_SAMPLES, n)
        idxs = random.sample(range(n), n_s)

        fisher = {
            name: torch.zeros_like(p, device=DEVICE)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.model.train()
        for i in idxs:
            x  = self.buf_x[i].to(DEVICE)
            tf = self.buf_tf[i].to(DEVICE)
            y  = self.buf_y[i].to(DEVICE)
            self.model.zero_grad()
            self.criterion(self.model(x, tf), y).backward()
            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.data.pow(2)

        for name in fisher:
            fisher[name] /= n_s

        self.fisher      = fisher
        self.params_star = {
            name: p.data.clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        self.model.eval()

    def ewc_loss(self, x: torch.Tensor, tf: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        """L_task + λ Σ F_i(θ_i − θ*_i)²  (Eq. 22)"""
        task = self.criterion(self.model(x, tf), y)
        if not self.params_star:
            return task
        pen = torch.tensor(0.0, device=DEVICE)
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.fisher:
                pen = pen + (self.fisher[name]
                             * (p - self.params_star[name]).pow(2)).sum()
        return task + self.lam * pen

    def step(self, x: torch.Tensor, tf: torch.Tensor, y: torch.Tensor,
             optimizer: torch.optim.Optimizer,
             x_hist: torch.Tensor, tf_hist: torch.Tensor) -> None:
        """
        One TS-EWC gradient step, followed by re-consolidation and cache refresh.
        x_hist / tf_hist: recent normalized history for cache refresh.
        """
        self.model.train()
        optimizer.zero_grad()
        self.ewc_loss(x, tf, y).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        self.consolidate()
        # Weights changed (including gate) → FIFO K/V caches are stale → refresh
        self.model.refresh_caches(x_hist, tf_hist)


# =============================================================================
# TRAINING HELPER
# =============================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module) -> float:
    model.train()
    total = 0.0
    for x_enc, tf_enc, y in loader:
        x_enc, tf_enc, y = x_enc.to(DEVICE), tf_enc.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x_enc, tf_enc), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
    return total / max(len(loader.dataset), 1)


def _make_model(hp: dict) -> ILETransformerIFS:
    return ILETransformerIFS(
        d_model    = hp["d_model"],
        nhead      = hp["nhead"],
        enc_layers = hp["enc_layers"],
        dec_layers = 1,
        ff_dim     = hp["ff_dim"],
        dropout    = hp["dropout"],
        seq_len    = hp["seq_len"],
    ).to(DEVICE)


# =============================================================================
# VALIDATION  (walk-forward, dynamic scaler — used for random search)
# =============================================================================

def _seed_familiarity_buf(tsewc: TSEWC, warmup: pd.DataFrame,
                          seq_len: int) -> None:
    """Populate familiarity buffer from last ~200 warmup samples."""
    feat = warmup[FEATURE_COLS].values.astype(np.float32)
    tf   = warmup[TIME_COLS].values.astype(np.int64)
    h    = deque(list(feat), maxlen=SCALE_WINDOW + seq_len)
    start = max(0, len(warmup) - seq_len - 200)
    for i in range(start, len(warmup) - seq_len - 1):
        h_arr    = np.array(list(h), dtype=np.float32)
        m, s     = dynamic_stats(h_arr)
        wf_n     = norm_feat(feat[i: i + seq_len], m, s)
        wt       = tf[i: i + seq_len]
        tgt_n    = norm_target(float(warmup["delta_target"].iloc[i + seq_len]), m, s)
        tsewc.add_sample(
            torch.tensor(wf_n, dtype=torch.float32).unsqueeze(0),
            torch.tensor(wt,   dtype=torch.long).unsqueeze(0),
            torch.tensor([tgt_n], dtype=torch.float32),
        )
    tsewc.consolidate()


def evaluate_val_mae(model: ILETransformerIFS,
                     warmup: pd.DataFrame,
                     val: pd.DataFrame,
                     hp: dict) -> float:
    seq_len    = hp["seq_len"]
    online_opt = torch.optim.Adam(model.parameters(), lr=hp["online_lr"])
    tsewc      = TSEWC(model, hp["ewc_lambda"])

    def _infer_delay(frame: pd.DataFrame) -> int:
        close = frame["close"].values.astype(np.float64)
        tgt   = frame["delta_target"].values.astype(np.float64)
        for h in TARGET_HORIZONS_BARS:
            if len(frame) <= h + 5:
                continue
            ok = np.isclose(tgt[:-h], close[h:] - close[:-h], rtol=1e-7, atol=1e-5)
            if float(np.mean(ok)) > 0.99:
                return int(h)
        return 1

    delay = _infer_delay(val)

    warmup_feat = warmup[FEATURE_COLS].values.astype(np.float32)
    warmup_tf   = warmup[TIME_COLS].values.astype(np.int64)
    val_feat    = val[FEATURE_COLS].values.astype(np.float32)
    val_tf      = val[TIME_COLS].values.astype(np.int64)
    val_tgt     = val["delta_target"].values.astype(np.float64)

    h_feat = deque(list(warmup_feat), maxlen=SCALE_WINDOW + seq_len)
    h_tf   = deque(list(warmup_tf),   maxlen=seq_len)

    h_arr         = np.array(list(h_feat), dtype=np.float32)
    mean, std     = dynamic_stats(h_arr)
    xw  = torch.tensor(norm_feat(warmup_feat[-seq_len:], mean, std), dtype=torch.float32)\
              .unsqueeze(0).to(DEVICE)
    tfw = torch.tensor(warmup_tf[-seq_len:], dtype=torch.long).unsqueeze(0).to(DEVICE)
    model.init_online(xw, tfw)

    _seed_familiarity_buf(tsewc, warmup, seq_len)

    preds, acts = [], []
    pending_updates = deque()

    for t in range(len(val)):
        h_arr     = np.array(list(h_feat), dtype=np.float32)
        mean, std = dynamic_stats(h_arr)
        feat_n    = norm_feat(val_feat[t:t+1], mean, std)
        tf_t      = val_tf[t:t+1]

        x_new  = torch.tensor(feat_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        tf_new = torch.tensor(tf_t,   dtype=torch.long).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            pred_n = model.forward_online(x_new, tf_new).item()
        preds.append(denorm_target(pred_n, mean, std))
        acts.append(float(val_tgt[t]))

        h_window_n  = norm_feat(h_arr[-seq_len:], mean, std)
        h_tf_window = np.array(list(h_tf))[-seq_len:]
        x_buf  = torch.tensor(h_window_n,  dtype=torch.float32).unsqueeze(0)
        tf_buf = torch.tensor(h_tf_window, dtype=torch.long).unsqueeze(0)

        pending_updates.append({
            "pred_n": pred_n,
            "mean": mean.copy(),
            "std": std.copy(),
            "x_buf": x_buf,
            "tf_buf": tf_buf,
            "target_idx": t,
        })

        if len(pending_updates) > delay:
            item    = pending_updates.popleft()
            true_n  = norm_target(float(val_tgt[item["target_idx"]]),
                                  item["mean"], item["std"])
            y_t     = torch.tensor([true_n], dtype=torch.float32).to(DEVICE)
            mse_err = (item["pred_n"] - true_n) ** 2

            tsewc.add_sample(item["x_buf"], item["tf_buf"], y_t)
            tsewc.update_threshold(mse_err)

            if (t + 1) % EWC_UPDATE_FREQ == 0 and tsewc.is_novel(mse_err):
                h_n   = norm_feat(h_arr[-seq_len:], mean, std)
                h_tf_ = np.array(list(h_tf)[-seq_len:], dtype=np.int64)
                xh    = torch.tensor(h_n,   dtype=torch.float32).unsqueeze(0).to(DEVICE)
                tfh   = torch.tensor(h_tf_, dtype=torch.long).unsqueeze(0).to(DEVICE)
                tsewc.step(
                    item["x_buf"].to(DEVICE),
                    item["tf_buf"].to(DEVICE),
                    y_t, online_opt, xh, tfh,
                )

        h_feat.append(val_feat[t])
        h_tf.append(val_tf[t])

    return float(mean_absolute_error(acts, preds))



# =============================================================================
# RANDOM SEARCH  —  Bergstra & Bengio (2012)
# =============================================================================

def sample_hp(rng: random.Random) -> dict:
    hp = {k: rng.choice(v) for k, v in HP_SPACE.items()}
    valid = [n for n in HP_SPACE["nhead"] if hp["d_model"] % n == 0]
    hp["nhead"] = rng.choice(valid) if valid else 1
    return hp


def run_trial(hp: dict, warmup: pd.DataFrame, val: pd.DataFrame,
              warmup_mean: np.ndarray, warmup_std: np.ndarray) -> float:
    set_seed(RANDOM_SEED)
    ds = WarmupDataset(warmup, hp["seq_len"], warmup_mean, warmup_std)
    if len(ds) == 0:
        return float("inf")
    loader    = DataLoader(ds, hp["batch_size"], shuffle=True)
    model     = _make_model(hp)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion = nn.MSELoss()
    for _ in range(hp["epochs"]):
        train_one_epoch(model, loader, optimizer, criterion)
    return evaluate_val_mae(model, warmup, val, hp)


def random_search(warmup: pd.DataFrame, val: pd.DataFrame,
                  warmup_mean: np.ndarray, warmup_std: np.ndarray) -> dict:
    rng      = random.Random(RANDOM_SEED)
    best_hp  = None
    best_mae = float("inf")
    print(f"  [random_search] {N_TRIALS} trials — Bergstra & Bengio (2012)")
    for t in range(N_TRIALS):
        hp   = sample_hp(rng)
        mae  = run_trial(hp, warmup, val, warmup_mean, warmup_std)
        flag = " *" if mae < best_mae else ""
        print(f"    trial {t+1:02d}/{N_TRIALS} | val_mae={mae:.6f}{flag} | {hp}")
        if mae < best_mae:
            best_mae, best_hp = mae, hp
    print(f"  => best_val_mae={best_mae:.6f} | best_hp={best_hp}")
    return best_hp


# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

def train_final(hp: dict, warmup: pd.DataFrame,
                warmup_mean: np.ndarray,
                warmup_std: np.ndarray) -> ILETransformerIFS:
    set_seed(RANDOM_SEED)
    ds = WarmupDataset(warmup, hp["seq_len"], warmup_mean, warmup_std)
    if len(ds) == 0:
        raise ValueError("Empty warmup dataset.")
    loader    = DataLoader(ds, hp["batch_size"], shuffle=True)
    model     = _make_model(hp)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    criterion = nn.MSELoss()
    for epoch in range(hp["epochs"]):
        loss = train_one_epoch(model, loader, optimizer, criterion)
        if (epoch + 1) % 5 == 0:
            print(f"    epoch {epoch+1}/{hp['epochs']} | train_loss={loss:.6f}")
    model.eval()
    return model


# =============================================================================
# ONLINE EVALUATION  (with gate history logging for RQ2)
# =============================================================================

def evaluate_online(model: ILETransformerIFS,
                    warmup: pd.DataFrame,
                    online: pd.DataFrame,
                    hp: dict,
                    lookback: int,
                    symbol: str,
                    horizon: int) -> dict:
    """
    Walk-forward online evaluation:
      1. Dynamic per-feature scaler  (rolling SCALE_WINDOW)
      2. ConMHA FIFO caches         (initialized from warmup, refreshed post-EWC)
      3. TS-EWC online update       (every EWC_UPDATE_FREQ bars, novelty-gated)
      4. IFS gate logging           (every EWC_UPDATE_FREQ bars → gate_history)

    Direction: N/A — delta target is regression only.
    First `lookback` steps skipped — matches all other baselines.
    """
    seq_len    = hp["seq_len"]
    delay      = int(horizon)
    online_opt = torch.optim.Adam(model.parameters(), lr=hp["online_lr"])
    tsewc      = TSEWC(model, hp["ewc_lambda"])

    warmup_feat = warmup[FEATURE_COLS].values.astype(np.float32)
    warmup_tf   = warmup[TIME_COLS].values.astype(np.int64)
    online_feat = online[FEATURE_COLS].values.astype(np.float32)
    online_tf   = online[TIME_COLS].values.astype(np.int64)
    delta_tgt   = online["delta_target"].values.astype(np.float64)
    datetimes   = online["datetime"].values   # for gate_history timestamps

    h_feat = deque(list(warmup_feat), maxlen=SCALE_WINDOW + seq_len)
    h_tf   = deque(list(warmup_tf),   maxlen=seq_len)

    # Initialize FIFO caches from warmup
    h_arr         = np.array(list(h_feat), dtype=np.float32)
    mean, std     = dynamic_stats(h_arr)
    xw  = torch.tensor(norm_feat(warmup_feat[-seq_len:], mean, std), dtype=torch.float32)\
              .unsqueeze(0).to(DEVICE)
    tfw = torch.tensor(warmup_tf[-seq_len:], dtype=torch.long).unsqueeze(0).to(DEVICE)
    model.init_online(xw, tfw)

    _seed_familiarity_buf(tsewc, warmup, seq_len)

    pred_delta_all = []
    pending_updates = deque()
    gate_records   = []   # Accumulated gate snapshots for this (symbol, horizon)

    for t in range(len(online)):
        h_arr     = np.array(list(h_feat), dtype=np.float32)
        mean, std = dynamic_stats(h_arr)
        feat_n    = norm_feat(online_feat[t:t+1], mean, std)
        tf_t      = online_tf[t:t+1]

        x_new  = torch.tensor(feat_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        tf_new = torch.tensor(tf_t,   dtype=torch.long).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            pred_n = model.forward_online(x_new, tf_new).item()
        pred_delta_all.append(denorm_target(pred_n, mean, std))

        h_window_n  = norm_feat(h_arr[-seq_len:], mean, std)
        h_tf_window = np.array(list(h_tf))[-seq_len:]
        x_buf  = torch.tensor(h_window_n,  dtype=torch.float32).unsqueeze(0)
        tf_buf = torch.tensor(h_tf_window, dtype=torch.long).unsqueeze(0)

        pending_updates.append({
            "pred_n": pred_n,
            "mean": mean.copy(),
            "std": std.copy(),
            "x_buf": x_buf,
            "tf_buf": tf_buf,
            "target_idx": t,
        })

        ewc_update_done = False
        if len(pending_updates) > delay:
            item    = pending_updates.popleft()
            true_n  = norm_target(float(delta_tgt[item["target_idx"]]),
                                  item["mean"], item["std"])
            y_t     = torch.tensor([true_n], dtype=torch.float32).to(DEVICE)
            mse_err = (item["pred_n"] - true_n) ** 2

            tsewc.add_sample(item["x_buf"], item["tf_buf"], y_t)
            tsewc.update_threshold(mse_err)

            if (t + 1) % EWC_UPDATE_FREQ == 0 and tsewc.is_novel(mse_err):
                h_n   = norm_feat(h_arr[-seq_len:], mean, std)
                h_tf_ = np.array(list(h_tf)[-seq_len:], dtype=np.int64)
                xh    = torch.tensor(h_n,   dtype=torch.float32).unsqueeze(0).to(DEVICE)
                tfh   = torch.tensor(h_tf_, dtype=torch.long).unsqueeze(0).to(DEVICE)
                tsewc.step(
                    item["x_buf"].to(DEVICE),
                    item["tf_buf"].to(DEVICE),
                    y_t, online_opt, xh, tfh,
                )
                ewc_update_done = True

        if (t + 1) % EWC_UPDATE_FREQ == 0:
            gates = model.ifs_gate.gate_values()
            gate_records.append({
                "bar_index":        t,
                "datetime":         datetimes[t],
                "symbol":           symbol,
                "horizon":          horizon,
                "gate_open":        float(gates[0]),
                "gate_high":        float(gates[1]),
                "gate_low":         float(gates[2]),
                "gate_close":       float(gates[3]),
                "gate_change_ratio":float(gates[4]),
                "gate_macd":        float(gates[5]),
                "ewc_update":       bool(ewc_update_done),
            })

        h_feat.append(online_feat[t])
        h_tf.append(online_tf[t])

    # Save gate history for this (symbol, horizon)
    if gate_records:
        gate_df = pd.DataFrame(gate_records)
        append_gate_history(symbol, horizon, gate_df)
        print(
            f"  => gate history ({len(gate_df)} snapshots) saved to "
            f"{GATE_DIR / f'{GATE_HISTORY_PREFIX}_{symbol}_h{horizon}.csv'}"
    )

    pred_delta_all = np.array(pred_delta_all, dtype=np.float64)

    return {
        "pred_delta":   pred_delta_all[lookback:],
        "actual_delta": delta_tgt[lookback:],
    }



# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(out: dict) -> dict:
    pred_p = out["pred_delta"]
    act_p  = out["actual_delta"]
    mae    = float(mean_absolute_error(act_p, pred_p))
    rmse   = float(math.sqrt(mean_squared_error(act_p, pred_p)))
    rel    = float(mae / (np.mean(np.abs(act_p)) + 1e-8) * 100.0)
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
    print("MODEL #7b — IL-ETransformer + IFS  |  6 features  |  delta + direction")
    print("=" * 100)
    print(f"symbols      = {SYMBOLS}")
    print(f"horizons     = {TARGET_HORIZONS_BARS}")
    print(f"features     = {FEATURE_COLS}")
    print(f"snapshot     = {SNAPSHOT_START_NY} → {SNAPSHOT_END_NY}")
    print(f"split        = {WARMUP_RATIO}:{VAL_RATIO}:{ONLINE_RATIO}")
    print(f"lookback     = {LOOKBACK}")
    print(f"scale_window = {SCALE_WINDOW}")
    print(f"ewc_freq     = {EWC_UPDATE_FREQ} bars")
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

            d                   = make_targets(df_sym, horizon)
            warmup, val, online = split_warmup_val_online(d)

            print(f"rows={len(d):,} | warmup={len(warmup):,} | "
                  f"val={len(val):,} | online={len(online):,} | "
                  f"eval_rows={len(online) - LOOKBACK:,}")

            warmup_mean, warmup_std = compute_static_stats(warmup)
            best_hp = random_search(warmup, val, warmup_mean, warmup_std)

            print("  [train] final model on warmup ...")
            try:
                model = train_final(best_hp, warmup, warmup_mean, warmup_std)
            except Exception as e:
                print(f"[ERROR] training failed: {e}"); continue

            print("  [eval] online phase (with gate tracking) ...")
            try:
                out = evaluate_online(
                    model, warmup, online, best_hp, LOOKBACK, symbol, horizon
                )
            except Exception as e:
                print(f"[ERROR] evaluation failed: {e}"); continue

            m = compute_metrics(out)
            print(
                f"delta_mae={m['delta_mae']:.6f} | "
                f"delta_rmse={m['delta_rmse']:.6f} | "
                f"delta_rel_mae={m['delta_relative_mae_pct']:.4f}%"
            )

            result_row = {
                "symbol": symbol, "horizon": horizon,
                "model":           "iltransformer_ifs_delta",
                "best_seq_len":    best_hp["seq_len"],
                "best_d_model":    best_hp["d_model"],
                "best_nhead":      best_hp["nhead"],
                "best_enc_layers": best_hp["enc_layers"],
                "best_dec_layers": 1,
                "best_ff_dim":     best_hp["ff_dim"],
                "best_dropout":    best_hp["dropout"],
                "best_lr":         best_hp["learning_rate"],
                "best_online_lr":  best_hp["online_lr"],
                "best_batch_size": best_hp["batch_size"],
                "best_epochs":     best_hp["epochs"],
                "best_ewc_lambda": best_hp["ewc_lambda"],
                "delta_mae":              m["delta_mae"],
                "delta_rmse":             m["delta_rmse"],
                "delta_relative_mae_pct": m["delta_relative_mae_pct"],
            }
            append_row(OUTPUT_SUMMARY, result_row)
            print(f"  => saved to {OUTPUT_SUMMARY}")

            pred_df = online.iloc[LOOKBACK:].reset_index(drop=True)[
                ["symbol","datetime","open","high","low","close",
                 "change_ratio","macd","delta_target"]
            ].copy()
            pred_df["horizon"]         = horizon
            pred_df["model"]           = "iltransformer_ifs_delta"
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
        cols    = ["symbol","horizon","model","delta_mae","delta_rmse",
                   "delta_relative_mae_pct"]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()