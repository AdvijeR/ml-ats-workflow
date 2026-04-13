from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd


def load_full_prediction_timeline(
    ticker: str,
    predictions_dir: Path,
) -> pd.DataFrame:
    """
    Load existing saved no-retrain prediction files and concatenate them
    into one full chronological prediction timeline.
    """
    val_path = predictions_dir / f"lstm_noretrain_{ticker}_val_preds.csv"
    test_path = predictions_dir / f"lstm_noretrain_{ticker}_preds.csv"

    val_df = pd.read_csv(val_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])

    full_df = pd.concat([val_df, test_df], axis=0, ignore_index=True)
    full_df = full_df.sort_values("date")
    full_df = full_df.drop_duplicates(subset=["date"]).reset_index(drop=True)

    return full_df


def split_prediction_timeline_for_ats(
    full_df: pd.DataFrame,
    split_ratio: float = 0.4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split one full prediction timeline into ATS validation and ATS test sets.
    Earlier chunk = validation, later chunk = test.
    """
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1.")

    full_df = full_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(full_df) * split_ratio)

    ats_val_df = full_df.iloc[:split_idx].copy()
    ats_test_df = full_df.iloc[split_idx:].copy()

    return ats_val_df, ats_test_df


def prepare_bt_df_from_prediction_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prediction dataframe into Backtrader-ready dataframe.

    Expected input columns at minimum:
    - date
    - prev_open
    - pred_return

    This uses prev_open as a pragmatic proxy for OHLC in the current baseline.
    """
    required_cols = ["date", "prev_open", "pred_return"]
    missing = [c for c in required_cols if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prediction df: {missing}")

    df = pred_df.copy()
    df = df.sort_values("date").set_index("date")

    df["open"] = df["prev_open"]
    df["high"] = df["prev_open"]
    df["low"] = df["prev_open"]
    df["close"] = df["prev_open"]
    df["volume"] = 1.0

    bt_df = df[["open", "high", "low", "close", "volume", "pred_return"]].copy()
    return bt_df