from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def build_prediction_timeline(
    dates,
    raw_df_split: pd.DataFrame,
    y_true,
    y_pred,
    price_col: str = "open",
) -> pd.DataFrame:
    """
    Build a chronological prediction timeline for one split.

    Parameters
    ----------
    dates : sequence
        Dates aligned with the target values produced by sequence creation.
    raw_df_split : pd.DataFrame
        Original dataframe for the current split. Must contain at least
        ['date', price_col].
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    price_col : str, default="open"
        Price column used as the forecasting target/reference.

    Returns
    -------
    pd.DataFrame
        Dataframe with aligned prediction outputs and simple return fields
        for downstream ATS evaluation.
    """
    out = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "y_true": np.asarray(y_true).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1),
    })

    raw_df = raw_df_split.copy()
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_df = raw_df.sort_values("date").reset_index(drop=True)

    if price_col not in raw_df.columns:
        raise ValueError(f"Missing required price column: {price_col}")

    prev_col_name = f"prev_{price_col}"
    raw_df[prev_col_name] = raw_df[price_col].shift(1)

    lookup_df = raw_df[["date", prev_col_name]].copy()
    out = out.merge(lookup_df, on="date", how="left")

    out["true_return"] = (out["y_true"] - out[prev_col_name]) / out[prev_col_name]
    out["pred_return"] = (out["y_pred"] - out[prev_col_name]) / out[prev_col_name]

    out["true_direction"] = (out["true_return"] > 0).astype(int)
    out["pred_direction"] = (out["pred_return"] > 0).astype(int)

    out = out.sort_values("date").reset_index(drop=True)
    out = out.dropna(subset=[prev_col_name]).reset_index(drop=True)

    if price_col == "open":
        out = out.rename(columns={prev_col_name: "prev_open"})

    return out


def load_full_prediction_timeline(
    ticker: str,
    predictions_dir: Path,
) -> pd.DataFrame:
    """
    Load saved validation and test prediction files and concatenate them
    into one full chronological prediction timeline.
    """
    val_path = predictions_dir / f"{ticker}_val_predictions.csv"
    test_path = predictions_dir / f"{ticker}_test_predictions.csv"

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

    Earlier chunk = ATS validation
    Later chunk = ATS test
    """
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1.")

    full_df = full_df.copy()
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(full_df) * split_ratio)

    ats_val_df = full_df.iloc[:split_idx].copy()
    ats_test_df = full_df.iloc[split_idx:].copy()

    if len(ats_val_df) == 0 or len(ats_test_df) == 0:
        raise ValueError("ATS split produced an empty validation or test set.")

    return ats_val_df, ats_test_df


def prepare_bt_df_from_prediction_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a prediction dataframe into a Backtrader-ready dataframe.

    Expected input columns at minimum:
    - date
    - prev_open
    - pred_return

    For this educational baseline, prev_open is used as a simple proxy
    for OHLC values.
    """
    required_cols = ["date", "prev_open", "pred_return"]
    missing = [c for c in required_cols if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prediction df: {missing}")

    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    df["open"] = df["prev_open"]
    df["high"] = df["prev_open"]
    df["low"] = df["prev_open"]
    df["close"] = df["prev_open"]
    df["volume"] = 1.0

    bt_df = df[["open", "high", "low", "close", "volume", "pred_return"]].copy()
    return bt_df