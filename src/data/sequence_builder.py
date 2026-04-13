import numpy as np
import pandas as pd


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_cols,
    target_col="open",
    sequence_length=50,
):
    X, y, target_dates = [], [], []

    working_df = df.copy().reset_index(drop=True)

    required_cols = list(feature_cols) + [target_col, "date"]
    for col in required_cols:
        if col not in working_df.columns:
            raise ValueError(f"Missing required column: {col}")

    for i in range(sequence_length, len(working_df)):
        X.append(working_df.iloc[i-sequence_length:i][feature_cols].values)
        y.append(working_df.iloc[i][target_col])
        target_dates.append(working_df.iloc[i]["date"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    assert len(X) > 0, "No sequences were created"
    assert len(X) == len(y) == len(target_dates), "Sequence output lengths do not match"
    assert X.shape[1] == sequence_length, f"Expected sequence length {sequence_length}, got {X.shape[1]}"
    assert X.shape[2] == len(feature_cols), f"Expected {len(feature_cols)} features, got {X.shape[2]}"

    return X, y, target_dates