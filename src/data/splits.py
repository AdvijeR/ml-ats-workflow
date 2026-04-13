import pandas as pd


def make_date_splits(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    date_col: str = "date",
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    train_df = df[df[date_col] <= pd.to_datetime(train_end)].copy()
    val_df = df[
        (df[date_col] > pd.to_datetime(train_end)) &
        (df[date_col] <= pd.to_datetime(val_end))
    ].copy()
    test_df = df[df[date_col] > pd.to_datetime(val_end)].copy()

    assert len(train_df) > 0, "Empty train split"
    assert len(val_df) > 0, "Empty validation split"
    assert len(test_df) > 0, "Empty test split"

    assert train_df[date_col].max() < val_df[date_col].min(), "LEAKAGE: train/val overlap"
    assert val_df[date_col].max() < test_df[date_col].min(), "LEAKAGE: val/test overlap"

    print(f"train: {train_df[date_col].min().date()} → {train_df[date_col].max().date()} ({len(train_df)} rows)")
    print(f"val:   {val_df[date_col].min().date()} → {val_df[date_col].max().date()} ({len(val_df)} rows)")
    print(f"test:  {test_df[date_col].min().date()} → {test_df[date_col].max().date()} ({len(test_df)} rows)")

    return train_df, val_df, test_df