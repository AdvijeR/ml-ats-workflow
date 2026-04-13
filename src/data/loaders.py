from pathlib import Path
import yfinance as yf
import pandas as pd


def download_stock_data(
    tickers,
    start_date,
    end_date,
    data_folder,
    overwrite=False,
):
    data_folder = Path(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for ticker in tickers:
        csv_path = data_folder / f"{ticker}.csv"

        if csv_path.exists() and not overwrite:
            print(f"{ticker}: already exists. Skipping.")
            saved_paths.append(csv_path)
            continue

        print(f"Downloading {ticker}...")
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                print(f"{ticker}: no data.")
                continue

            df = df.reset_index()
            df["ticker"] = ticker

            df = df[["Date", "Open", "High", "Low", "Close", "Volume", "ticker"]]
            df.columns = ["date", "open", "high", "low", "close", "volume", "ticker"]

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            assert len(df) > 0, f"{ticker}: empty dataframe before saving"
            assert df["date"].is_monotonic_increasing, f"{ticker}: dates not sorted"

            df.to_csv(csv_path, index=False)
            saved_paths.append(csv_path)

            print(f"{ticker}: saved to {csv_path}")

        except Exception as e:
            print(f"{ticker}: failed -> {e}")

    return saved_paths


def load_local_stock_csv(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    assert len(df) > 0, f"{csv_path}: loaded empty dataframe"
    assert df["date"].is_monotonic_increasing, f"{csv_path}: dates not sorted"

    return df