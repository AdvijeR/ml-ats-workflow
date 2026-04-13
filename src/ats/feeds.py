import backtrader as bt


class PredictionData(bt.feeds.PandasData):
    """
    Minimal Backtrader feed for prediction-based ATS.
    Expects a DataFrame with columns:
      open, high, low, close, volume, pred_return
    """
    lines = ("pred_return",)

    params = (
        ("datetime", None),
        ("open",     "open"),
        ("high",     "high"),
        ("low",      "low"),
        ("close",    "close"),
        ("volume",   "volume"),
        ("openinterest", -1),
        ("pred_return",  "pred_return"),
    )