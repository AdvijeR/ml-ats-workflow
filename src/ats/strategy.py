import backtrader as bt


class PortfolioThresholdLongFlatStrategy(bt.Strategy):
    """
    Multi-stock portfolio strategy.

    If pred_return > threshold:
        stock is active and gets equal weight among all active stocks.
    Else:
        stock is flat.
    """

    params = dict(
        threshold=0.003,
        printlog=False,
    )

    def __init__(self):
        self.orders = {}
        self.trade_count = 0

    def log(self, txt):
        if self.p.printlog:
            dt = self.datas[0].datetime.date(0)
            print(f"{dt} | {txt}")

    def next(self):
        active_datas = []

        for data in self.datas:
            if data.pred_return[0] > self.p.threshold:
                active_datas.append(data)

        n_active = len(active_datas)
        target_pct = 1.0 / n_active if n_active > 0 else 0.0

        for data in self.datas:
            if self.orders.get(data) is not None:
                continue

            in_position = self.getposition(data).size != 0
            signal_on = data.pred_return[0] > self.p.threshold

            if signal_on:
                self.orders[data] = self.order_target_percent(data=data, target=target_pct)
            else:
                if in_position:
                    self.orders[data] = self.order_target_percent(data=data, target=0.0)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            self.trade_count += 1

        self.orders[order.data] = None