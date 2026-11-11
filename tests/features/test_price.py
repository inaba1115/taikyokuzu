from __future__ import annotations

import pandas as pd

from taikyokuzu.features import (
    rolling_volatility,
)


def test_rolling_volatility_can_disable_annualization():
    prices = pd.Series([100.0, 110.0, 99.0, 108.9])
    result = rolling_volatility(prices, window=2, annualization=None)
    expected_returns = prices.pct_change(fill_method=None)
    expected = expected_returns.rolling(window=2).std()
    pd.testing.assert_series_equal(result, expected)
