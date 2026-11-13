from __future__ import annotations

import pandas as pd
import pytest

from taikyokuzu.targets import forward_returns, min_forward_returns


def test_forward_returns():
    prices = pd.Series([100.0, 110.0, 121.0, 133.1])
    result = forward_returns(prices, periods=2)
    assert result.iloc[0] == pytest.approx(0.21)
    assert result.iloc[1] == pytest.approx(0.21)
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])


def test_min_forward_returns():
    prices = pd.Series([100.0, 110.0, 105.0, 103.0, 108.0])

    result = min_forward_returns(prices, periods=3)

    assert result.iloc[0] == pytest.approx(0.03)
    assert result.iloc[1] == pytest.approx(103.0 / 110.0 - 1.0)
    assert result.iloc[2:].isna().all()
