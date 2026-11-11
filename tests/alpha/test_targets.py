from __future__ import annotations

import pandas as pd
import pytest

from taikyokuzu.alpha import forward_returns


def test_forward_returns():
    prices = pd.Series([100.0, 110.0, 121.0, 133.1])
    result = forward_returns(prices, periods=2)
    assert result.iloc[0] == pytest.approx(0.21)
    assert result.iloc[1] == pytest.approx(0.21)
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])
