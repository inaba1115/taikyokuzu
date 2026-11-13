from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taikyokuzu.targets import (
    forward_returns,
    min_forward_returns,
    stable_forward_returns,
)


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


def test_stable_forward_returns_for_steady_rise() -> None:
    prices = pd.Series([100.0, 102.0, 104.0, 106.0])

    result = stable_forward_returns(prices, periods=3, penalty=0.5)

    assert result.iloc[0] == pytest.approx(0.02)
    assert result.iloc[1:].isna().all()
    assert result.name == "StableForwardReturn"


def test_stable_forward_returns_penalizes_reversal() -> None:
    prices = pd.Series([100.0, 110.0, 105.0, 103.0])

    result = stable_forward_returns(prices, periods=3, penalty=0.5)

    # 3日間の最低リターンは+3%、翌日のリターンは+10%。
    assert result.iloc[0] == pytest.approx(0.03 - 0.5 * (0.10 - 0.03))


def test_stable_forward_returns_for_dip_and_recovery() -> None:
    prices = pd.Series([100.0, 98.0, 102.0, 105.0])

    result = stable_forward_returns(prices, periods=3, penalty=0.5)

    assert result.iloc[0] == pytest.approx(-0.02)


def test_stable_forward_returns_with_zero_penalty() -> None:
    prices = pd.Series([100.0, 110.0, 105.0, 103.0, 108.0])

    result = stable_forward_returns(prices, periods=3, penalty=0.0)
    expected = min_forward_returns(prices, periods=3).rename("StableForwardReturn")

    pd.testing.assert_series_equal(result, expected)


def test_stable_forward_returns_with_full_penalty() -> None:
    prices = pd.Series([100.0, 110.0, 105.0, 103.0, 108.0])

    result = stable_forward_returns(prices, periods=3, penalty=1.0)
    r1 = forward_returns(prices, periods=1)
    rmin = min_forward_returns(prices, periods=3)
    expected = (2 * rmin - r1).rename("StableForwardReturn")

    pd.testing.assert_series_equal(result, expected)


def test_stable_forward_returns_preserves_index() -> None:
    index = pd.date_range("2026-01-05", periods=5, freq="B", name="Date")
    prices = pd.Series([100.0, 102.0, 101.0, 105.0, 107.0], index=index)

    result = stable_forward_returns(prices, periods=3)

    assert result.index.equals(prices.index)
    assert result.name == "StableForwardReturn"


def test_stable_forward_returns_requires_complete_future_data() -> None:
    prices = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, 105.0])

    result = stable_forward_returns(prices, periods=3)

    assert pd.isna(result.iloc[0])  # 2日先が欠損
    assert result.iloc[-3:].isna().all()  # 3日先まで確定していない


@pytest.mark.parametrize("periods", [0, -1])
def test_stable_forward_returns_rejects_non_positive_periods(periods: int) -> None:
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])

    with pytest.raises(ValueError, match="periods must be positive"):
        stable_forward_returns(prices, periods=periods)


@pytest.mark.parametrize("penalty", [-0.5, np.nan, np.inf, -np.inf])
def test_stable_forward_returns_rejects_invalid_penalty(penalty: float) -> None:
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])

    with pytest.raises(ValueError, match="penalty must be finite and non-negative"):
        stable_forward_returns(prices, penalty=penalty)
