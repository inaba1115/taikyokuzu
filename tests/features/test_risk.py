from typing import cast

import numpy as np
import pandas as pd
import pytest

from taikyokuzu.features.risk import drawdown


def test_drawdown_tracks_running_peak() -> None:
    index = pd.date_range("2026-01-05", periods=5, freq="B")
    prices = pd.Series([100.0, 120.0, 90.0, 108.0, 120.0], index=index)

    result = drawdown(prices)
    expected = pd.Series(
        [0.0, 0.0, -0.25, -0.10, 0.0],
        index=index,
        name="Drawdown",
    )

    pd.testing.assert_series_equal(result, expected)


def test_drawdown_is_zero_for_rising_prices() -> None:
    prices = pd.Series([100.0, 105.0, 110.0])

    result = drawdown(prices)

    assert (result == 0.0).all()


def test_drawdown_preserves_peak_across_nan() -> None:
    prices = pd.Series([100.0, 120.0, np.nan, 90.0, 126.0])

    result = drawdown(prices)
    expected = pd.Series(
        [0.0, 0.0, np.nan, -0.25, 0.0],
        name="Drawdown",
    )

    pd.testing.assert_series_equal(result, expected)


def test_drawdown_matches_return_based_drawdown() -> None:
    prices = pd.Series([100.0, 120.0, 90.0, 108.0, 120.0])
    returns = prices.pct_change()
    returns.iloc[0] = 0.0

    equity = (1.0 + returns).cumprod()
    expected = (equity / equity.cummax() - 1.0).rename("Drawdown")

    pd.testing.assert_series_equal(drawdown(prices), expected)


def test_drawdown_can_be_applied_per_code() -> None:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-05", "2026-01-06"]), ["A", "B"]],
        names=["Date", "Code"],
    )
    prices = pd.Series([100.0, 200.0, 90.0, 220.0], index=index)

    result = (
        prices.groupby(level="Code", group_keys=False)
        .apply(drawdown)
        .rename("Drawdown")
    )
    expected = pd.Series([0.0, 0.0, -0.1, 0.0], index=index, name="Drawdown")

    pd.testing.assert_series_equal(result.sort_index(), expected)


def test_drawdown_requires_series() -> None:
    with pytest.raises(TypeError, match="prices must be a pandas Series"):
        drawdown(cast(pd.Series, [100.0, 90.0]))


def test_drawdown_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="prices must not be empty"):
        drawdown(pd.Series(dtype=float))


def test_drawdown_rejects_non_numeric_prices() -> None:
    with pytest.raises(TypeError, match="prices must contain numeric values"):
        drawdown(pd.Series([100.0, "invalid"]))


@pytest.mark.parametrize("invalid_price", [0.0, -1.0])
def test_drawdown_rejects_non_positive_prices(invalid_price: float) -> None:
    with pytest.raises(ValueError, match="prices must be positive"):
        drawdown(pd.Series([100.0, invalid_price]))


@pytest.mark.parametrize("invalid_price", [np.inf, -np.inf])
def test_drawdown_rejects_infinite_prices(invalid_price: float) -> None:
    with pytest.raises(ValueError, match="only finite values or NaN"):
        drawdown(pd.Series([100.0, invalid_price]))
