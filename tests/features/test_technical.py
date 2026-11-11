import pandas as pd
import pytest

from taikyokuzu.features import ema, macd


def test_ema_matches_pandas_adjust_false():
    values = pd.Series([1.0, 2.0, 3.0])
    result = ema(values, span=2)
    expected = values.ewm(span=2, adjust=False).mean()
    expected.name = "EMA2"

    pd.testing.assert_series_equal(result, expected)


def test_macd_returns_consistent_components():
    values = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0])
    result = macd(values, fast=2, slow=3, signal=2)
    expected_histogram = result["MACD"] - result["MACDSignal"]
    expected_histogram.name = "MACDHistogram"

    pd.testing.assert_series_equal(result["MACDHistogram"], expected_histogram)
    assert result.columns.to_list() == ["MACD", "MACDSignal", "MACDHistogram"]


def test_macd_requires_fast_smaller_than_slow():
    values = pd.Series([100.0, 101.0])

    with pytest.raises(ValueError, match="fast must be smaller than slow"):
        macd(values, fast=3, slow=3)
