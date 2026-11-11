import pandas as pd
import pytest

from taikyokuzu.features import ema, macd, rsi


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


def test_rsi_matches_wilder_smoothing():
    values = pd.Series(
        [
            54.80,
            56.80,
            57.85,
            59.85,
            60.57,
            61.10,
            62.17,
            60.60,
            62.35,
            62.15,
            62.35,
            61.45,
            62.80,
            61.37,
            62.50,
            62.57,
        ]
    )

    result = rsi(values, window=14)

    assert result.iloc[:14].isna().all()
    assert result.iloc[14] == pytest.approx(74.21383647798743)
    assert result.iloc[15] == pytest.approx(74.33551617873653)
    assert result.name == "RSI14"


def test_rsi_handles_rising_falling_and_flat_series():
    values = pd.DataFrame(
        {
            "Rising": [float(value) for value in range(1, 21)],
            "Falling": [float(value) for value in range(20, 0, -1)],
            "Flat": [100.0] * 20,
        }
    )

    result_rasing = rsi(values["Rising"], window=5)
    result_falling = rsi(values["Falling"], window=5)
    result_flat = rsi(values["Flat"], window=5)

    assert result_rasing.iloc[-1] == pytest.approx(100.0)
    assert result_falling.iloc[-1] == pytest.approx(0.0)
    assert result_flat.iloc[-1] == pytest.approx(50.0)

    assert result_rasing.name == "RSI5"
    assert result_falling.name == "RSI5"
    assert result_flat.name == "RSI5"


def test_rsi_resets_after_missing_price():
    values = pd.Series(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            float("nan"),
            110.0,
            111.0,
            112.0,
            113.0,
        ]
    )

    result = rsi(values, window=3)

    assert result.iloc[5:9].isna().all()
    assert result.iloc[9] == pytest.approx(100.0)
