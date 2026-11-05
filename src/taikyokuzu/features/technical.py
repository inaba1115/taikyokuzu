from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TechnicalData = pd.Series | pd.DataFrame


@dataclass(frozen=True)
class MACDResult:
    """Moving Average Convergence Divergenceの計算結果。"""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


def ema(values: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (EMA) を計算する。"""
    _validate_series(values)
    _validate_positive_int(span, name="span")

    result = values.astype(float).ewm(span=span, adjust=False).mean()
    result.name = f"EMA{span}"

    return result


def macd(
    values: pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> MACDResult:
    """
    Moving Average Convergence Divergence (MACD) を計算する。

    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    _validate_positive_int(fast, name="fast")
    _validate_positive_int(slow, name="slow")
    _validate_positive_int(signal, name="signal")

    if fast >= slow:
        raise ValueError("fast must be smaller than slow")

    fast_ema = ema(values, span=fast)
    slow_ema = ema(values, span=slow)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, span=signal)
    histogram = macd_line - signal_line

    macd_line.name = "MACD"
    signal_line.name = "Signal"
    histogram.name = "Histogram"

    return MACDResult(
        macd=macd_line,
        signal=signal_line,
        histogram=histogram,
    )


def _validate_series(values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series")

    if values.empty:
        raise ValueError("values must not be empty")

    converted = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & converted.isna()
    if invalid.any():
        raise ValueError("values must contain only numeric values")


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")
