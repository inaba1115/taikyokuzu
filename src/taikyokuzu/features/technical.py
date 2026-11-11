from __future__ import annotations

import numpy as np
import pandas as pd


def sma(
    values: pd.Series,
    window: int,
) -> pd.Series:
    """Simple Moving Average (SMA) を計算する。"""
    _validate_series(values)
    _validate_positive_int(window, name="window")

    result = (
        values.astype(float)
        .rolling(
            window=window,
            min_periods=window,
        )
        .mean()
    )
    result.name = f"SMA{window}"
    return result


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
) -> pd.DataFrame:
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

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "MACDSignal": signal_line,
            "MACDHistogram": histogram,
        }
    )


def rsi(
    values: pd.Series,
    *,
    window: int = 14,
) -> pd.Series:
    """
    Wilder方式のRelative Strength Index (RSI)を計算する。

    Parameters
    ----------
    values:
        価格系列。

    window:
        Wilder平均に使用する期間数。デフォルトは14。

    Returns
    -------
    Series
        0から100の範囲のRSI。
    """
    _validate_positive_int(window, name="window")
    values = _prepare_rsi_series(values)

    deltas = values.diff().to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)

    seed_gains: list[float] = []
    seed_losses: list[float] = []
    average_gain: float | None = None
    average_loss: float | None = None

    for position in range(1, len(values)):
        delta = deltas[position]

        if np.isnan(delta):
            seed_gains.clear()
            seed_losses.clear()
            average_gain = None
            average_loss = None
            continue

        gain = max(float(delta), 0.0)
        loss = max(float(-delta), 0.0)

        if average_gain is None or average_loss is None:
            seed_gains.append(gain)
            seed_losses.append(loss)

            if len(seed_gains) < window:
                continue

            average_gain = float(np.mean(seed_gains))
            average_loss = float(np.mean(seed_losses))
        else:
            average_gain = ((window - 1) * average_gain + gain) / window
            average_loss = ((window - 1) * average_loss + loss) / window

        result[position] = _rsi_value(
            average_gain=average_gain, average_loss=average_loss
        )

    return pd.Series(
        result,
        index=values.index,
        name=f"RSI{window}",
        dtype=float,
    )


def _rsi_value(
    *,
    average_gain: float,
    average_loss: float,
) -> float:
    if average_gain == 0.0 and average_loss == 0.0:
        return 50.0

    if average_loss == 0.0:
        return 100.0

    if average_gain == 0.0:
        return 0.0

    relative_strength = average_gain / average_loss
    return float(100.0 - 100.0 / (1.0 + relative_strength))


def _prepare_rsi_series(values: pd.Series) -> pd.Series:
    _validate_series(values)
    prepared = pd.to_numeric(values, errors="raise").astype(float)

    finite = np.isfinite(prepared.dropna().to_numpy(dtype=float))
    if not finite.all():
        raise ValueError("values must contain only finite values or NaN")

    return prepared


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
