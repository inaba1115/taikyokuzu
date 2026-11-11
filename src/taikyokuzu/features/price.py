from __future__ import annotations

import pandas as pd


def rolling_volatility(
    prices: pd.Series,
    *,
    window: int = 20,
    annualization: int | None = 252,
) -> pd.Series:
    """
    単純リターンの rolling volatility を計算する。

    ``annualization=None`` の場合は期間 volatility をそのまま返す。
    """
    _validate_positive_int(window, name="window")

    if annualization is not None:
        _validate_positive_int(annualization, name="annualization")

    returns = prices.pct_change(fill_method=None)
    volatility = returns.rolling(window=window).std()

    if annualization is not None:
        volatility = volatility * annualization**0.5

    return volatility


def _validate_positive_int(
    value: int,
    *,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")
