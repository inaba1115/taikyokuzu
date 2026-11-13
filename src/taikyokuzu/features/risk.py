from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(
    prices: pd.Series,
) -> pd.Series:
    """
    価格系列からドローダウン系列を計算する。

    Parameters
    ----------
    prices : pd.Series
        時系列順に並んだ価格系列。NaNを含んでもよい。

    Returns
    -------
    pd.Series
        過去最高値からの下落率。値は0以下となる。
        元のindexを保持し、価格がNaNの箇所はNaNとする。

    Notes
    -----
    各時点のドローダウンは ``prices / prices.cummax() - 1`` で計算する。
    NaNの箇所では結果をNaNとし、過去最高値は維持する。
    """
    data = _validation_prices(prices)

    result = data / data.cummax() - 1.0
    result.name = "Drawdown"
    return result


def _validation_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series")

    if prices.empty:
        raise ValueError("prices must not be empty")

    try:
        data = pd.to_numeric(prices, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("prices must contain numeric values") from exc

    values = data.dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("prices must contain only finite values or NaN")

    if (values <= 0).any():
        raise ValueError("prices must be positive")

    return data
