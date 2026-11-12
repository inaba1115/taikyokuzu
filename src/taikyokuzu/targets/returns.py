from __future__ import annotations

import pandas as pd


def forward_returns(
    prices: pd.Series,
    *,
    periods: int = 5,
) -> pd.Series:
    """
    現在価格から ``periods`` 期間先までの forward return を計算する。

    Notes
    -----
    これは alpha feature ではなく、screening / model evaluation 用の
    将来ラベルである。予測時点の feature として使用しないこと。
    """
    _validate_positive_int(periods, name="periods")
    return prices.shift(-periods) / prices - 1.0


def _validate_positive_int(
    value: int,
    *,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")
