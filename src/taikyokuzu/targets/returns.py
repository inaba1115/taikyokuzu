from __future__ import annotations

import pandas as pd


def forward_returns(
    prices: pd.Series,
    *,
    periods: int = 5,
) -> pd.Series:
    """
    現在価格から ``periods`` 期間先までのforward returnを計算する。

    Parameters
    ----------
    prices:
        時系列順に並んだ価格Series。

    periods:
        将来リターンを計算する期間数。デフォルトは5。

    Returns
    -------
    pd.Series
        各時点から ``periods`` 期間先までの単純リターン。
        必要な将来データが存在しない場合はNaNとする。
        入力と同じindexを保持する。

    Notes
    -----
    これはalpha featureではなく、screening / model evaluation用の
    将来ラベルである。予測時点のfeatureとして使用しないこと。
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
