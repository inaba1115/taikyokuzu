from __future__ import annotations

import numpy as np
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


def min_forward_returns(
    prices: pd.Series,
    *,
    periods: int = 3,
) -> pd.Series:
    """
    ``periods`` 期間先までのforward returnの最小値を計算する。

    Parameters
    ----------
    prices:
        時系列順に並んだ価格Series。

    periods:
        将来リターンを評価する期間数。

    Returns
    -------
    pd.Series
        各時点から将来``periods``期間までのforward returnの最小値。
        必要な将来データが揃わない場合はNaNとする。

    Notes
    -----
    現在価格に対する将来期間中の最小リターンを表す。
    下値の安定性を学習するtargetとして利用する。
    """
    _validate_positive_int(periods, name="periods")

    series_list = [forward_returns(prices, periods=p) for p in range(1, periods + 1)]

    result = pd.concat(series_list, axis=1).min(axis=1, skipna=False)
    result.name = "MinForwardReturn"

    return result


def stable_forward_returns(
    prices: pd.Series,
    *,
    periods: int = 3,
    penalty: float = 0.5,
) -> pd.Series:
    """将来の最低リターンから短期的な反落を減点したTargetを計算する。

    Parameters
    ----------
    prices:
        時系列順に並んだ価格Series。

    periods:
        将来リターンを評価する期間数。

    penalty:
        翌日リターンからの悪化に対するペナルティ係数。

    Returns
    -------
    pd.Series
        反落ペナルティを考慮したTarget。
        必要な将来データが不足する場合はNaN。
    """
    _validate_positive_int(periods, name="periods")

    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative")

    r1 = forward_returns(prices, periods=1)
    rmin = min_forward_returns(prices, periods=periods)

    result = rmin - penalty * (r1 - rmin)
    result.name = "StableForwardReturn"

    return result


def _validate_positive_int(
    value: int,
    *,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")
