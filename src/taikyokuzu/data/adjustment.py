"""J-Quants の Bulk データから調整済み株価・出来高を計算するモジュール。

References
----------
J-Quants 公式仕様：「調整済み株価の計算方法」
https://jpx-jquants.com/ja/spec/eq-bars-daily/adj
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

_PRICE_COLUMNS = ("O", "H", "L", "C")


def _round_half_up(values: pd.Series) -> pd.Series:
    """非負の数値を小数第 1 位に四捨五入する。

    Parameters
    ----------
    values : pandas.Series
        丸める前の価格または出来高。非負の値を想定する。

    Returns
    -------
    pandas.Series
        小数第 1 位に四捨五入した値。

    Notes
    -----
    ``pandas.Series.round`` の偶数丸めではなく、J-Quants の
    四捨五入に合わせた計算を行う。累積係数の計算途中では丸めない。
    """
    return np.floor(values * 10 + 0.5) / 10


def _cumulative_adjustment(factor: pd.Series, code: pd.Series) -> pd.Series:
    """銘柄ごとに、対象日より新しい日付の調整係数を累積する。

    Parameters
    ----------
    factor : pandas.Series
        各行に対応する調整係数。
    code : pandas.Series
        各行の銘柄コード。

    Returns
    -------
    pandas.Series
        対象日より後の調整係数を掛け合わせた累積調整係数。

    Notes
    -----
    入力行は ``Code`` の昇順、``Date`` の降順で並んでいることを前提とする。
    権利落ち日の調整係数は当日の価格には適用せず、それより過去の日付に適用する。
    このため、銘柄ごとに調整係数を 1 行ずらしてから累積積を計算する。
    """
    previous_factor = factor.groupby(code, sort=False).shift(1, fill_value=1.0)
    return previous_factor.groupby(code, sort=False).cumprod()


def adjust_prices(
    raw_df: pd.DataFrame,
    *,
    as_of: str | date | None = None,
) -> pd.DataFrame:
    """Bulk データから J-Quants 方式の調整済み株価・出来高を計算する。

    Parameters
    ----------
    raw_df : pandas.DataFrame
        ``Code``、``Date``、``AdjFactor`` を含む日足データ。
        ``O``、``H``、``L``、``C``、``Vo`` のうち少なくとも 1 列が必要。
        ``Vo`` が存在する場合は ``ExRT`` も必要。
    as_of : str or datetime.date or None, optional
        調整計算の基準日。この日付より後のデータを除外する。
        ``None`` の場合は入力データに含まれる全期間を使用する。

    Returns
    -------
    pandas.DataFrame
        元データに累積調整係数 ``CumAdj`` と、存在する価格列に応じた
        ``AdjO``、``AdjH``、``AdjL``、``AdjC`` を追加したデータ。
        ``Vo`` が存在する場合は ``CumAdjVo`` と ``AdjVo`` も追加する。
        ``Code``、``Date`` の昇順に並ぶ。入力データは変更しない。

    Raises
    ------
    ValueError
        必須列が不足する場合、同じ銘柄・日付が重複する場合、
        ``Code`` または ``Date`` が欠損する場合、
        ``AdjFactor`` が正の有限値でない場合など。

    Notes
    -----
    J-Quants の公式仕様に従い、銘柄ごとに日付降順へ並べ、
    対象日より新しい日付の ``AdjFactor`` の積を ``CumAdj`` とする。
    価格には ``CumAdj`` を掛け、出来高は ``CumAdjVo`` で割る。
    出来高の計算ではライツイシュー（``ExRT == 3``）の係数を除外する。
    最終結果のみを小数第 1 位に四捨五入する。

    ``as_of`` を指定しても、取得後に訂正された過去データを元に戻すことはできない。
    また、正確な調整には基準日までの該当する権利落ち日データが必要となる。
    """
    required = {"Code", "Date", "AdjFactor"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    prices = [col for col in _PRICE_COLUMNS if col in raw_df.columns]
    has_volume = "Vo" in raw_df.columns
    if not prices and not has_volume:
        raise ValueError("At least one of O/H/L/C/Vo is required")
    if has_volume and "ExRT" not in raw_df.columns and not raw_df.empty:
        raise ValueError("ExRT is required when adjusting Vo")

    df = raw_df.copy()
    df["Code"] = df["Code"].astype("string")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize()
        if pd.isna(cutoff):
            raise ValueError("as_of must be a valid date")
        df = df.loc[df["Date"] <= cutoff].copy()

    if df.empty:
        df["CumAdj"] = pd.Series(dtype="float64")
        for col in prices:
            df[f"Adj{col}"] = pd.Series(dtype="float64")
        if has_volume:
            df["CumAdjVo"] = pd.Series(dtype="float64")
            df["AdjVo"] = pd.Series(dtype="float64")
        return df

    if df[["Code", "Date"]].isna().any().any():
        raise ValueError("Code and Date must not be null")
    if df.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate (Code, Date) records")

    # 新しい日付から順に、過去へ向かって調整係数を累積できる並びにする。
    df = df.sort_values(["Code", "Date"], ascending=[True, False]).reset_index(
        drop=True
    )

    factor = pd.to_numeric(df["AdjFactor"], errors="raise").astype(float)
    if factor.isna().any() or (~np.isfinite(factor)).any() or (factor <= 0).any():
        raise ValueError("AdjFactor must be positive and finite for every row")

    # 権利落ち日の係数は、その日より古い株価のみに適用する。
    df["CumAdj"] = _cumulative_adjustment(factor, df["Code"])

    # 価格には累積調整係数を掛け、最後に小数第 1 位へ丸める。
    for col in prices:
        value = pd.to_numeric(df[col], errors="raise")
        df[f"Adj{col}"] = _round_half_up(value * df["CumAdj"])

    if has_volume:
        # ライツイシューでは価格のみを調整し、出来高の係数からは除外する。
        rights_issue = pd.to_numeric(df["ExRT"], errors="coerce").eq(3).fillna(False)
        volume_factor = factor.where(~rights_issue, 1.0)
        df["CumAdjVo"] = _cumulative_adjustment(volume_factor, df["Code"])
        volume = pd.to_numeric(df["Vo"], errors="raise")
        df["AdjVo"] = _round_half_up(volume / df["CumAdjVo"])

    # EMA や MACD を計算しやすいよう、日付の昇順に戻す。
    return df.sort_values(["Code", "Date"]).reset_index(drop=True)
