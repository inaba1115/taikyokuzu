"""Calculate J-Quants adjusted daily bars from Bulk CSV columns.

Reference: https://jpx-jquants.com/ja/spec/eq-bars-daily/adj
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

_PRICE_COLUMNS = ("O", "H", "L", "C")


def _round_half_up(values: pd.Series) -> pd.Series:
    """Round nonnegative prices/volumes to 1 decimal, at the final step only."""
    # pandas.Series.round() uses ties-to-even, unlike J-Quants' 四捨五入.
    return np.floor(values * 10 + 0.5) / 10


def _cumulative_adjustment(factor: pd.Series, code: pd.Series) -> pd.Series:
    """Multiply only factors at strictly *later* dates within each code.

    Inputs must be ordered by Code ascending and Date descending.
    For each code, the newest row's factor is 1; an ex-date's AdjFactor
    first affects the *next older* row, not the ex-date itself.
    """
    previous_factor = factor.groupby(code, sort=False).shift(1, fill_value=1.0)
    return previous_factor.groupby(code, sort=False).cumprod()


def adjust_prices(
    raw_df: pd.DataFrame,
    *,
    as_of: str | date | None = None,
) -> pd.DataFrame:
    """Reproduce J-Quants AdjO/AdjH/AdjL/AdjC/AdjVo from Bulk data.

    Follows the steps in the official specification:
      1. Sort each Code by Date descending.
      2. CumAdj = product of AdjFactor on strictly newer dates.
      3. AdjO/H/L/C = round_half_up(raw price * CumAdj, 1 decimal).
      4. AdjVo = round_half_up(Vo / CumAdjVo, 1 decimal), where
         CumAdjVo excludes AdjFactor on ExRT == 3 (rights issues).

    Intermediate columns CumAdj (and CumAdjVo, if Vo exists) are
    retained for verification. Results are sorted by Code, Date ascending.
    Input is not modified. Duplicate (Code, Date) rows are rejected.

    ``as_of`` excludes later ex-dates; it does not by itself guarantee a
    full point-in-time snapshot if the source data was revised later.
    Accurate adjustment requires all relevant ex-dates through ``as_of``.
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

    # Official step 2: make newer observations appear *before* older ones.
    df = df.sort_values(["Code", "Date"], ascending=[True, False]).reset_index(
        drop=True
    )

    factor = pd.to_numeric(df["AdjFactor"], errors="raise").astype(float)
    if factor.isna().any() or (~np.isfinite(factor)).any() or (factor <= 0).any():
        raise ValueError("AdjFactor must be positive and finite for every row")

    # Official step 3: an ex-date factor applies only to older prices.
    df["CumAdj"] = _cumulative_adjustment(factor, df["Code"])

    # Official step 4: multiply first; round the FINAL result to 1 decimal.
    for col in prices:
        value = pd.to_numeric(df[col], errors="raise")
        df[f"Adj{col}"] = _round_half_up(value * df["CumAdj"])

    if has_volume:
        # Rights issues (ExRT == 3) affect prices, but *not* share volume.
        rights_issue = pd.to_numeric(df["ExRT"], errors="coerce").eq(3).fillna(False)
        volume_factor = factor.where(~rights_issue, 1.0)
        df["CumAdjVo"] = _cumulative_adjustment(volume_factor, df["Code"])
        volume = pd.to_numeric(df["Vo"], errors="raise")
        df["AdjVo"] = _round_half_up(volume / df["CumAdjVo"])

    # Return in chronological order for indicators such as EMA/MACD.
    return df.sort_values(["Code", "Date"]).reset_index(drop=True)
