"""Reconstruct adjusted bars using J-Quants AdjFactor."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _round_half_up_nonnegative(values: pd.Series) -> pd.Series:
    """Round nonnegative floats to 1 dp (half up), preserving NaNs."""
    return np.floor(values * 10 + 0.5) / 10


def adjust_prices(
    raw_df: pd.DataFrame,
    *,
    as_of: str | date | None = None,
) -> pd.DataFrame:
    """Compute AdjO/AdjH/AdjL/AdjC/AdjVo for each Code.

    AdjFactor for a given ex-date applies *only to older observations*.
    Only factors on/before as_of are used, avoiding future corporate actions
    when evaluating a historical date. Raw input is not modified.

    The data must include all ex-rights dates in the requested interval; an
    incomplete per-symbol slice cannot correctly reconstruct adjusted prices.
    """
    required = {"Code", "Date", "AdjFactor"}
    missing = required.difference(raw_df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    available = [c for c in ("O", "H", "L", "C", "Vo") if c in raw_df.columns]
    if not available:
        raise ValueError("At least one of O/H/L/C/Vo is required")

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype("string")
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize()
        df = df.loc[df["Date"] <= cutoff].copy()

    if df.empty:
        return df.assign(**{f"Adj{c}": pd.Series(dtype="float64") for c in available})

    if df.duplicated(["Code", "Date"]).any():
        raise ValueError("Duplicate (Code, Date) records")
    df = df.sort_values(["Code", "Date"], ascending=[True, False]).reset_index(drop=True)

    factor = pd.to_numeric(df["AdjFactor"], errors="raise")
    if factor.isna().any() or (~np.isfinite(factor)).any() or (factor <= 0).any():
        raise ValueError("AdjFactor must be positive and finite for every row")

    def cumulative_future_adjustments(s: pd.Series) -> pd.Series:
        shifted = s.groupby(df["Code"], sort=False).shift(1).fillna(1.0)
        return shifted.groupby(df["Code"], sort=False).cumprod()

    price_factor = cumulative_future_adjustments(factor)
    for column in ("O", "H", "L", "C"):
        if column in df:
            raw = pd.to_numeric(df[column], errors="raise")
            df[f"Adj{column}"] = _round_half_up_nonnegative(raw * price_factor)

    if "Vo" in df:
        # ExRT=3 is a rights issue: adjust price, but NOT volume.
        if "ExRT" not in df:
            raise ValueError("ExRT is required when adjusting Vo")
        rights = df["ExRT"].astype("string").eq("3").fillna(False)
        volume_factors = cumulative_future_adjustments(factor.where(~rights, 1.0))
        raw_volume = pd.to_numeric(df["Vo"], errors="raise")
        df["AdjVo"] = _round_half_up_nonnegative(raw_volume / volume_factors)

    return df.sort_values(["Code", "Date"]).reset_index(drop=True)
