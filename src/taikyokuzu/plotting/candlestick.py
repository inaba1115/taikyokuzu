from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


def plot_candlestick(
    df: pd.DataFrame,
    *,
    code: str | None = None,
    adjusted: bool = False,
    figsize: tuple[float, float] = (12, 6),
    candle_width: float = 0.6,
    up_color: str = "tab:red",
    down_color: str = "tab:blue",
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(
        figsize=figsize,
    )

    draw_candlestick(
        df,
        ax=ax,
        code=code,
        adjusted=adjusted,
        candle_width=candle_width,
        up_color=up_color,
        down_color=down_color,
        show_dates=True,
    )

    fig.tight_layout()

    return fig, ax


def draw_candlestick(
    df: pd.DataFrame,
    *,
    ax: Axes,
    code: str | None = None,
    adjusted: bool = False,
    candle_width: float = 0.6,
    up_color: str = "tab:red",
    down_color: str = "tab:blue",
    show_dates: bool = True,
    max_date_ticks: int = 10,
) -> None:
    pass

def _prepare_candlestick_data(
    df: pd.DataFrame,
    *,
    code: str | None,
    adjusted: bool,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if df.empty:
        raise ValueError("df must not be empty")


def get_ohlc_columns(
    *,
    adjusted: bool,
) -> tuple[str, str, str, str]:
    if adjusted:
        return ["AdjO", "AdjH", "AdjL", "AdjC"]

    return ["O", "H", "L", "C"]
    


def _normalize_code(
    code: str
) -> str:
    code = str(code)

    if not code.isdigit():
        raise ValueError(f"code must be numeric: {code!r}")

    if len(code) == 4:
        return f"{code}0"

    if len(code) == 5:
        return code

    raise ValueError(f"code must be 4 or 5 digits: {code!r}")
