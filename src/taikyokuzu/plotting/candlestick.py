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
    """
    ローソク足チャートを新しいFigureへ描画する。

    Parameters
    ----------
    df:
        J-Quants形式の日足DataFrame。

    code:
        描画対象の銘柄コード。

        dfに複数銘柄が含まれる場合は必須。

    adjusted:
        Trueの場合は以下の調整済み価格を使用する。

            AdjO, AdjH, AdjL, AdjC

        Falseの場合は以下の未調整価格を使用する。

            O, H, L, C

    figsize:
        Figureサイズ。

    candle_width:
        ローソク足の実体部分の幅。

    up_color:
        陽線の色。

    down_color:
        陰線の色。

    Returns
    -------
    tuple[Figure, Axes]
        作成したFigureとAxes。
    """
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
    """
    指定されたAxesへローソク足を描画する。

    Parameters
    ----------
    df:
        J-Quants形式の日足DataFrame。

    ax:
        描画対象のAxes。

    code:
        描画対象の銘柄コード。

        dfに複数銘柄が含まれる場合は必須。

    adjusted:
        Trueの場合はAdjO / AdjH / AdjL / AdjCを使用する。

        Falseの場合はO / H / L / Cを使用する。

    candle_width:
        ローソク足の実体部分の幅。

    up_color:
        陽線の色。

    down_color:
        陰線の色。

    show_dates:
        Trueの場合はx軸に日付を表示する。

    max_date_ticks:
        表示する日付tickの最大値。
    """
    if candle_width <= 0.0:
        raise ValueError("candle_width must be positive")

    if max_date_ticks <= 0:
        raise ValueError("max_date_tickes must be positive")

    data = _prepare_candlestick_data(df, code=code, adjusted=adjusted)

    (open_column, high_column, low_column, close_column) = get_ohlc_columns(
        adjusted=adjusted
    )

    open_values = data[open_column].to_numpy(dtype=float)
    high_values = data[high_column].to_numpy(dtype=float)
    low_values = data[low_column].to_numpy(dtype=float)
    close_values = data[close_column].to_numpy(dtype=float)

    x = np.arange(len(data))

    minimum_body_height = np.finfo(float).eps

    for (
        position,
        open_,
        high,
        low,
        close,
    ) in zip(
        x,
        open_values,
        high_values,
        low_values,
        close_values,
        strict=True,
    ):
        color = up_color if close >= open_ else down_color

        ax.vlines(position, low, high, color=color, linewidth=1.0)

        body_bottom = min(open_, close)

        body_height = abs(close - open_)

        if body_height <= minimum_body_height:
            ax.hlines(
                open_,
                position - candle_width / 2.0,
                position + candle_width / 2.0,
                color=color,
                linewidth=1.0,
            )

            continue

        rectangle = Rectangle(
            (
                position - candle_width / 2.0,
                body_bottom,
            ),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color,
        )

        ax.add_patch(rectangle)

    actual_code = str(data["Code"].iloc[0])

    price_label = "Adjusted Price" if adjusted else "Price"

    ax.set_title(f"{actual_code} Candlestick")

    ax.set_ylabel(price_label)

    ax.grid(axis="y", alpha=0.3)

    ax.set_xlim(-1.0, float(len(data)))

    if show_dates:
        set_date_axis(
            data["Date"],
            ax=ax,
            max_ticks=max_date_ticks,
        )


def set_date_axis(
    dates: pd.Series | pd.Index,
    *,
    ax: Axes,
    max_ticks: int = 10,
) -> None:
    """
    整数x座標に対応する日付ラベルをAxesへ設定する。

    休日をx軸上の空白として表示しないため、
    ローソク足では実際の日付ではなく整数位置を使用する。
    """
    if max_ticks <= 0:
        raise ValueError("max_ticks must be positive")

    date_index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))

    if len(date_index) == 0:
        return

    tick_count = min(max_ticks, len(date_index))

    positions = np.linspace(
        0,
        len(date_index) - 1,
        num=tick_count,
        dtype=int,
    )

    positions = np.unique(positions)

    labels = [date_index[position].strftime("%Y-%m-%d") for position in positions]

    ax.set_xticks(positions)

    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("Date")


def get_ohlc_columns(
    *,
    adjusted: bool,
) -> tuple[str, str, str, str]:
    if adjusted:
        return ("AdjO", "AdjH", "AdjL", "AdjC")

    return ("O", "H", "L", "C")


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

    (
        open_column,
        high_column,
        low_column,
        close_column,
    ) = get_ohlc_columns(adjusted=adjusted)

    required_columns = {
        "Date",
        "Code",
        open_column,
        high_column,
        low_column,
        close_column,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="raise",
    )

    data["Code"] = data["Code"].astype(str)

    for column in (
        open_column,
        high_column,
        low_column,
        close_column,
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        ).astype(float)

    if code is not None:
        normalize_code = _normalize_code(code)

        data = data[data["Code"] == normalize_code]

    codes = data["Code"].unique()

    if len(codes) == 0:
        raise ValueError("no data to plot")

    if len(codes) > 1:
        raise ValueError("df contains multiple codes. Specify code=...")

    values = data[
        [
            open_column,
            high_column,
            low_column,
            close_column,
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("OHLC values must contain only finite values")

    open_values = values[:, 0]
    high_values = values[:, 1]
    low_values = values[:, 2]
    close_values = values[:, 3]

    if (high_values < np.maximum(open_values, close_values)).any():
        raise ValueError("high must be greater than or equal to open and close")

    if (low_values > np.minimum(open_values, close_values)).any():
        raise ValueError("low must be less than or equal to open and close")

    if (low_values > high_values).any():
        raise ValueError("low must not exceed high")

    return data.sort_values("Date").reset_index(drop=True)


def _normalize_code(code: str) -> str:
    code = str(code)

    if not code.isdigit():
        raise ValueError(f"code must be numeric: {code!r}")

    if len(code) == 4:
        return f"{code}0"

    if len(code) == 5:
        return code

    raise ValueError(f"code must be 4 or 5 digits: {code!r}")
