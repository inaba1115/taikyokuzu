from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from taikyokuzu.features import MACDResult


def draw_indicator(
    values: pd.Series,
    *,
    ax: Axes,
    label: str | None = None,
    linewidth: float = 1.2,
    linestyle: str = "-",
) -> None:
    """
    計算済みの指標系列をAxesへ描画する。

    Parameters
    ----------
    values:
        描画する指標系列。

    ax:
        描画対象のAxes。

    label:
        凡例名。
        Noneの場合はSeries.nameを使用する。

    linewidth:
        線の太さ。

    linestyle:
        線のスタイル。
    """
    data = _to_float_array(values)

    if label is None:
        label = str(values.name) if values.name is not None else "Indicator"

    ax.plot(
        np.arange(len(data)),
        data,
        label=label,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def draw_moving_averate(
    values: pd.Series,
    *,
    ax: Axes,
    label: str | None = None,
    linewidth: float = 1.2,
) -> None:
    """
    計算済みの移動平均系列をAxesへ描画する。

    SMAやEMAの計算自体はfeaturesパッケージで行う。
    """
    draw_indicator(
        values,
        ax=ax,
        label=label,
        linewidth=linewidth,
    )


def draw_macd(
    result: MACDResult,
    *,
    ax: Axes,
    histogram_width: float = 0.8,
    histogram_alpha: float = 0.4,
) -> None:
    """
    MACDを指定されたAxesへ描画する。

    Parameters
    ----------
    result:
        taikyokuzu.features.macd() の計算結果。

    ax:
        描画対象のAxes。

    histogram_width:
        Histogramの棒の幅。

    histogram_alpha:
        Histogramの透明度。
    """
    macd_values = _to_float_array(result.macd)
    signal_values = _to_float_array(result.signal)
    histogram_values = _to_float_array(result.histogram)

    lengths = {
        len(macd_values),
        len(signal_values),
        len(histogram_values),
    }

    if len(lengths) != 1:
        raise ValueError("MACD series must have the same length")

    x = np.arange(len(macd_values))

    ax.plot(x, macd_values, label="MACD", linewidth=1.2)

    ax.plot(x, signal_values, label="Signal", linewidth=1.2)

    ax.bar(
        x,
        histogram_values,
        width=histogram_width,
        alpha=histogram_alpha,
        label="Histogram",
    )

    ax.axhline(0.0, linewidth=0.0)

    ax.set_ylabel("MACD")

    ax.grid(axis="y", alpha=0.3)

    ax.legend()


def _to_float_array(values: pd.Series) -> np.ndarray:
    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series")

    if values.empty:
        raise ValueError("values must not be empty")

    converted = pd.to_numeric(values, errors="raise")

    return converted.to_numpy(dtype=float)
