from __future__ import annotations

import pandas as pd
import pytest

from taikyokuzu.data import adjust_prices


def test_official_stock_split_example():
    # Example from https://jpx-jquants.com/ja/spec/eq-bars-daily/adj
    raw = pd.DataFrame(
        {
            "Code": ["51010"] * 3,
            "Date": ["2024-01-12", "2024-01-11", "2024-01-10"],
            "C": [500.0, 480.0, 980.0],
            "Vo": [1_200_000, 2_400_000, 1_100_000],
            "AdjFactor": [1.0, 0.5, 1.0],
            "ExRT": [None, "1", None],
        }
    )
    out = adjust_prices(raw)
    # Function returns chronological (ascending) rows for MACD.
    assert out["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
    ]
    assert out["CumAdj"].tolist() == [0.5, 1.0, 1.0]
    assert out["AdjC"].tolist() == [490.0, 480.0, 500.0]
    assert out["CumAdjVo"].tolist() == [0.5, 1.0, 1.0]
    assert out["AdjVo"].tolist() == [2_200_000.0, 2_400_000.0, 1_200_000.0]
    assert "AdjC" not in raw  # Input is not modified.


def test_as_of_prevents_future_ex_date_lookahead():
    raw = pd.DataFrame(
        {
            "Code": ["51010", "51010", "51010"],
            "Date": ["2024-01-10", "2024-01-11", "2024-01-12"],
            "C": [980, 480, 500],
            "AdjFactor": [1.0, 0.5, 1.0],
        }
    )
    out = adjust_prices(raw, as_of="2024-01-10")
    assert out["CumAdj"].tolist() == [1.0]
    assert out["AdjC"].tolist() == [980.0]


@pytest.mark.parametrize("exrt", ["3", 3, 3.0])
def test_rights_issue_adjusts_price_but_not_volume(exrt):
    raw = pd.DataFrame(
        {
            "Code": ["94330", "94330"],
            "Date": ["2024-01-10", "2024-01-11"],
            "C": [1000, 600],
            "Vo": [100, 200],
            "AdjFactor": [1, 0.6],
            "ExRT": [None, exrt],
        }
    )
    out = adjust_prices(raw)
    assert out["CumAdj"].tolist() == [0.6, 1.0]
    assert out["AdjC"].tolist() == [600.0, 600.0]
    assert out["CumAdjVo"].tolist() == [1.0, 1.0]
    assert out["AdjVo"].tolist() == [100.0, 200.0]


def test_volume_excludes_only_rights_factor_when_mixed_with_split():
    raw = pd.DataFrame(
        {
            "Code": ["10000", "10000", "10000"],
            "Date": ["2024-01-10", "2024-01-11", "2024-01-12"],
            "C": [1000.0, 600.0, 300.0],
            "Vo": [100, 200, 400],
            "AdjFactor": [1.0, 0.6, 0.5],
            "ExRT": [None, "3", "1"],
        }
    )
    out = adjust_prices(raw)
    assert out["CumAdj"].tolist() == pytest.approx([0.3, 0.5, 1.0])
    assert out["CumAdjVo"].tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert out["AdjC"].tolist() == [300.0, 300.0, 300.0]
    assert out["AdjVo"].tolist() == [200.0, 400.0, 400.0]


def test_cumulative_adjustment_for_multiple_events_and_ohlc():
    raw = pd.DataFrame(
        {
            "Code": ["10000"] * 3,
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "O": [100, 60, 130],
            "H": [110, 65, 140],
            "L": [90, 55, 120],
            "C": [105, 62, 135],
            "AdjFactor": [1, 0.5, 2.0],
        }
    )
    out = adjust_prices(raw)
    assert out["CumAdj"].tolist() == [1.0, 2.0, 1.0]
    assert out["AdjO"].tolist() == [100.0, 120.0, 130.0]
    assert out["AdjH"].tolist() == [110.0, 130.0, 140.0]
    assert out["AdjL"].tolist() == [90.0, 110.0, 120.0]
    assert out["AdjC"].tolist() == [105.0, 124.0, 135.0]


def test_multiple_symbols_nan_close_and_half_up_rounding():
    raw = pd.DataFrame(
        {
            "Code": ["51010", "51010", "94330", "94330"],
            "Date": ["2024-01-10", "2024-01-11", "2024-01-10", "2024-01-11"],
            "C": [2.5, 1.0, None, 120.0],
            "AdjFactor": [1.0, 0.5, 1.0, 1.0],
        }
    )
    out = adjust_prices(raw)
    a = out[out["Code"] == "51010"]
    b = out[out["Code"] == "94330"]
    assert a["AdjC"].tolist() == [1.3, 1.0]  # 2.5 * .5 = 1.25 -> 1.3
    assert pd.isna(b.iloc[0]["AdjC"])
    assert b.iloc[1]["AdjC"] == 120.0
    assert b["CumAdj"].tolist() == [1.0, 1.0]


def test_empty_output_when_as_of_before_first_row():
    raw = pd.DataFrame(
        {
            "Code": ["51010"],
            "Date": ["2024-01-10"],
            "C": [100.0],
            "AdjFactor": [1.0],
        }
    )
    out = adjust_prices(raw, as_of="2024-01-09")
    assert out.empty
    assert {"CumAdj", "AdjC"}.issubset(out.columns)


@pytest.mark.parametrize("bad_factor", [0.0, -0.5, float("nan"), float("inf")])
def test_invalid_factors_raise(bad_factor):
    raw = pd.DataFrame(
        {
            "Code": ["51010"],
            "Date": ["2024-01-10"],
            "C": [100.0],
            "AdjFactor": [bad_factor],
        }
    )
    with pytest.raises(ValueError, match="AdjFactor"):
        adjust_prices(raw)


def test_volume_requires_exrt():
    raw = pd.DataFrame(
        {
            "Code": ["51010"],
            "Date": ["2024-01-10"],
            "C": [100.0],
            "Vo": [100],
            "AdjFactor": [1.0],
        }
    )
    with pytest.raises(ValueError, match="ExRT"):
        adjust_prices(raw)


def test_duplicate_code_date_raises():
    raw = pd.DataFrame(
        {
            "Code": ["51010", "51010"],
            "Date": ["2024-01-10", "2024-01-10"],
            "C": [100, 110],
            "AdjFactor": [1, 1],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        adjust_prices(raw)
