from __future__ import annotations

import pandas as pd

from taikyokuzu.data import adjust_prices


def test_adjust_factor_applies_only_to_older_rows():
    raw = pd.DataFrame({
        "Code": ["51010", "51010", "51010"],
        "Date": ["2024-01-10", "2024-01-11", "2024-01-12"],
        "C": [980.0, 480.0, 500.0],
        "Vo": [1_100_000, 2_400_000, 1_200_000],
        "AdjFactor": [1.0, 0.5, 1.0],
        "ExRT": [None, "1", None],
    })
    result = adjust_prices(raw)
    assert result["AdjC"].tolist() == [490.0, 480.0, 500.0]
    assert result["AdjVo"].tolist() == [2_200_000.0, 2_400_000.0, 1_200_000.0]
    assert "AdjC" not in raw


def test_as_of_prevents_lookahead():
    raw = pd.DataFrame({
        "Code": ["51010", "51010", "51010"],
        "Date": ["2024-01-10", "2024-01-11", "2024-01-12"],
        "C": [980, 480, 500],
        "AdjFactor": [1.0, 0.5, 1.0],
    })
    result = adjust_prices(raw, as_of="2024-01-10")
    assert result["AdjC"].tolist() == [980.0]


def test_rights_issue_applies_only_to_price():
    raw = pd.DataFrame({
        "Code": ["94330", "94330"],
        "Date": ["2024-01-10", "2024-01-11"],
        "C": [1000, 600],
        "Vo": [100, 200],
        "AdjFactor": [1, 0.6],
        "ExRT": [None, "3"],
    })
    out = adjust_prices(raw)
    assert out["AdjC"].tolist() == [600.0, 600.0]
    assert out["AdjVo"].tolist() == [100.0, 200.0]


def test_multiple_symbols_and_nan_close():
    raw = pd.DataFrame({
        "Code": ["51010", "51010", "94330", "94330"],
        "Date": ["2024-01-10", "2024-01-11", "2024-01-10", "2024-01-11"],
        "C": [2.5, 1.0, None, 120.0],
        "AdjFactor": [1.0, 0.5, 1.0, 1.0],
    })
    out = adjust_prices(raw)
    a = out[out["Code"] == "51010"]
    b = out[out["Code"] == "94330"]
    assert a["AdjC"].tolist() == [1.3, 1.0]  # 2.5 * 0.5 -> 1.25 -> 1.3
    assert pd.isna(b.iloc[0]["AdjC"])
    assert b.iloc[1]["AdjC"] == 120.0
