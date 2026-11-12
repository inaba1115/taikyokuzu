from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from taikyokuzu.alpha.lightgbm import fit_lightgbm, predict_lightgbm

LIGHTGBM_TEST_PARAMS = {
    "learning_rate": 0.2,
    "num_leaves": 5,
    "min_data_in_leaf": 1,
    "min_data_in_bin": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
    "seed": 7,
}


def _panel() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.MultiIndex.from_product(
        [
            pd.date_range("2026-01-05", periods=12, freq="B"),
            ["13010", "13020", "13030"],
        ],
        names=["Date", "Code"],
    )

    values = np.arange(len(index), dtype=float)
    X = pd.DataFrame(
        {
            "FeatureA": values / 10.0,
            "FeatureB": np.tile([0.0, 0.5, 1.0], 12),
        },
        index=index,
    )
    y = (X["FeatureA"] * 0.01 + X["FeatureB"] * 0.02).rename("Target")
    return X, y


def test_fit_and_predict_lightgbm():
    X, y = _panel()

    model = fit_lightgbm(
        X,
        y,
        params=LIGHTGBM_TEST_PARAMS,
        num_boost_round=10,
    )
    result = predict_lightgbm(model, X.iloc[-3:])

    assert isinstance(model, lgb.Booster)
    assert isinstance(result, pd.Series)
    assert result.name == "Alpha"
    assert result.index.equals(X.index[-3:])
    assert np.isfinite(result.to_numpy()).all()

    expected = np.asarray(
        model.predict(X.iloc[-3:]),
        dtype=float,
    ).reshape(-1)

    np.testing.assert_allclose(
        result.to_numpy(),
        expected,
    )


def test_fit_lightgbm_accepts_missing_features():
    X, y = _panel()
    X.loc[X.index[0], "FeatureA"] = np.nan

    model = fit_lightgbm(X, y, params=LIGHTGBM_TEST_PARAMS)
    result = predict_lightgbm(model, X.iloc[:3])

    assert np.isfinite(result.to_numpy()).all()


def test_fit_lightgbm_with_categorical_feature():
    X, y = _panel()
    X["Sector33"] = pd.Categorical(np.tile(["Bank", "IT", "Retail"], 12))

    model = fit_lightgbm(
        X,
        y,
        params=LIGHTGBM_TEST_PARAMS,
        categorical_features=["Sector33"],
    )
    result = predict_lightgbm(model, X.iloc[-3:])

    assert "Sector33" in model.feature_name()
    assert np.isfinite(result.to_numpy()).all()


def test_predict_lightgbm_with_empty_data():
    X, y = _panel()
    model = fit_lightgbm(X, y, params=LIGHTGBM_TEST_PARAMS)

    result = predict_lightgbm(model, X.iloc[:0])

    assert result.empty
    assert result.name == "Alpha"
    assert result.dtype == float
    assert result.index.equals(X.index[:0])


def test_fit_lightgbm_requires_matching_indices():
    X, y = _panel()

    with pytest.raises(ValueError, match="must have the same index"):
        fit_lightgbm(X, y.iloc[::-1])


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_fit_lightgbm_requires_finite_target(invalid_value):
    X, y = _panel()
    y.iloc[0] = invalid_value

    with pytest.raises(ValueError, match="only finite values"):
        fit_lightgbm(X, y)


def test_fit_lightgbm_requires_multiindex():
    X, y = _panel()
    X = X.reset_index(drop=True)

    with pytest.raises(ValueError, match="MultiIndex"):
        fit_lightgbm(X, y)


def test_fit_lightgbm_rejects_duplicate_index():
    X, y = _panel()
    X = pd.concat([X, X.iloc[[0]]])

    with pytest.raises(ValueError, match="index must be unique"):
        fit_lightgbm(X, y)


@pytest.mark.parametrize("num_boost_round", [0, -1, 1.5, True])
def test_fit_lightgbm_requires_positive_integer_rounds(num_boost_round):
    X, y = _panel()

    with pytest.raises((TypeError, ValueError), match="num_boost_round"):
        fit_lightgbm(X, y, num_boost_round=num_boost_round)


def test_predict_lightgbm_requires_matching_feature_order():
    X, y = _panel()
    model = fit_lightgbm(X, y, params=LIGHTGBM_TEST_PARAMS)

    with pytest.raises(ValueError, match="feature names and order"):
        predict_lightgbm(model, X[["FeatureB", "FeatureA"]])


def test_predict_lightgbm_requires_matching_feature_names():
    X, y = _panel()
    model = fit_lightgbm(X, y, params=LIGHTGBM_TEST_PARAMS)
    X_pred = X.rename(columns={"FeatureA": "OtherFeature"})

    with pytest.raises(ValueError, match="feature names and order"):
        predict_lightgbm(model, X_pred)
