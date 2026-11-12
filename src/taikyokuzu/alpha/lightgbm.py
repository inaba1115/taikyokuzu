from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

_DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "seed": 42,
}


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    num_boost_round: int = 100,
    params: Mapping[str, Any] | None = None,
    categorical_features: Sequence[str] | Literal["auto"] = "auto",
) -> lgb.Booster:
    """LightGBMの回帰モデルを学習する。

    Parameters
    ----------
    X_train:
        学習用の特徴量。
        MultiIndex(Date, Code)を持つDataFrame。

    y_train:
        学習用の正解ラベル。
        X_trainと同じindexを持つSeries。

    num_boost_round:
        Boostingの反復回数。

    params:
        LightGBMの学習パラメータ。
        デフォルトパラメータを上書きする。

    categorical_features:
        カテゴリ特徴量の列名。
        "auto"の場合はpandasのcategory型を自動判定する。

    Returns
    -------
    lgb.Booster
        学習済みのLightGBMモデル。

    Notes
    -----
    学習期間やtargetの利用可能時点は呼び出し元で管理する。
    """
    _validation_training_data(X_train, y_train)
    _validation_positive_int(num_boost_round, name="num_boost_round")

    train_params = dict(_DEFAULT_PARAMS)
    if params is not None:
        train_params.update(params)

    categorical: list[str] | Literal["auto"] = "auto"
    if categorical_features != "auto":
        categorical = list(categorical_features)

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical,
    )

    return lgb.train(
        train_params,
        train_set,
        num_boost_round=num_boost_round,
    )


def predict_lightgbm(
    model: lgb.Booster,
    X_pred: pd.DataFrame,
) -> pd.Series:
    """学習済みLightGBMモデルでAlpha値を予測する。

    Parameters
    ----------
    model:
        学習済みのLightGBMモデル。

    X_pred:
        予測対象の特徴量。
        MultiIndex(Date, Code)を持つDataFrame。
        特徴量の列名と順序は学習時と一致させる。

    Returns
    -------
    pd.Series
        予測されたAlpha値。
        X_predと同じindexを持つSeries。
    """
    _validation_prediction_data(model, X_pred)

    if X_pred.empty:
        return pd.Series(index=X_pred.index, name="Alpha", dtype=float)

    predictions = model.predict(X_pred, validate_features=True)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)

    return pd.Series(predictions, index=X_pred.index, name="Alpha")


def _validation_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    _validation_features(X_train)

    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series")

    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must have the same index")

    try:
        y_values = y_train.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("y_train must contain numeric values") from exc

    if not np.isfinite(y_values).all():
        raise ValueError("y_train must contain only finite values")


def _validation_prediction_data(
    model: lgb.Booster,
    X_pred: pd.DataFrame,
) -> None:
    if not isinstance(model, lgb.Booster):
        raise TypeError("model must be a LightGBM Booster")

    _validation_features(X_pred, allow_empty=True)

    if list(X_pred.columns) != model.feature_name():
        raise ValueError("X_pred feature names and order must match the model")


def _validation_features(
    X: pd.DataFrame,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if not isinstance(X.index, pd.MultiIndex) or X.index.names != [
        "Date",
        "Code",
    ]:
        raise ValueError("X must have a MultiIndex (Date, Code)")

    if X.index.has_duplicates:
        raise ValueError("X index must be unique")

    if X.columns.has_duplicates:
        raise ValueError("X columns must be unique")

    if X.shape[1] == 0 or (X.empty and not allow_empty):
        raise ValueError("X must not be empty")

    if not all(isinstance(column, str) for column in X.columns):
        raise TypeError("X column names must be strings")


def _validation_positive_int(
    value: int,
    *,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")
