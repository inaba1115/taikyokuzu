import jquantsapi
import numpy as np
import pandas as pd
from joblib import Memory

from taikyokuzu.alpha import fit_lightgbm, predict_lightgbm
from taikyokuzu.data import adjust_prices, load_bulk_data
from taikyokuzu.features import (
    drawdown,
    macd,
    rolling_drawdown,
    rolling_volatility,
    rsi,
)
from taikyokuzu.targets import forward_returns

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

client = jquantsapi.ClientV2()
cache_dir = "./_joblib_cache"
cache_verbose = 2
mem = Memory(cache_dir, verbose=cache_verbose)
client.get_eq_master = mem.cache(client.get_eq_master, ignore=["self"])

from_date = pd.Timestamp("2021-11-12")
target_date = pd.Timestamp("2026-11-12")

# --- Data ---

# 日足データ
bars_df = load_bulk_data(
    start=from_date,
    end=target_date,
    client=client,
    cache_dir=cache_dir,
    cache_verbose=cache_verbose,
)
bars_df = adjust_prices(bars_df)
bars_df = bars_df.set_index(["Date", "Code"]).sort_index()
assert bars_df.index.is_unique

if bars_df.index.get_level_values(level="Date").max() != target_date:
    raise ValueError("please specify a business day for target_date")

# 上場銘柄一覧
eq_master_df = client.get_eq_master(date=target_date.date())
eq_master_df = eq_master_df.set_index("Code")
assert eq_master_df.index.is_unique
eq_master_df = eq_master_df[["CoName", "ProdCat", "S33"]]

# 国内普通株式のみ(内国株券 & 5桁の証券コードの末尾が0なら普通株式)
eq_master_df = eq_master_df.loc[
    (eq_master_df["ProdCat"] == "011")
    & (eq_master_df.index.get_level_values("Code").str.endswith("0"))
]

# --- Market Features ---

# 市場指数の代理となるETFリターンを先に取得
nikkei_return = (
    bars_df["AdjC"].xs("13210", level="Code").pct_change().rename("NIKKEI_ETF_Return")
)

topix_return = (
    bars_df["AdjC"].xs("13060", level="Code").pct_change().rename("TOPIX_ETF_Return")
)

# --- Universe ---

# 普通株式のみ抽出
valid_codes = bars_df.index.get_level_values("Code").isin(eq_master_df.index)
bars_df = bars_df.loc[valid_codes]

# --- Features ---

code_prices = bars_df["AdjC"].groupby(level="Code", group_keys=False)
code_va = bars_df["Va"].groupby(level="Code", group_keys=False)
code_vo = bars_df["AdjVo"].groupby(level="Code", group_keys=False)

X = pd.DataFrame(index=bars_df.index)

X = X.join(eq_master_df["S33"].astype("category").rename("Sector33"), on="Code")
X = X.join(nikkei_return, on="Date")
X = X.join(topix_return, on="Date")

# Calendar
X["Weekday"] = pd.Categorical(
    pd.DatetimeIndex(X.index.get_level_values("Date")).day_of_week
)

# Technical
macd_features = (
    code_prices.filter(lambda x: len(x) >= 100)  # データ不足の銘柄を除外
    .groupby(level="Code", group_keys=False)
    .apply(macd)
)

# 正規化MACD
# macd_features["MACDNorm"] = macd_features["MACD"] / bars_df["AdjC"]
# macd_features["MACDSignalNorm"] = macd_features["MACDSignal"] / bars_df["AdjC"]

rsi_features = (
    bars_df["AdjC"].groupby(level="Code", group_keys=False).apply(rsi).rename("RSI14")
)

X = X.join(macd_features)
X = X.join(rsi_features)

# Volatility
for window in [5, 10, 20, 60]:
    X[f"Volatility{window}D"] = code_prices.apply(rolling_volatility, window=window)

# Drawdown
X["Drawdown"] = code_prices.apply(drawdown)

for window in [5, 10, 20, 60]:
    X[f"Drawdown{window}D"] = code_prices.apply(rolling_drawdown, window=window)

# Returns
for periods in [1, 2, 3, 5, 10, 20, 60]:
    X[f"Return{periods}D"] = code_prices.pct_change(periods=periods)

for lag in range(1, 6):
    X[f"ReturnLag{lag}"] = X["Return1D"].groupby(level="Code").shift(lag)

# Volume
for periods in [1, 2, 3, 5, 10, 20, 60]:
    X[f"VaChange{periods}D"] = code_va.pct_change(periods=periods)

for lag in range(1, 6):
    X[f"VaChangeLag{lag}"] = X["VaChange1D"].groupby(level="Code").shift(lag)

for periods in [1, 2, 3, 5, 10, 20, 60]:
    X[f"AdjVoChange{periods}D"] = code_vo.pct_change(periods=periods)

for lag in range(1, 6):
    X[f"AdjVoChangeLag{lag}"] = X["AdjVoChange1D"].groupby(level="Code").shift(lag)

# --- Target ---

y = code_prices.apply(forward_returns, periods=1)

assert X.index.equals(y.index)

# --- Fit/Predict ---

# 無限大はnanにしておく
numeric_cols = X.select_dtypes(include="number").columns
X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

dates = X.index.get_level_values("Date")

train_mask = (dates < target_date) & np.isfinite(y)
pred_mask = (dates == target_date) & bars_df.loc[
    X.index, "AdjC"
].notna()  # target_dateに終値が存在する銘柄だけを予測する

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]
X_pred = X.loc[pred_mask]

print("Training observations:", len(X_train))
print("Prediction observations:", len(X_pred))
print("Features:", len(X_train.columns))
print("Missing feature ratio:")
print(X_train.isna().mean().sort_values(ascending=False).head(20))
print()

model = fit_lightgbm(X_train, y_train)
alpha = predict_lightgbm(model, X_pred)

importance = pd.Series(
    model.feature_importance(importance_type="gain"),
    index=model.feature_name(),
    name="Gain",
).sort_values(ascending=False)

print("Feature Importance:")
print(importance)
print()

# --- Screening ---

result_df = alpha.sort_values(ascending=False).head(30)
print("Alpha:")
print(result_df)
