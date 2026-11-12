import jquantsapi
import pandas as pd
from joblib import Memory

from taikyokuzu.alpha import fit_lightgbm, predict_lightgbm
from taikyokuzu.data import adjust_prices, load_bulk_data
from taikyokuzu.features import macd, rolling_volatility, rsi
from taikyokuzu.targets import forward_returns

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

client = jquantsapi.ClientV2()
cache_dir = "./_joblib_cache"
mem = Memory(cache_dir, verbose=2)
client.get_eq_master = mem.cache(client.get_eq_master)

from_date = pd.Timestamp("2025-01-01")
target_date = pd.Timestamp("2026-10-30")

# --- 上場銘柄一覧 ---
eq_master_df = client.get_eq_master(date=target_date.date())
eq_master_df = eq_master_df.set_index("Code")
assert eq_master_df.index.is_unique
eq_master_df = eq_master_df[["CoName", "ProdCat", "S33"]]

# 国内普通株式のみ(内国株券 & 5桁の証券コードの末尾が0なら普通株式)
eq_master_df = eq_master_df.loc[
    (eq_master_df["ProdCat"] == "011")
    & (eq_master_df.index.get_level_values("Code").str.endswith("0"))
]

# --- 日足データ ---
bars_df = load_bulk_data(
    start=from_date,
    end=target_date,
    client=client,
    cache_dir=cache_dir,
    cache_verbose=2,
)
bars_df = adjust_prices(bars_df)
bars_df = bars_df.set_index(["Date", "Code"]).sort_index()
assert bars_df.index.is_unique

if bars_df.index.get_level_values(level="Date").max() != target_date:
    raise ValueError("please specify a business day for target_date")

# --- X用意 ---
X = pd.DataFrame(index=bars_df.index)

# --- 33業種コード ---
X = X.join(eq_master_df["S33"].astype("category").rename("Sector33"), on="Code")

# --- 日経平均(13210)、TOPIX(13060) ---
X = X.join(
    bars_df["AdjC"]
    .xs("13210", level="Code")
    .pct_change()
    .rename("NIKKEI_Return")
    .sort_index(),
    on="Date",
)
X = X.join(
    bars_df["AdjC"]
    .xs("13060", level="Code")
    .pct_change()
    .rename("TOPIX_Return")
    .sort_index(),
    on="Date",
)

# --- MACD ---
macd_features = (
    bars_df["AdjC"]
    .groupby(level="Code")
    .filter(lambda x: len(x) >= 100)  # データ不足の銘柄を除外
    .groupby(level="Code", group_keys=False)
    .apply(macd)
    .sort_index()
)
X = X.join(macd_features)

# --- RSI ---
rsi_features = (
    bars_df["AdjC"]
    .groupby(level="Code", group_keys=False)
    .apply(rsi)
    .rename("RSI14")
    .sort_index()
)
X = X.join(rsi_features)

# --- Returns ---
returns_features = pd.DataFrame(index=bars_df.index)

for periods in [1, 2, 3, 5, 10, 20, 60]:
    returns_features[f"Return_{periods}D"] = (
        bars_df["AdjC"].groupby(level="Code").pct_change(periods=periods)
    )

for lag in range(1, 6):
    returns_features[f"Return_Lag{lag}"] = (
        returns_features["Return_1D"].groupby(level="Code").shift(lag)
    )

X = X.join(returns_features)

# --- Va, AdjVo ---
X = X.join(bars_df["Va"].groupby(level="Code").pct_change().sort_index())
X = X.join(bars_df["AdjVo"].groupby(level="Code").pct_change().sort_index())

# --- Volatility ---
volatility_features = (
    bars_df["AdjC"]
    .groupby(level="Code", group_keys=False)
    .apply(rolling_volatility)
    .rename("Volatility20")
    .sort_index()
)
X = X.join(volatility_features)

# --- Target: Forward Returns ---
y = (
    bars_df["AdjC"]
    .groupby(level="Code", group_keys=False)
    .apply(lambda x: forward_returns(x, periods=1))
    .sort_index()
)

# --- 対象銘柄の選別 ---
valid_codes = X.index.get_level_values("Code").isin(eq_master_df.index)

X = X.loc[valid_codes]
y = y.loc[valid_codes]

assert X.index.equals(y.index)

# --- Fit/Predict ---
dates = X.index.get_level_values("Date")

train_mask = (dates < target_date) & y.notna()
pred_mask = dates == target_date

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]
X_pred = X.loc[pred_mask]

model = fit_lightgbm(X_train, y_train)
alpha = predict_lightgbm(model, X_pred)

# --- Screening ---
result_df = alpha.sort_values(ascending=False).head(30)
print(result_df)
