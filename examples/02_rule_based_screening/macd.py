import jquantsapi
import pandas as pd
from joblib import Memory

from taikyokuzu.data import adjust_prices, load_bulk_data
from taikyokuzu.features import macd

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.max_colwidth", None)
# pd.set_option("display.max_columns", None)

client = jquantsapi.ClientV2()
cache_dir = "./_joblib_cache"
mem = Memory(cache_dir, verbose=2)
client.get_eq_master = mem.cache(client.get_eq_master)

# 日付
target_date = pd.Timestamp("2026-11-06")
from_date = target_date - pd.Timedelta(days=300)  # 200営業日ほど欲しいため長めにとる

# 上場銘柄一覧
eq_master_df = client.get_eq_master(date=target_date.date())
eq_master_df = eq_master_df.set_index("Code")
assert eq_master_df.index.is_unique
eq_master_df = eq_master_df[["CoName", "ProdCat"]]

# 国内普通株式のみ(内国株券 & 5桁の証券コードの末尾が0なら普通株式)
eq_master_df = eq_master_df.loc[
    (eq_master_df["ProdCat"] == "011")
    & (eq_master_df.index.get_level_values("Code").str.endswith("0"))
]

# 日足データ取得(bulk)
raw_df = load_bulk_data(
    start=from_date,
    end=target_date,
    client=client,
    cache_dir=cache_dir,
    cache_verbose=2,
)
adjusted_df = adjust_prices(raw_df)

# MultiIndexを設定
adjusted_df = adjusted_df.set_index(["Date", "Code"]).sort_index()
assert adjusted_df.index.is_unique

# 基準日に終値が存在する銘柄 & eq_master_dfに存在
latest_df = adjusted_df.xs(target_date, level="Date")
valid_codes = latest_df.index[
    latest_df["AdjC"].notna() & latest_df.index.isin(eq_master_df.index)
]

print("All Codes:", len(adjusted_df.index.get_level_values("Code").unique()))
print("Valid Codes:", len(valid_codes))

# 対象銘柄の有効な株価系列を取得
screening_df = (
    adjusted_df.loc[adjusted_df.index.get_level_values("Code").isin(valid_codes)]
    .dropna(subset=["AdjC"])
    .sort_index()
)


def calc_macd_features(code_df: pd.DataFrame) -> pd.DataFrame:
    macd_df = macd(code_df["AdjC"])

    macd_line = macd_df["MACD"]
    signal_line = macd_df["MACDSignal"]

    golden_cross = (
        (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    ).astype(int)
    golden_cross.name = "MACDGoldenCross"

    return macd_df.join(golden_cross)


# データ不足の銘柄を除外
screening_df = screening_df.groupby(level="Code").filter(lambda x: len(x) >= 100)

macd_features = (
    screening_df.groupby(level="Code", group_keys=False)
    .apply(calc_macd_features)
    .sort_index()
)

X = screening_df.join(macd_features)

result_df = X.loc[
    (X.index.get_level_values("Date") == target_date) & X["MACDGoldenCross"] == 1
]

result_df = result_df.join(eq_master_df, on="Code")
result_df = result_df.reset_index()
result_df = result_df[["Code", "CoName", "AdjC", "MACD", "MACDSignal", "AdjVo", "Va"]]
result_df = result_df.sort_values(by="Va", ascending=False)

print("MACD GoldenCross:")
print(result_df.head(30))
print("Total:", len(result_df))
