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
eq_master = client.get_eq_master(date=target_date.date())
eq_master = eq_master.set_index("Code")
eq_master = eq_master[["CoName", "S17", "S33", "ProdCat"]]
assert eq_master.index.is_unique

# 国内普通株式のみ(内国株券 & 5桁の証券コードの末尾が0なら普通株式)
eq_master = eq_master.loc[
    (eq_master["ProdCat"] == "011")
    & (eq_master.index.get_level_values("Code").str.endswith("0"))
]

# 銘柄名の辞書を作成
# code_to_name = eq_master.set_index("Code")["CoName"].to_dict()

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
indexed_df = adjusted_df.set_index(["Date", "Code"]).sort_index()
assert indexed_df.index.is_unique

valid_codes = indexed_df.loc[
    (indexed_df.index.get_level_values("Date") == target_date)
    & indexed_df.index.get_level_values("Code").isin(
        eq_master.index.get_level_values("Code")
    )
    & indexed_df["AdjC"].notna()
].index.get_level_values("Code")

print("All Codes:", len(indexed_df.index.get_level_values("Code").unique()))
print("Valid Codes:", len(valid_codes))

# 対象銘柄の有効な株価系列を取得
screening_df = (
    indexed_df.loc[indexed_df.index.get_level_values("Code").isin(valid_codes)]
    .dropna(subset=["AdjC"])
    .sort_index()
)


def calc_macd_features(code_df: pd.DataFrame) -> None | pd.DataFrame:
    # データ不足の銘柄を除外
    if len(code_df) < 100:
        return None

    macd_df = macd(code_df["AdjC"])

    macd_line = macd_df["MACD"]
    signal_line = macd_df["MACDSignal"]
    golden_cross = (macd_line > signal_line) & (
        macd_line.shift(1) <= signal_line.shift(1)
    ).astype(int)
    golden_cross.name = "MACDGoldenCross"

    return pd.concat([macd_df, golden_cross], axis=1)


macd_features = (
    screening_df.groupby(level="Code", group_keys=False)
    .apply(calc_macd_features)  # type: ignore
    .sort_index()
)


X = screening_df.join(macd_features)
result_df = X.loc[
    (X.index.get_level_values("Date") == target_date) & X["MACDGoldenCross"] == 1
]
result_df = result_df.join(eq_master, on="Code")
result_df = result_df.reset_index()
result_df = result_df[["Code", "CoName", "AdjC", "MACD", "MACDSignal", "AdjVo", "Va"]]
result_df = result_df.sort_values(by="Va", ascending=False)

print("MACD GoldenCross:")
print(result_df.head(30))
print("Total:", len(result_df))
