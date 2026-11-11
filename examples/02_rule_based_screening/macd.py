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

# 国内普通株式のみ(内国株券 & 5桁の証券コードの末尾が0なら普通株式)
eq_master = eq_master.loc[
    (eq_master["ProdCat"] == "011") & (eq_master["Code"].str.endswith("0"))
]

# 銘柄名の辞書を作成
code_to_name = eq_master.set_index("Code")["CoName"].to_dict()

# 日足データ取得(bulk)
raw_df = load_bulk_data(
    start=from_date,
    end=target_date,
    client=client,
    cache_dir=cache_dir,
    cache_verbose=2,
)
adjusted_df = adjust_prices(raw_df)

# 基準日に終値が存在する銘柄 & eq_masterに存在
valid_codes = adjusted_df.loc[
    (adjusted_df["Date"] == target_date)
    & adjusted_df["AdjC"].notna()
    & adjusted_df["Code"].isin(eq_master["Code"]),
    "Code",
]

print("All Codes:", len(adjusted_df["Code"].unique()))
print("Valid Codes:", len(valid_codes))

# 対象銘柄の有効な株価系列を取得
screening_df = (
    adjusted_df[
        adjusted_df["Code"].isin(valid_codes) & (adjusted_df["Date"] <= target_date)
    ]
    .dropna(subset=["AdjC"])
    .sort_values(["Code", "Date"])
)

macd_screening_results = []

for code, code_df in screening_df.groupby("Code", sort=False):
    # データ不足の銘柄を除外
    if len(code_df) < 100:
        continue

    code_df = code_df.set_index("Date")

    macd_result = macd(code_df["AdjC"], fast=12, slow=26, signal=9)

    macd_df = pd.DataFrame(
        {
            "MACD": macd_result["MACD"],
            "Signal": macd_result["MACDSignal"],
        },
        index=code_df.index,
    )

    # 最終2営業日を比較
    prev = macd_df.iloc[-2]
    curr = macd_df.iloc[-1]

    golden_cross = (prev["MACD"] <= prev["Signal"]) and (curr["MACD"] > curr["Signal"])

    # NOTE: ゴールデンクロスが発生した日付と、エントリー用の日付は別で扱うのがおすすめ
    # entry_signal = golden_cross.shift(1).fillna(False) といったイメージ

    if not golden_cross:
        continue

    # 銘柄名が取得できなければ銘柄コードで代用
    co_name = code_to_name.get(code, code)

    macd_screening_results.append(
        {
            "Code": code,
            "CoName": co_name,
            "AdjC": code_df["AdjC"].iloc[-1],
            "MACD": curr["MACD"],
            "Signal": curr["Signal"],
            "AdjVo": code_df["AdjVo"].iloc[-1],
            "Va": code_df["Va"].iloc[-1],
        }
    )

# 0件の場合にも対応
macd_screening_df = pd.DataFrame(
    macd_screening_results,
    columns=[
        "Code",
        "CoName",
        "AdjC",
        "MACD",
        "Signal",
        "AdjVo",
        "Va",
    ],
)

macd_screening_df = macd_screening_df.sort_values(by="Va", ascending=False)

print("MACD GoldenCross:")
print(macd_screening_df.head(30))
print("Total:", len(macd_screening_df))
