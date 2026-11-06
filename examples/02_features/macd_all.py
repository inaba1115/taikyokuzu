import jquantsapi
import pandas as pd
from joblib import Memory

from taikyokuzu.data import adjust_prices, load_bulk_data
from taikyokuzu.features import macd

client = jquantsapi.ClientV2()
cache_dir = "./_joblib_cache"
mem = Memory(cache_dir, verbose=2)
client.get_eq_master = mem.cache(client.get_eq_master)

# 日付
target_date = pd.Timestamp("2026-11-05")
from_date = target_date - pd.Timedelta(days=300)  # 200営業日ほど欲しいため長めにとる

# 上場銘柄一覧
eq_master = client.get_eq_master(date=target_date.date())

# 日足データ取得(bulk)
raw_df = load_bulk_data(
    start=from_date,
    end=target_date,
    client=client,
    cache_dir=cache_dir,
    cache_verbose=2,
)
adjusted_df = adjust_prices(raw_df)

# 基準日に終値が存在する銘柄
valid_codes = adjusted_df.loc[
    (adjusted_df["Date"] == target_date) & adjusted_df["AdjC"].notna(),
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

for code in valid_codes:
    code_df = screening_df[screening_df["Code"] == code].set_index("Date")

    macd_result = macd(code_df["AdjC"], fast=12, slow=26, signal=9)

    macd_df = pd.DataFrame(
        {
            "AdjC": code_df["AdjC"],
            "MACD": macd_result.macd,
            "Signal": macd_result.signal,
        }
    )

    # NOTE: 次の日に1が立つようにしたほうが良いかも
    macd_df["GoldenCross"] = (
        (macd_df["MACD"] > macd_df["Signal"])
        & (macd_df["MACD"].shift(1) <= macd_df["Signal"].shift(1))
    ).astype(int)

    if macd_df["GoldenCross"][target_date] == 1:
        co_name = eq_master.loc[eq_master["Code"] == code, "CoName"].values[0]
        adj_vo = code_df.at[target_date, "AdjVo"]

        macd_screening_results.append(
            {"Code": code, "CoName": co_name, "AdjVo": adj_vo}
        )

macd_screening_df = pd.DataFrame(macd_screening_results)
macd_screening_df = macd_screening_df.sort_values(
    by="AdjVo", ascending=False
)  # 出来高が多い順で見る

print("MACD GoldenCross:")
print(macd_screening_df.head())
print("Total:", len(macd_screening_df))
