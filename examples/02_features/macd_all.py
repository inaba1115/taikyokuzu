from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import jquantsapi
import pandas as pd

from taikyokuzu.data import adjust_prices, load_bulk_data
from taikyokuzu.features import macd

client = jquantsapi.ClientV2()
cache_dir = "./_joblib_cache"

# 日付関連
to_datetime = datetime(2026, 11, 5, tzinfo=ZoneInfo("Asia/Tokyo"))
backword_days = 300  # 200営業日ほど欲しいため長めにとる
from_datetime = to_datetime - timedelta(days=backword_days)

# データ取得(bulk)
raw_df = load_bulk_data(
    start=from_datetime.date(),
    end=to_datetime.date(),
    client=client,
    cache_dir=cache_dir,
    cache_verbose=2,
)
adjusted_df = adjust_prices(raw_df)

# codes = adjusted_df["Code"].unique()
# codes = ["13050"]
# codes = ["94330", "72590", "51010", "92780", "94380"]
codes = ["136A0"]

macd_screening_results = []

for code in codes:
    code_df = adjusted_df[adjusted_df["Code"] == code].set_index("Date")
    print(code_df.head())
    print(len(code_df))

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

    print(code)
    print(macd_df.tail())

    if macd_df["GoldenCross"][str(to_datetime.date())] == 1:
        adj_vo = code_df["AdjVo"][str(to_datetime.date())]
        macd_screening_results.append({"Code": code, "AdjVo": adj_vo})

macd_screening_df = pd.DataFrame(macd_screening_results)
macd_screening_df = macd_screening_df.sort_values(
    by="AdjVo", ascending=False
)  # 出来高が多い順で見る

print("MACD GoldenCross:", macd_screening_df.head())
