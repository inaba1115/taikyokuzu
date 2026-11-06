from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import jquantsapi
import pandas as pd
from joblib import Memory

from taikyokuzu.features import macd

client = jquantsapi.ClientV2()

# cache設定
cachedir = "./_joblib_cache"
mem = Memory(cachedir, verbose=2)
client.get_eq_master = mem.cache(client.get_eq_master)
client.get_eq_bars_daily = mem.cache(client.get_eq_bars_daily)

# 日付関連
to_datetime = datetime(2026, 11, 5, tzinfo=ZoneInfo("Asia/Tokyo"))
backword_days = 300  # 200営業日必要なため
from_datetime = to_datetime - timedelta(days=backword_days)

macd_fast = 12
macd_slow = 26
macd_signal = 9

# 銘柄code取得
eq_master_df = client.get_eq_master(date=to_datetime.date())
eq_master_df = eq_master_df[
    eq_master_df["Code"].isin(["94330", "72590", "51010", "92780", "94380"])
]
print(eq_master_df)


# 株価データ取得
eq_bars_dict = {}

for _, row in eq_master_df.iterrows():
    bars = client.get_eq_bars_daily(
        code=row["Code"],
        from_yyyymmdd=from_datetime.date(),
        to_yyyymmdd=to_datetime.date(),
    )

    bars = bars.set_index("Date")
    eq_bars_dict[row["Code"]] = bars["AdjC"]

eq_bars_df = pd.DataFrame(eq_bars_dict)

# macd計算
macd_screen_results = []

for _, row in eq_master_df.iterrows():
    macd_result = macd(
        eq_bars_df[row["Code"]], fast=macd_fast, slow=macd_slow, signal=macd_signal
    )

    macd_df = pd.DataFrame(
        {
            "AdjC": eq_bars_df[row["Code"]],
            "MACD": macd_result.macd,
            "Signal": macd_result.signal,
        },
    )

    macd_df["GoldenCross"] = (
        (macd_df["MACD"] > macd_df["Signal"])
        & (macd_df["MACD"].shift(1) <= macd_df["Signal"].shift(1))
    ).astype(int)

    if macd_df["GoldenCross"][str(to_datetime.date())] == 1:
        macd_screen_results.append(row["Code"])

print(macd_screen_results)
