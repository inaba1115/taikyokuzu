import jquantsapi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory

from taikyokuzu.features import macd
from taikyokuzu.plotting import draw_candlestick, draw_macd

client = jquantsapi.ClientV2()

cachedir = "./_joblib_cache"
mem = Memory(cachedir, verbose=2)
client.get_eq_bars_daily = mem.cache(client.get_eq_bars_daily)

code = "72030"  # Toyota
from_yyyymmdd = "2025-01-01"
to_yyyymmdd = "2025-03-31"

bars = client.get_eq_bars_daily(
    code=code,
    from_yyyymmdd=from_yyyymmdd,
    to_yyyymmdd=to_yyyymmdd,
)

macd_result = macd(bars["AdjC"])

df = pd.DataFrame(
    {
        "Date": bars["Date"],
        "AdjC": bars["AdjC"],
    }
)

df["MACD"] = macd_result.macd
df["Signal"] = macd_result.signal

df["MACD>Signal"] = (df["MACD"] > df["Signal"]).astype(bool)
df["Signal>MACD"] = (df["Signal"] > df["MACD"]).astype(bool)
df["Signal>MACD+1"] = df["Signal>MACD"].shift(1).fillna(False).astype(bool)
df["MACD>Signal+1"] = df["MACD>Signal"].shift(1).fillna(False).astype(bool)

df["GoldenCross"] = np.where(df["MACD>Signal"] & df["Signal>MACD+1"], 1, 0)
df["DeathCross"] = np.where(df["Signal>MACD"] & df["MACD>Signal+1"], 1, 0)

print("J-Quants real daily bars:")
print(df.head(100))
print()
print("Rows:", len(df))
print("period:", df["Date"].min(), "to", df["Date"].max())
print()
print("MACD")
print(macd_result.macd.head())

fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
)

draw_candlestick(bars, ax=axes[0], code=code, adjusted=True)
draw_macd(macd_result, ax=axes[1])

fig.tight_layout()

plt.show()
