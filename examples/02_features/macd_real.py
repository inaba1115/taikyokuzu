import os

import jquantsapi
import matplotlib.pyplot as plt
import pandas as pd

from taikyokuzu.features import macd
from taikyokuzu.plotting import draw_candlestick, draw_macd

code = "72030"  # Toyota
from_yyyymmdd = "2025-01-01"
to_yyyymmdd = "2025-03-31"

cache = f"{code}_{from_yyyymmdd}_{to_yyyymmdd}.pkl"


if os.path.isfile(cache):
    print(f"use cache: {cache}")
    bars = pd.read_pickle(cache)
else:
    client = jquantsapi.ClientV2()

    bars = client.get_eq_bars_daily(
        code=code,
        from_yyyymmdd=from_yyyymmdd,
        to_yyyymmdd=to_yyyymmdd,
    )

    print(f"save cache: {cache}")
    bars.to_pickle(cache)

macd_result = macd(bars["AdjC"])

print("J-Quants real daily bars:")
print(bars.head())
print()
print("Rows:", len(bars))
print("period:", bars["Date"].min(), "to", bars["Date"].max())

fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]}
)

draw_candlestick(bars, ax=axes[0], code=code, adjusted=True)
draw_macd(macd_result, ax=axes[1])

fig.tight_layout()

plt.show()
