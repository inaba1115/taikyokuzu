import jquantsapi
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Memory

from taikyokuzu.plotting import plot_candlestick

client = jquantsapi.ClientV2()

cachedir = "./_joblib_cache"
mem = Memory(cachedir, verbose=2)
client.get_eq_bars_daily = mem.cache(client.get_eq_bars_daily)

code = "72030"  # Toyota
from_date = pd.Timestamp("2025-01-01")
to_date = pd.Timestamp("2025-03-31")

bars = client.get_eq_bars_daily(
    code=code,
    from_yyyymmdd=from_date.date(),
    to_yyyymmdd=to_date.date(),
)

bars = bars.set_index(["Date", "Code"]).sort_index()
assert bars.index.is_unique

print("J-Quants real daily bars:")
print(bars.head())
print()
print("Rows:", len(bars))
print(
    "period:",
    bars.index.get_level_values("Date").min(),
    "to",
    bars.index.get_level_values("Date").max(),
)

# TODO: plot_candlestickが"Date", "Code"がカラムである前提の実装になっているため、
# 一旦、MultiIndexをreset_indexでカラムに戻してから使うこととする
bars = bars.reset_index()

plot_candlestick(
    bars,
    code=code,
    adjusted=True,
)
plt.show()
