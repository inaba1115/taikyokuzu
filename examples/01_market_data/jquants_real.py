import jquantsapi
import matplotlib.pyplot as plt
from joblib import Memory

from taikyokuzu.plotting import plot_candlestick

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

print("J-Quants real daily bars:")
print(bars.head())
print()
print("Rows:", len(bars))
print("period:", bars["Date"].min(), "to", bars["Date"].max())

plot_candlestick(
    bars,
    code=code,
    adjusted=True,
)
plt.show()
