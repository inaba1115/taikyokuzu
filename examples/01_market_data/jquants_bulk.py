import jquantsapi
import matplotlib.pyplot as plt
import pandas as pd

from taikyokuzu.data import adjust_prices, load_bulk_data
from taikyokuzu.plotting import plot_candlestick

client = jquantsapi.ClientV2()

cache_dir = "./_joblib_cache"

code = "72030"  # Toyota
start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-03-31")

# 日足データ取得(bulk)
raw_df = load_bulk_data(
    start=start_date,
    end=end_date,
    client=client,
    cache_dir=cache_dir,
    cache_verbose=2,
)
adjusted_df = adjust_prices(raw_df)

df = adjusted_df.set_index(["Date", "Code"]).sort_index()
assert df.index.is_unique

print("All Codes:", len(df.index.get_level_values("Code").unique()))

# 使うcodeを抜き出す
bars = df.xs(code, level="Code", drop_level=False)

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
