import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taikyokuzu.features import macd
from taikyokuzu.plotting import draw_macd

index = pd.bdate_range("2025-01-01", periods=60)
prices = pd.DataFrame(
    {
        "B": 100.0 + np.sin(np.linspace(0.0, 10.0, len(index))) * 4.0,
    },
    index=index,
)

asset = prices["B"]
asset_macd = macd(asset, fast=12, slow=26, signal=9)

print("B: MACD")
print(
    pd.DataFrame(
        {
            "MACD": asset_macd.macd,
            "Signal": asset_macd.signal,
        }
    ).tail()
)

fig, ax = plt.subplots(figsize=(10, 8))
draw_macd(asset_macd, ax=ax)

fig.tight_layout()
plt.show()
