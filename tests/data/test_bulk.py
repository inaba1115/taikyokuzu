from __future__ import annotations

import gzip

import pandas as pd

from taikyokuzu.data import load_bulk_data


class FakeClient:
    def __init__(self):
        self.download_calls = []
        self.version = "2026-11-05T18:00:00Z"

    def get_bulk_list(self, **kwargs):
        self.params = kwargs
        return pd.DataFrame([
            {"Key": "equities_bars_daily_20261105.csv.gz", "LastModified": self.version, "Size": 123}
        ])

    def download_bulk(self, key, output_path):
        self.download_calls.append(key)
        raw = pd.DataFrame({
            "Code": ["51010", "94330"],
            "Date": ["2026-11-05"] * 2,
            "C": [7251.0, 2940.0],
            "AdjFactor": [1.0, 1.0],
        })
        with gzip.open(output_path, "wt") as out:
            raw.to_csv(out, index=False)


def test_load_bulk_joblib_cache_and_invalidation(tmp_path):
    client = FakeClient()
    out = load_bulk_data("2026-11-05", "2026-11-05", client=client, cache_dir=tmp_path)
    assert set(out["Code"]) == {"51010", "94330"}
    assert len(client.download_calls) == 1

    second = load_bulk_data("2026-11-05", "2026-11-05", client=client, cache_dir=tmp_path)
    assert len(client.download_calls) == 1

    client.version = "2026-11-06T18:00:00Z"
    third = load_bulk_data("2026-11-05", "2026-11-05", client=client, cache_dir=tmp_path)
    assert len(client.download_calls) == 2
