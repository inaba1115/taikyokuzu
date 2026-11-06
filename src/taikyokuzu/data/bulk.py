"""J-Quants API V2 Bulk market data loading."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import jquantsapi
import pandas as pd
from joblib import Memory

_ENDPOINT = "/equities/bars/daily"


def _fetch_bulk_file(
    key: str,
    last_modified: str,
    size: int,
    client: jquantsapi.ClientV2,
) -> pd.DataFrame:
    """Download and parse one gzip CSV (metadata participates in cache key)."""
    # last_modified and size are intentionally part of the joblib cache key,
    # even though download_bulk needs only the key.
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "bars.csv.gz"
        client.download_bulk(key=key, output_path=str(path))
        return pd.read_csv(
            path,
            compression="gzip",
            dtype={"Code": str},
            parse_dates=["Date"],
            low_memory=False,
        )


def load_bulk_data(
    start: str | date,
    end: str | date,
    *,
    client: jquantsapi.ClientV2,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """Return all TSE daily bars between start and end (inclusive).

    Uses ClientV2.get_bulk_list/download_bulk. File data is cached using
    joblib.Memory; a change to LastModified or Size invalidates that entry.
    Passing cache_dir=None disables caching. The listing itself is fetched
    each call so that newly published/revised files are discovered.
    """
    start_dt = pd.Timestamp(start).normalize()
    end_dt = pd.Timestamp(end).normalize()
    if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
        raise ValueError("start must be <= end, and both must be valid dates")

    listing = client.get_bulk_list(
        endpoint=_ENDPOINT,
        from_date=str(start_dt),
        to_date=str(end_dt),
    )
    if listing.empty:
        raise ValueError(f"No available Bulk files for {start_dt.date()}..{end_dt.date()}")

    mem = Memory(cache_dir, verbose=0)
    cached_fetch = mem.cache(_fetch_bulk_file, ignore=["client"])

    # In case monthly and daily files overlap, prefer the newest published file.
    listing = listing.sort_values("LastModified", na_position="first")
    frames: list[pd.DataFrame] = []
    for item in listing.itertuples(index=False):
        frame = cached_fetch(
            key=str(item.Key),
            last_modified=str(item.LastModified),
            size=int(item.Size),
            client=client,
        )
        frames.append(frame)

    bars = pd.concat(frames, ignore_index=True)
    if not {"Date", "Code", "AdjFactor"}.issubset(bars.columns):
        raise ValueError("Bulk CSV must include Date, Code and AdjFactor")
    
    bars["Date"] = pd.to_datetime(bars["Date"])

    bars["Code"] = bars["Code"].astype("string")

    bars = bars.loc[bars["Date"].between(start_dt, end_dt)]
    
    if bars.empty:
        raise ValueError(f"Bulk files contained no rows for {start_dt.date()}..{end_dt.date()}")
    
    return (
        bars.drop_duplicates(subset=["Code", "Date"], keep="last")
        .sort_values(["Code", "Date"])
        .reset_index(drop=True)
    )
