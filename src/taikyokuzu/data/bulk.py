"""J-Quants API V2 の Bulk データを取得するモジュール。"""

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
    """Bulk ファイルをダウンロードし、gzip 圧縮された CSV を読み込む。

    Parameters
    ----------
    key : str
        ダウンロード対象の Bulk ファイルを識別するキー。
    last_modified : str
        ファイルの最終更新日時。キャッシュの識別に使用する。
    size : int
        ファイルサイズ。キャッシュの識別に使用する。
    client : jquantsapi.ClientV2
        J-Quants API V2 のクライアント。

    Returns
    -------
    pandas.DataFrame
        CSV から読み込んだ株価データ。``Date`` は日時型、``Code`` は文字列型。

    Notes
    -----
    ``last_modified`` と ``size`` はダウンロード処理には使用しないが、
    ``joblib.Memory`` のキャッシュキーに含めることでファイルの更新を検出する。
    一時ファイルは読み込み後に削除する。
    """
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
    cache_dir: str | Path | None,
    cache_verbose: int = 0,
) -> pd.DataFrame:
    """指定期間の東証上場銘柄の日足株価を Bulk データから取得する。

    Parameters
    ----------
    start : str or datetime.date
        取得期間の開始日（当日を含む）。
    end : str or datetime.date
        取得期間の終了日（当日を含む）。
    client : jquantsapi.ClientV2
        J-Quants API V2 のクライアント。
    cache_dir : str or pathlib.Path
        ``joblib.Memory`` のキャッシュ保存先。
        実行時に ``None`` を渡した場合、キャッシュは無効になる。

    Returns
    -------
    pandas.DataFrame
        指定期間内の日足データ。``Code`` と ``Date`` の組は一意で、
        ``Code``、``Date`` の昇順に並ぶ。

    Raises
    ------
    ValueError
        日付の指定が不正な場合、利用可能なファイルがない場合、
        必須列が不足する場合、または指定期間に該当するデータがない場合。

    Notes
    -----
    ``get_bulk_list`` によるファイル一覧の取得は毎回実行する。
    各ファイルの内容は ``joblib.Memory`` でキャッシュし、
    ``LastModified`` または ``Size`` が変わると再取得する。
    月次・日次ファイルのレコードが重複した場合は、
    公開日時が新しいファイルのデータを優先する。
    """
    start_dt = pd.Timestamp(start).normalize()
    end_dt = pd.Timestamp(end).normalize()
    if pd.isna(start_dt) or pd.isna(end_dt) or start_dt > end_dt:
        raise ValueError("start must be <= end, and both must be valid dates")

    listing = client.get_bulk_list(
        endpoint=_ENDPOINT,
        from_date=str(start_dt.date()),
        to_date=str(end_dt.date()),
    )
    if listing.empty:
        raise ValueError(
            f"No available Bulk files for {start_dt.date()}..{end_dt.date()}"
        )

    mem = Memory(cache_dir, verbose=cache_verbose)
    cached_fetch = mem.cache(_fetch_bulk_file, ignore=["client"])

    # 月次・日次ファイルが重複した場合、公開日時が新しいデータを優先する。
    listing = listing.sort_values("LastModified", na_position="first")
    frames: list[pd.DataFrame] = []
    for item in listing.itertuples(index=False):
        frame = cached_fetch(
            key=str(item.Key),
            last_modified=str(item.LastModified),
            size=int(item.Size),  # type: ignore
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
        raise ValueError(
            f"Bulk files contained no rows for {start_dt.date()}..{end_dt.date()}"
        )

    return (
        bars.drop_duplicates(subset=["Code", "Date"], keep="last")
        .sort_values(["Code", "Date"])
        .reset_index(drop=True)
    )
