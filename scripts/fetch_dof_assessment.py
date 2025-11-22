"""Fetch and minimally clean NYC DOF assessment data (dataset id: 8y4t-faws).

Usage:
    python scripts/fetch_dof_assessment.py --out-dir data/ --app-token YOUR_TOKEN

What it does:
- Attempts to use `sodapy` to page through the Socrata dataset `8y4t-faws` and save a raw Parquet/CSV.
- If `sodapy` is not installed, falls back to the HTTP view endpoint you provided and pages using `$limit`/`$offset`.
- Loads the raw data into pandas, prints the discovered columns, and writes a cleaned CSV with a conservative set
  of fields likely useful for valuation/ownership analysis (bbl/parid, owner name, assessment/market/exemption/taxable fields,
  property/tax class, lot/area fields). The script will select columns by case-insensitive substring matches so it adapts to
  slight variations in field names.

Notes:
- Large downloads may take time; supply an `--app-token` to increase rate limits.
- The script does not make assumptions about exact column names; it inspects and picks columns by matching keywords.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List

import pandas as pd


def _chunked_get_sodapy(dataset_id: str, client, limit: int = 50000):
    """Yield rows in chunks from a sodapy client.get call using offset."""
    offset = 0
    while True:
        rows = client.get(dataset_id, limit=limit, offset=offset)
        if not rows:
            break
        yield from rows
        offset += len(rows)


def _chunked_get_requests(view_url: str, limit: int = 50000):
    """Yield rows by querying the Socrata view endpoint with $limit and $offset parameters using requests."""
    import requests

    offset = 0
    session = requests.Session()
    while True:
        params = {"$limit": limit, "$offset": offset}
        resp = session.get(view_url, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        yield from rows
        offset += len(rows)


def normalize_col(col: str) -> str:
    return col.strip().lower().replace(" ", "_").replace("/", "_")


def choose_columns(cols: List[str]) -> List[str]:
    """Choose a conservative set of columns from the raw DOF dataset by matching keywords.

    This function returns column names (exact as in the source) that match useful keywords.
    """
    keep_keywords = [
        "bbl",
        "parid",
        "block",
        "lot",
        "borough",
        "owner",
        "market",
        "assess",
        "assessed",
        "actual",
        "taxable",
        "exempt",
        "taxclas",
        "building",
        "lotarea",
        "lot_area",
        "lot_front",
        "year",
    ]

    cols_lower = {c: c.lower() for c in cols}
    chosen = []
    for c in cols:
        low = cols_lower[c]
        for kw in keep_keywords:
            if kw in low:
                chosen.append(c)
                break
    # Always include any exact matches for common ids
    for must in ("bbbl", "bbl", "parid"):
        for c in cols:
            if c.lower() == must and c not in chosen:
                chosen.insert(0, c)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and clean DOF assessment data (dataset 8y4t-faws)"
    )
    parser.add_argument(
        "--out-dir", default="data", help="Directory to save raw + cleaned outputs"
    )
    parser.add_argument(
        "--app-token", default=None, help="Socrata app token (optional)"
    )
    parser.add_argument(
        "--limit", type=int, default=50000, help="Chunk size to page (default 50k)"
    )
    parser.add_argument(
        "--dataset-id", default="8y4t-faws", help="Socrata dataset id or view id"
    )
    parser.add_argument(
        "--view-url",
        default="https://data.cityofnewyork.us/api/v3/views/8y4t-faws/query.json",
        help="Fallback view URL",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_parquet = out_dir / "dof_assessment_raw.parquet"
    cleaned_csv = out_dir / "dof_assessment_cleaned.csv"

    # Try to download using sodapy first
    rows = []
    try:
        from sodapy import Socrata

        print("Using sodapy to download data (recommended).")
        client = Socrata("data.cityofnewyork.us", args.app_token or None)
        # Page through
        count = 0
        for r in _chunked_get_sodapy(args.dataset_id, client, limit=args.limit):
            rows.append(r)
            count += 1
            if count % 10000 == 0:
                print(f"Downloaded {count} rows so far...")
        print(f"Finished download: {count} rows")
    except Exception as ex:
        print(
            "sodapy not available or failed — falling back to requests against view URL."
        )
        import requests

        count = 0
        for r in _chunked_get_requests(args.view_url, limit=args.limit):
            rows.append(r)
            count += 1
            if count % 10000 == 0:
                print(f"Downloaded {count} rows so far...")
        print(f"Finished download via requests: {count} rows")

    if not rows:
        print("No rows fetched — exiting.")
        return

    # Convert to DataFrame
    df = pd.DataFrame.from_records(rows)
    print(f"Raw DataFrame loaded: {len(df)} rows, {len(df.columns)} columns")
    print("Columns:")
    for c in df.columns:
        print("  ", c)

    # Choose columns conservatively by keyword matching
    chosen = choose_columns(list(df.columns))
    print(f"Chosen {len(chosen)} columns for cleaning: {chosen}")

    df_chosen = df[chosen].copy()

    # Normalize column names to safe snake_case
    df_chosen.columns = [normalize_col(c) for c in df_chosen.columns]

    # Try to coerce numeric columns
    for c in df_chosen.columns:
        if any(
            k in c
            for k in (
                "market",
                "assess",
                "actual",
                "taxable",
                "exempt",
                "area",
                "lot",
                "front",
            )
        ):
            # Remove commas and coerce
            df_chosen[c] = pd.to_numeric(
                df_chosen[c].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ).fillna(0)

    # If a BBL-like column exists (bbl or bbl*), try to standardize
    bbl_cols = [
        c
        for c in df_chosen.columns
        if c.lower().startswith("bbl") or c.lower().endswith("_bbl")
    ]
    if bbl_cols:
        # pick first
        bbl_col = bbl_cols[0]
        df_chosen["bbl"] = (
            df_chosen[bbl_col].astype(str).str.replace("\D", "", regex=True)
        )
        # Some BBLs may be zero-padded or concatenated; leave as string for joins
    else:
        # Try parid
        parid_cols = [c for c in df_chosen.columns if "parid" in c]
        if parid_cols:
            df_chosen["parid"] = df_chosen[parid_cols[0]].astype(str)

    # Save raw snapshot and cleaned CSV
    try:
        df.to_parquet(raw_parquet)
        print(f"Saved raw parquet to {raw_parquet}")
    except Exception:
        raw_csv = out_dir / "dof_assessment_raw.csv"
        df.to_csv(raw_csv, index=False)
        print(f"Saved raw CSV to {raw_csv}")

    df_chosen.to_csv(cleaned_csv, index=False)
    print(f"Saved cleaned CSV to {cleaned_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
