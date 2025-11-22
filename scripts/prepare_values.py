"""Slim the DOF valuation CSV into a Parquet with BBL/value/tax_class columns.

Usage:
    python scripts/prepare_values.py \\
        --csv data/Property_Valuation_and_Assessment_Data_Tax_Classes_1,2,3,4_20251120.csv \\
        --out data/property_values.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nyt_ibx_lvt.pipeline import prepare_values_parquet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DOF valuation CSV to Parquet with BBL + value."
    )
    parser.add_argument(
        "--csv", required=True, help="Path to the raw DOF valuation CSV."
    )
    parser.add_argument(
        "--out",
        default="data/property_values.parquet",
        help="Destination Parquet path for slimmed data.",
    )
    parser.add_argument(
        "--value-field",
        default="CURMKTTOT",
        help="Column to use as the primary market value (default CURMKTTOT).",
    )
    parser.add_argument(
        "--tax-class-field",
        default="CURTAXCLASS",
        help="Tax class field to keep (default CURTAXCLASS).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows per chunk when streaming the CSV (default 250k).",
    )
    args = parser.parse_args()

    out_path = prepare_values_parquet(
        csv_path=Path(args.csv),
        out_parquet=Path(args.out),
        value_field=args.value_field,
        tax_class_field=args.tax_class_field,
        chunk_size=args.chunk_size,
    )
    print(f"Wrote slimmed valuation data to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
