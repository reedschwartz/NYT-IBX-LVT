from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd
import pyarrow as pa  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from shapely.geometry import LineString, Point  # type: ignore

FEET_PER_MILE = 5280.0


def format_bbl(boro: str | int, block: str | int, lot: str | int) -> Optional[str]:
    """Format NYC borough, block, lot into a zero-padded BBL string."""
    try:
        return f"{int(boro)}{int(block):05d}{int(lot):04d}"
    except (TypeError, ValueError):
        return None


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a string series with commas to numeric, coercing errors to NaN."""
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def prepare_values_parquet(
    csv_path: Path,
    out_parquet: Path,
    value_field: str = "CURMKTTOT",
    tax_class_field: str = "CURTAXCLASS",
    keep_extra_fields: Optional[Sequence[str]] = None,
    chunk_size: int = 250_000,
) -> Path:
    """Stream the wide DOF CSV into a slim Parquet with BBL, value, and class fields.

    Reading the 5–7 GB CSV directly into memory is brittle; this function keeps only
    the columns needed for joins and aggregation and writes them to a Parquet file.
    """
    usecols: List[str] = ["BORO", "BLOCK", "LOT", value_field]
    if tax_class_field:
        usecols.append(tax_class_field)
    if keep_extra_fields:
        usecols.extend(list(keep_extra_fields))

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if out_parquet.exists():
        out_parquet.unlink()

    writer: pq.ParquetWriter | None = None
    for chunk in pd.read_csv(
        csv_path, usecols=usecols, chunksize=chunk_size, dtype=str, low_memory=False
    ):
        rename_map = {value_field: "value"}
        if tax_class_field:
            rename_map[tax_class_field] = "tax_class"
        chunk = chunk.rename(columns=rename_map)
        chunk["bbl"] = [
            format_bbl(boro, block, lot) for boro, block, lot in zip(chunk["BORO"], chunk["BLOCK"], chunk["LOT"])
        ]
        chunk["value"] = _clean_numeric(chunk["value"])
        if "tax_class" not in chunk.columns:
            chunk["tax_class"] = None
        chunk = chunk.dropna(subset=["bbl"]).drop_duplicates(subset=["bbl"])
        cols = ["bbl", "value", "tax_class"] + [c for c in chunk.columns if c not in {"bbl", "value", "tax_class"}]
        table = pa.Table.from_pandas(chunk[cols], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema)
        writer.write_table(table)

    if writer is None:
        raise RuntimeError(f"No data was written from {csv_path}")
    writer.close()
    return out_parquet


def load_values(values_path: Path) -> pd.DataFrame:
    """Load the slimmed value file (Parquet or CSV with bbl/value columns)."""
    suffix = values_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(values_path)
    else:
        df = pd.read_csv(values_path, dtype={"bbl": str})
    df["bbl"] = df["bbl"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    if "tax_class" in df.columns:
        df["tax_class"] = df["tax_class"].astype(str).str.strip()
    return df


def load_parcels(pluto_path: Path, bbl_field: str = "bbl") -> gpd.GeoDataFrame:
    """Load parcel geometries and normalize the BBL field."""
    parcels = gpd.read_file(pluto_path)
    if parcels.crs is None:
        parcels = parcels.set_crs("EPSG:4326")
    elif parcels.crs.to_epsg() != 4326:
        parcels = parcels.to_crs(epsg=4326)

    if bbl_field not in parcels.columns:
        if {"boro", "block", "lot"}.issubset({c.lower() for c in parcels.columns}):
            boro_col = [c for c in parcels.columns if c.lower() == "boro"][0]
            block_col = [c for c in parcels.columns if c.lower() == "block"][0]
            lot_col = [c for c in parcels.columns if c.lower() == "lot"][0]
            parcels["bbl"] = [
                format_bbl(b, blk, lt) for b, blk, lt in zip(parcels[boro_col], parcels[block_col], parcels[lot_col])
            ]
        else:
            raise KeyError(f"BBL field '{bbl_field}' not found and could not be inferred in {pluto_path}")
    else:
        parcels["bbl"] = parcels[bbl_field]
    parcels["bbl"] = parcels["bbl"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
    return parcels


def build_route_from_stations(stations_path: Path) -> gpd.GeoSeries:
    """Create a line from ordered station points (CSV or GeoJSON/JSON).

    For GeoJSON inputs, prefers a LineString feature if present; otherwise orders
    Point features by the `order` property (if available) to build the line.
    """
    suffix = stations_path.suffix.lower()
    if suffix in {".json", ".geojson"}:
        gdf = gpd.read_file(stations_path)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        # If a LineString exists, use the first one as the alignment
        lines = gdf[gdf.geometry.type == "LineString"]
        if not lines.empty:
            line = lines.geometry.iloc[0]
        else:
            points = gdf[gdf.geometry.type == "Point"].copy()
            if "order" in points.columns:
                points = points.sort_values("order")
            coords = [(geom.x, geom.y) for geom in points.geometry]
            if len(coords) < 2:
                raise ValueError("Stations GeoJSON must contain at least two points to build a line.")
            line = LineString(coords)
        return gpd.GeoSeries([line], crs="EPSG:4326")

    # CSV fallback
    stations_df = pd.read_csv(stations_path)
    if not {"lon", "lat"}.issubset(stations_df.columns):
        raise ValueError("stations file must contain lon and lat columns")
    points = [Point(lon, lat) for lon, lat in zip(stations_df["lon"], stations_df["lat"])]
    line = LineString(points)
    return gpd.GeoSeries([line], crs="EPSG:4326")


def buffer_geometry(line: gpd.GeoSeries, radius_miles: float = 0.5, project_epsg: int = 32118) -> gpd.GeoSeries:
    """Buffer the route line by the desired radius on each side (default 0.5 miles)."""
    line_proj = line.to_crs(epsg=project_epsg)
    buffer_dist = FEET_PER_MILE * radius_miles
    buffered = line_proj.buffer(buffer_dist)
    return buffered.to_crs("EPSG:4326")


def join_parcels_with_values(
    parcels: gpd.GeoDataFrame, values: pd.DataFrame, value_column: str = "value"
) -> gpd.GeoDataFrame:
    """Join parcel geometries to valuation records on BBL."""
    merged = parcels.merge(values, how="inner", on="bbl", validate="1:1")
    if value_column not in merged.columns:
        raise KeyError(f"Value column '{value_column}' not found after merge")
    return merged


def select_taxable(
    parcels: gpd.GeoDataFrame, tax_class_field: str = "tax_class", exclude_classes: Iterable[str] = ("0",)
) -> gpd.GeoDataFrame:
    """Filter out parcels that fall in excluded tax classes (public/utility, etc.)."""
    if tax_class_field not in parcels.columns:
        return parcels
    classes = parcels[tax_class_field].astype(str).str.strip()
    mask = ~classes.isin(set(exclude_classes))
    return parcels[mask]


def clip_to_walkshed(parcels: gpd.GeoDataFrame, walkshed: gpd.GeoSeries) -> gpd.GeoDataFrame:
    """Return parcels intersecting the walkshed polygon."""
    if parcels.crs != walkshed.crs:
        parcels = parcels.to_crs(walkshed.crs)
    idx = parcels.sindex.query(walkshed.iloc[0], predicate="intersects")
    return parcels.iloc[idx]


def compute_uplift(total_value: float, uplift_rates: Iterable[float]) -> pd.DataFrame:
    """Create a table of uplift scenarios."""
    rows = [{"uplift_rate": rate, "incremental_value": total_value * rate} for rate in uplift_rates]
    return pd.DataFrame(rows)


def run_pipeline(
    parcels_path: Path,
    values_path: Path,
    stations_path: Path,
    parcel_bbl_field: str = "bbl",
    value_field: str = "value",
    tax_class_field: str = "tax_class",
    exclude_tax_classes: Iterable[str] = ("0",),
    radius_miles: float = 0.5,
    uplift_rates: Iterable[float] = (0.04, 0.06, 0.08, 0.10),
    output_geojson: Optional[Path] = None,
) -> Tuple[float, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoSeries, gpd.GeoSeries]:
    """Execute the IBX walkshed pipeline end-to-end."""
    values = load_values(values_path)
    parcels = load_parcels(parcels_path, bbl_field=parcel_bbl_field)
    route = build_route_from_stations(stations_path)
    walkshed = buffer_geometry(route, radius_miles=radius_miles)

    merged = join_parcels_with_values(parcels, values, value_column=value_field)
    clipped = clip_to_walkshed(merged, walkshed)
    taxable = select_taxable(clipped, tax_class_field=tax_class_field, exclude_classes=exclude_tax_classes)

    total = float(pd.to_numeric(taxable[value_field], errors="coerce").fillna(0).sum())
    uplift_table = compute_uplift(total, uplift_rates)

    if output_geojson:
        taxable.to_crs(epsg=4326).to_file(output_geojson, driver="GeoJSON")

    return total, uplift_table, taxable, route, walkshed


def export_walkshed_map(
    route: gpd.GeoSeries,
    walkshed: gpd.GeoSeries,
    parcels: gpd.GeoDataFrame,
    output_html: Path,
    value_field: str = "value",
    sample_size: int = 5000,
) -> None:
    """Write a lightweight interactive map using folium."""
    try:
        import folium  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install folium to export the HTML map (e.g., `pip install folium`).") from exc

    center = walkshed.to_crs(epsg=4326).iloc[0].centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="CartoDB positron")

    folium.GeoJson(
        walkshed.__geo_interface__,
        name="0.5 mile buffer",
        style_function=lambda _: {"color": "#1f78b4", "weight": 1, "fillOpacity": 0.12},
    ).add_to(fmap)

    folium.GeoJson(
        route.__geo_interface__,
        name="IBX alignment",
        style_function=lambda _: {"color": "#e34a33", "weight": 3},
    ).add_to(fmap)

    if not parcels.empty:
        sample = parcels.sample(min(len(parcels), sample_size), random_state=1).to_crs(epsg=4326)
        folium.GeoJson(
            sample[[value_field, "bbl", "geometry"]].__geo_interface__,
            name="Walkshed parcels (sample)",
            style_function=lambda _: {"color": "#31a354", "weight": 0.6, "fillOpacity": 0.1},
            tooltip=folium.GeoJsonTooltip(fields=["bbl", value_field]),
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output_html)


def _main() -> None:
    parser = argparse.ArgumentParser(description="IBX walkshed value estimator")
    parser.add_argument("--pluto", required=True, help="Path to MapPLUTO (GeoPackage/Shapefile) with BBL geometries.")
    parser.add_argument(
        "--values",
        required=True,
        help="Parquet or CSV created via prepare_values_parquet containing columns: bbl, value, tax_class.",
    )
    parser.add_argument("--stations", default="data/stations_ibx.csv", help="CSV of stations with lon/lat.")
    parser.add_argument(
        "--bbl-field",
        default="bbl",
        help="Column name in the parcel file containing the BBL (default 'bbl').",
    )
    parser.add_argument("--radius", type=float, default=0.5, help="Buffer radius in miles (default 0.5).")
    parser.add_argument(
        "--exclude-tax-classes",
        nargs="*",
        default=["0"],
        help="Tax class codes to exclude from taxable parcels (e.g., 0 for public/utility).",
    )
    parser.add_argument(
        "--uplift-rates",
        nargs="*",
        type=float,
        default=[0.04, 0.06, 0.08, 0.10],
        help="Value uplift rates to model (decimals).",
    )
    parser.add_argument("--out-geojson", help="Optional path to write the walkshed parcels as GeoJSON.")
    parser.add_argument(
        "--map-html",
        help="Optional path to save an interactive HTML map (requires folium; parcels are sampled for file size).",
    )
    args = parser.parse_args()

    total, uplift_table, taxable, route, walkshed = run_pipeline(
        parcels_path=Path(args.pluto),
        values_path=Path(args.values),
        stations_path=Path(args.stations),
        value_field="value",
        tax_class_field="tax_class",
        parcel_bbl_field=args.bbl_field,
        exclude_tax_classes=args.exclude_tax_classes,
        radius_miles=args.radius,
        uplift_rates=args.uplift_rates,
        output_geojson=Path(args.out_geojson) if args.out_geojson else None,
    )

    print(f"Baseline taxable value in walkshed: ${total:,.0f}")
    print("Uplift scenarios:")
    for _, row in uplift_table.iterrows():
        print(f"  {row['uplift_rate']*100:.1f}% -> ${row['incremental_value']:,.0f}")
    print(f"Parcels in walkshed: {len(taxable):,}")
    print(f"Route length: {route.length.iloc[0]:.3f} degrees (unprojected)")
    print(f"Walkshed polygon ready: {walkshed.iloc[0].geom_type}")
    if args.map_html:
        export_walkshed_map(
            route=route,
            walkshed=walkshed,
            parcels=taxable,
            output_html=Path(args.map_html),
            value_field="value",
        )
        print(f"Saved interactive map to {args.map_html}")


if __name__ == "__main__":  # pragma: no cover
    _main()
