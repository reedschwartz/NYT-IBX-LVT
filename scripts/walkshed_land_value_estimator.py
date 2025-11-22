"""
walkshed_land_value_estimator
================================

This module provides utilities for estimating the existing land and
property values within a half‑mile walking distance (a **walkshed**) of
a proposed transit station, and computing the potential value uplift
that could result from the construction of new transit infrastructure.

The functions below are intentionally modular so that analysts can
swap in different data sources or adjust assumptions without rewriting
the entire workflow.  To use this code you will need parcel‑level
geospatial data (for New York City this is commonly provided through
the `MapPLUTO` dataset) and a list of candidate station coordinates.

Key assumptions and caveats
---------------------------

* **Walkshed radius.**  The default radius is 0.5 miles (~800 m).
  Research cited in the accompanying policy document indicates that
  property value impacts from transit investment decay rapidly beyond
  a ten‑minute walk; therefore a half‑mile buffer is a standard
  definition for the impact zone【616782623773001†L798-L813】.

* **Property values.**  When available, market value should be used
  rather than assessed value.  In New York City the Department of
  Finance’s assessment roll includes `full_market_value`, and the
  Planning Department’s `MapPLUTO` dataset includes `AssessTot` and
  `AssessLand`.  If only assessed values are available, analysts may
  need to apply an equalisation ratio or other adjustment to reflect
  true market value【616782623773001†L818-L839】.

* **Tax‑exempt parcels.**  Public property, parks, non‑profits and
  similar parcels are often exempt from property tax.  These lots
  should not be included when modelling revenue capture because they
  will not directly contribute to a tax increment【616782623773001†L842-L848】.

* **Co‑ops, condos and abatements.**  Some housing types in New York
  are systematically assessed below market value.  If a large share
  of the study area consists of co‑ops or condominiums, or properties
  benefiting from abatements such as 421‑a or J‑51, you may wish to
  apply an upward adjustment factor or supplement the model with
  sales data【616782623773001†L855-L867】.

The example `estimate_walkshed_value` function below demonstrates how
to combine these pieces into a workflow: load parcel geometries,
create buffers around stations, spatially join parcels into the
walkshed, exclude exempt lots, sum the selected value field and
calculate value uplift scenarios.

Requires data.cityofnewyork.us API key.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
from shapely.geometry import Point  # type: ignore
from shapely.ops import unary_union  # type: ignore


@dataclass
class Station:
    """Representation of a station.

    Attributes
    ----------
    name: str
        Human‑readable identifier for the station.
    lon: float
        Longitude in decimal degrees.
    lat: float
        Latitude in decimal degrees.
    """

    name: str
    lon: float
    lat: float

    def to_point(self) -> Point:
        """Return a Shapely Point representing the station location."""
        return Point(self.lon, self.lat)


def miles_to_feet(miles: float) -> float:
    """Convert miles to feet.

    Parameters
    ----------
    miles: float
        Distance in miles.

    Returns
    -------
    float
        Distance in feet.
    """
    return miles * 5280.0


def load_parcels_from_shapefile(path: str, crs_epsg: int = 4326) -> gpd.GeoDataFrame:
    """Load parcel polygons from a shapefile or geopackage.

    The MapPLUTO dataset is distributed as a GIS file; this function
    wraps `geopandas.read_file` and converts the geometry to the desired
    coordinate reference system.

    Parameters
    ----------
    path: str
        Path to the shapefile, geopackage or other file supported by
        GeoPandas.
    crs_epsg: int, default 4326
        EPSG code for the desired output coordinate system.  WGS84
        (`EPSG:4326`) is a common choice for compatibility with web
        mapping and lat/lon coordinates.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing parcel geometries and attributes.
    """
    parcels = gpd.read_file(path)
    if parcels.crs is None:
        raise ValueError(
            f"No CRS defined for file {path}. Specify `crs_epsg` or set CRS on the GeoDataFrame."
        )
    if parcels.crs.to_epsg() != crs_epsg:
        parcels = parcels.to_crs(epsg=crs_epsg)
    return parcels


def load_parcels_from_socrata(
    dataset_id: str,
    app_token: str,
    user: Optional[str] = None,
    password: Optional[str] = None,
    limit: int = 200_000,
    geometry_column: str = "the_geom",
) -> gpd.GeoDataFrame:
    """Load parcel data from NYC Open Data via the Socrata API.

    This function uses the `sodapy` client to query the Socrata endpoint
    and convert the geometry into a GeoDataFrame.  You will need an
    application token to make large requests; the user can supply
    this token.  Because Socrata imposes row limits on unauthenticated
    requests, we recommend providing a login as well for production
    use.

    Parameters
    ----------
    dataset_id: str
        The dataset identifier on data.cityofnewyork.us (e.g. "64uk‑43di" for
        the 2023 MapPLUTO dataset).  See the NYC Open Data portal for
        a list of available datasets.
    app_token: str
        Socrata application token.  If you do not have one, ask the
        user for an API key; this will improve reliability and allow
        higher rate limits.
    user: str, optional
        Username for Socrata.  Only needed for authenticated requests.
    password: str, optional
        Password for Socrata.  Only needed for authenticated requests.
    limit: int, default 200000
        Maximum number of rows to fetch.  MapPLUTO for NYC contains
        around 900k tax lots; if `limit` is smaller you will need to
        page through results using the `offset` parameter of the
        `get` method.
    geometry_column: str, default "the_geom"
        Name of the field containing geometry in Well‑Known Text or JSON
        format.  On many NYC Open Data datasets the geometry is
        returned as a GeoJSON object under this key.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of parcels with geometry.
    """
    try:
        from sodapy import Socrata  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The 'sodapy' package is required to load data from Socrata.\n"
            "Install it via pip (e.g. `pip install sodapy`)."
        ) from e

    client = Socrata("data.cityofnewyork.us", app_token, username=user, password=password)
    # Fetch rows; if you need more than limit rows, page through using offset
    records = client.get(dataset_id, limit=limit)
    df = pd.DataFrame.from_records(records)
    # Convert geometry
    if geometry_column not in df.columns:
        raise KeyError(
            f"Geometry column '{geometry_column}' not found in the dataset."
        )
    # Socrata returns geometry as dict objects with coordinates/lat/lon; use GeoPandas to construct geometries
    geometries: List[Point | None] = []
    for geom in df[geometry_column]:
        if not geom:
            geometries.append(None)
            continue
        # geometry can be a dict like {'type': 'Point', 'coordinates': [-73.98, 40.75]}
        if isinstance(geom, dict):
            if geom.get("type") == "Point":
                coordinates = geom.get("coordinates", None)
                if coordinates and len(coordinates) == 2:
                    geometries.append(Point(coordinates))
                    continue
        # Fallback: unparseable geometry
        geometries.append(None)
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
    return gdf


def make_station_buffers(
    stations: Iterable[Station],
    radius_miles: float = 0.5,
    project_epsg: int = 32118,
) -> gpd.GeoDataFrame:
    """Create buffered polygons around station locations.

    Parameters
    ----------
    stations: Iterable[Station]
        Sequence of `Station` objects.  Each station must provide
        longitude and latitude in decimal degrees.
    radius_miles: float, default 0.5
        Radius of the buffer around each station, expressed in miles.
    project_epsg: int, default 32118
        EPSG code of a projected coordinate reference system suitable
        for buffering.  EPSG 32118 (NAD83 / New York Long Island) is
        appropriate for NYC and uses US survey feet as units.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with a geometry column containing the buffer
        polygons (in the same CRS as input points) and station
        attributes.
    """
    # Convert stations into a GeoDataFrame in WGS84
    station_records = [{"name": st.name, "lon": st.lon, "lat": st.lat} for st in stations]
    gdf = gpd.GeoDataFrame(
        station_records,
        geometry=[Point(rec["lon"], rec["lat"]) for rec in station_records],
        crs="EPSG:4326",
    )
    # Project to a coordinate system in feet for accurate buffering
    gdf_proj = gdf.to_crs(epsg=project_epsg)
    # Buffer distance in feet
    buffer_distance = miles_to_feet(radius_miles)
    gdf_proj["buffer_geom"] = gdf_proj.geometry.buffer(buffer_distance)
    # Return buffers back in WGS84
    buffered = gdf_proj.set_geometry("buffer_geom").to_crs("EPSG:4326")
    return buffered


def assemble_walkshed_polygon(buffers: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Unify multiple station buffers into a single walkshed polygon.

    Parameters
    ----------
    buffers: geopandas.GeoDataFrame
        GeoDataFrame returned by `make_station_buffers`.  Must
        contain a geometry column representing each station’s buffer.

    Returns
    -------
    geopandas.GeoSeries
        A GeoSeries with a single unified polygon representing the
        combined walkshed for all stations.
    """
    # Use unary union to merge all buffer polygons into one geometry
    merged_geom = unary_union(buffers.geometry)
    return gpd.GeoSeries([merged_geom], crs=buffers.crs)


def select_parcels_within_walkshed(
    parcels: gpd.GeoDataFrame,
    walkshed: gpd.GeoSeries,
    keep_columns: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    """Select parcels that intersect the walkshed polygon.

    Parameters
    ----------
    parcels: geopandas.GeoDataFrame
        Parcel dataset with geometry column in the same CRS as the
        walkshed.
    walkshed: geopandas.GeoSeries
        Unified walkshed polygon as returned by `assemble_walkshed_polygon`.
    keep_columns: list of str, optional
        Subset of parcel attributes to retain.  If None, all
        columns are preserved.

    Returns
    -------
    geopandas.GeoDataFrame
        Parcels that fall inside or intersect the walkshed.
    """
    if parcels.crs != walkshed.crs:
        parcels = parcels.to_crs(walkshed.crs)
    # Build spatial index for efficiency
    selected_idx = parcels.sindex.query(walkshed.iloc[0], predicate="intersects")
    selected = parcels.iloc[selected_idx].copy()
    # Filter columns
    if keep_columns is not None:
        # Always include geometry
        cols = list(set(keep_columns) | {"geometry"})
        existing = [c for c in cols if c in selected.columns]
        selected = selected[existing]
    return selected


def exclude_exempt_parcels(
    parcels: gpd.GeoDataFrame,
    exemption_fields: List[str] = ["exempttot", "exemptland"],
    property_class_field: Optional[str] = None,
    exempt_classes: Optional[Iterable[str]] = None,
) -> gpd.GeoDataFrame:
    """Remove parcels that are fully tax‑exempt or otherwise excluded from taxation.

    Parameters
    ----------
    parcels: geopandas.GeoDataFrame
        Parcels selected for the walkshed.  This should include
        fields indicating exemptions and optionally property class.
    exemption_fields: list of str, default ["exempttot", "exemptland"]
        Names of fields that contain exemption amounts.  If any of
        these fields is non‑zero for a parcel, the parcel is
        considered tax‑exempt.
    property_class_field: str, optional
        If provided, an additional field to filter on.  Commonly
        called `taxclass`, `taxclass_curr` or similar.
    exempt_classes: Iterable[str], optional
        Property class codes to exclude.  For example, New York City’s
        class "0" (public facilities) and class "4" (utility) might
        be excluded.

    Returns
    -------
    geopandas.GeoDataFrame
        Parcels after removing those that are tax‑exempt.
    """
    keep_mask = pd.Series(True, index=parcels.index)
    # Filter out parcels with any exemption value > 0
    for field in exemption_fields:
        if field in parcels.columns:
            # Convert to numeric; missing values become 0
            values = pd.to_numeric(parcels[field], errors="coerce").fillna(0)
            keep_mask &= values == 0
    if property_class_field and exempt_classes is not None:
        if property_class_field in parcels.columns:
            cls = parcels[property_class_field].astype(str).str.strip()
            keep_mask &= ~cls.isin([c.strip() for c in exempt_classes])
    return parcels[keep_mask]


def sum_property_values(
    parcels: gpd.GeoDataFrame,
    value_field: str,
    adjustment_factor: float = 1.0,
) -> float:
    """Compute the total property value in the walkshed.

    Parameters
    ----------
    parcels: geopandas.GeoDataFrame
        Parcels within the walkshed after exclusions.  Must include
        a numeric column containing the value to sum.
    value_field: str
        Name of the field representing market or assessed value.  If
        this field is missing, the function will raise a KeyError.
    adjustment_factor: float, default 1.0
        Optional multiplier applied uniformly to the sum.  You can use
        this to account for systematic underassessment (e.g. apply
        1.2 to inflate co‑op and condo values)【616782623773001†L855-L867】.

    Returns
    -------
    float
        Total (adjusted) property value across all parcels.
    """
    if value_field not in parcels.columns:
        raise KeyError(f"Value field '{value_field}' not found in parcel data.")
    values = pd.to_numeric(parcels[value_field], errors="coerce").fillna(0)
    total = float(values.sum())
    return total * adjustment_factor


def calculate_uplift(
    baseline_value: float,
    uplift_percentages: Iterable[float] = (0.04, 0.06, 0.08, 0.10),
) -> pd.DataFrame:
    """Compute incremental property value for several uplift scenarios.

    Parameters
    ----------
    baseline_value: float
        Total baseline property value in dollars.
    uplift_percentages: iterable of float
        Uplift rates expressed as decimals.  Defaults to 4%, 6%, 8%
        and 10% as suggested in the policy analysis【616782623773001†L885-L916】.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns: `uplift_rate` (percentage) and
        `incremental_value` (dollar value of the uplift).
    """
    records = []
    for rate in uplift_percentages:
        incremental = baseline_value * rate
        records.append({"uplift_rate": rate, "incremental_value": incremental})
    return pd.DataFrame(records)


def estimate_walkshed_value(
    parcels_gdf: gpd.GeoDataFrame,
    stations: Iterable[Station],
    value_field: str,
    radius_miles: float = 0.5,
    exempt_fields: List[str] = ["exempttot", "exemptland"],
    property_class_field: Optional[str] = None,
    exempt_classes: Optional[Iterable[str]] = None,
    adjustment_factor: float = 1.0,
    uplift_percentages: Iterable[float] = (0.04, 0.06, 0.08, 0.10),
) -> Tuple[float, pd.DataFrame]:
    """High‑level helper to estimate walkshed baseline value and uplift.

    This function ties together buffering stations, selecting parcels
    within the buffer, filtering out tax‑exempt lots, summing the
    selected value field and computing uplift scenarios.

    Parameters
    ----------
    parcels_gdf: geopandas.GeoDataFrame
        Parcel dataset with geometry and value fields.  Should be
        projected in WGS84 (EPSG 4326) or a CRS compatible with the
        station coordinates.
    stations: Iterable[Station]
        List of station locations.
    value_field: str
        Name of the column containing the property value to sum (e.g.
        `full_market_value`, `assessland`, etc.).
    radius_miles: float, default 0.5
        Buffer radius around each station.
    exempt_fields: list of str, default ["exempttot", "exemptland"]
        Fields used to identify tax‑exempt parcels.
    property_class_field: str, optional
        If provided, the column containing property class codes.
    exempt_classes: iterable of str, optional
        Property class codes to exclude from taxation.
    adjustment_factor: float, default 1.0
        Uniform multiplier applied to the summed value to account for
        underassessment.
    uplift_percentages: iterable of float, default (0.04, 0.06, 0.08, 0.10)
        Uplift rates to examine.

    Returns
    -------
    tuple
        (baseline_value, uplift_df) where baseline_value is the total
        (adjusted) property value in dollars and `uplift_df` is a
        DataFrame containing incremental values for each uplift rate.
    """
    # Create buffers around stations
    buffers = make_station_buffers(stations, radius_miles=radius_miles)
    # Merge into unified walkshed polygon
    walkshed = assemble_walkshed_polygon(buffers)
    # Select parcels within the walkshed
    selected_parcels = select_parcels_within_walkshed(parcels_gdf, walkshed)
    # Remove tax‑exempt parcels
    taxable_parcels = exclude_exempt_parcels(
        selected_parcels,
        exemption_fields=exempt_fields,
        property_class_field=property_class_field,
        exempt_classes=exempt_classes,
    )
    # Sum property values
    baseline = sum_property_values(
        taxable_parcels, value_field=value_field, adjustment_factor=adjustment_factor
    )
    # Compute uplift
    uplift_table = calculate_uplift(baseline, uplift_percentages)
    return baseline, uplift_table


def main() -> None:
    """Command line interface for estimating walkshed value.

    Example usage:

    ```bash
    python walkshed_land_value_estimator.py \
        --parcels path/to/mappluto.shp \
        --stations stations.csv \
        --value-field full_market_value \
        --radius 0.5
    ```

    The `stations.csv` file should contain three columns: `name`,
    `lon` and `lat` (in decimal degrees).  Alternatively you can
    specify stations on the command line using `--station Name lon lat`.
    """
    parser = argparse.ArgumentParser(description="Estimate land value within a walkshed and compute uplift.")
    parser.add_argument("--parcels", required=True, help="Path to parcels file (shapefile or geopackage).")
    parser.add_argument(
        "--value-field",
        required=True,
        help="Name of the field in the parcels dataset containing the property value (e.g. full_market_value).",
    )
    parser.add_argument(
        "--stations",
        help="CSV file with station definitions (columns: name, lon, lat).",
    )
    parser.add_argument(
        "--station",
        nargs=3,
        action="append",
        metavar=("NAME", "LON", "LAT"),
        help="Specify a station directly on the command line (name lon lat).  Can be repeated.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.5,
        help="Buffer radius in miles (default 0.5).  Property value effects decline beyond this distance.",
    )
    parser.add_argument(
        "--exempt-fields",
        nargs="*",
        default=["exempttot", "exemptland"],
        help="Names of parcel fields indicating exemption amounts.  Any parcel with non‑zero value in these fields will be excluded.",
    )
    parser.add_argument(
        "--property-class-field",
        help="Optional field identifying property class codes (e.g. taxclass).",
    )
    parser.add_argument(
        "--exempt-classes",
        nargs="*",
        help="List of property class codes to exclude (e.g. 0 4).",
    )
    parser.add_argument(
        "--adjustment-factor",
        type=float,
        default=1.0,
        help="Uniform multiplier applied to the summed value to account for underassessment.",
    )
    parser.add_argument(
        "--uplift-rates",
        nargs="*",
        type=float,
        help="List of uplift rates expressed as decimals (e.g. 0.04 0.06 0.08 0.10).  If omitted, defaults to (0.04,0.06,0.08,0.10).",
    )
    args = parser.parse_args()

    # Read parcels
    parcels = load_parcels_from_shapefile(args.parcels, crs_epsg=4326)

    # Assemble station list
    stations: List[Station] = []
    if args.stations:
        df_st = pd.read_csv(args.stations)
        for _, row in df_st.iterrows():
            stations.append(Station(name=row["name"], lon=float(row["lon"]), lat=float(row["lat"])))
    if args.station:
        for name, lon, lat in args.station:
            stations.append(Station(name=name, lon=float(lon), lat=float(lat)))
    if not stations:
        raise ValueError("No stations provided.  Use --stations or --station.")

    # Determine uplift rates
    uplift_rates: Iterable[float] = args.uplift_rates if args.uplift_rates else (0.04, 0.06, 0.08, 0.10)

    baseline, uplift_df = estimate_walkshed_value(
        parcels_gdf=parcels,
        stations=stations,
        value_field=args.value_field,
        radius_miles=args.radius,
        exempt_fields=args.exempt_fields,
        property_class_field=args.property_class_field,
        exempt_classes=args.exempt_classes,
        adjustment_factor=args.adjustment_factor,
        uplift_percentages=uplift_rates,
    )

    print(f"Baseline property value (adjusted): ${baseline:,.2f}")
    print()
    print("Uplift scenarios:")
    for _, row in uplift_df.iterrows():
        rate = row["uplift_rate"]
        incr = row["incremental_value"]
        print(f"  {rate*100:.1f}% uplift → incremental value: ${incr:,.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()