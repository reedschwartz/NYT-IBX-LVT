# IBX Land Value Project

Code + notebooks for:
- transit walkshed tessellation
- served census / DOTS
- land value estimation and uplift


Pipeline outline:
- Download the property value data from OpenNY (do this manually)
- Draw a line that tracks proposed paths for the IBX (make this modular so that it can also track other lines) as some sort of interpretable geometry over a map of NYC
- Expand that line to cover .5 miles on both sides (CHECK THIS AGAINST GUPTA PAPER)
- Turn the csv rows for relevant buildings (buildings roughly within the region) into geojsons 
- Pull every building that is within the rough .5m walkshed and their values
- Exclude every building in that sample that is not eligible for the law's value capture mechanism (public land etc.)
- Sum the value and print it, along with a map of the proposed walkshed. Multiply the number by 1.04, 1.06, 1.08, 1.1 to estimate different value uplift scenarios.
- Using the Gupta paper as a model, estimate what tax rates would lead to appropriate value capture over what period of time.

## Pipeline quickstart
- Install dependencies (uv): `uv sync --editable .`  
  Or with pip: `pip install -e .` (or prefix commands with `PYTHONPATH=src` so the package is importable).
- Slim the 5+ GB DOF valuation CSV to Parquet:  
  `uv run python scripts/prepare_values.py --csv data/Property_Valuation_and_Assessment_Data_Tax_Classes_1,2,3,4_20251120.csv --out data/property_values.parquet --value-field CURMKTTOT --tax-class-field CURTAXCLASS`
- Add parcel geometries (MapPLUTO GeoPackage/shapefile/FGDB) under `data/` (e.g., `data/nyc_mappluto_25v3_fgdb/MapPLUTO25v3.gdb`); ensure a `bbl` field exists.
- Run the IBX walkshed estimator (0.5 mile each side of `data/stations_ibx.csv`):  
  `uv run python -m nyt_ibx_lvt.pipeline --pluto data/nyc_mappluto_25v3_fgdb/MapPLUTO25v3.gdb --bbl-field BBL --values data/property_values.parquet --stations data/stations_ibx.csv --out-geojson data/ibx_walkshed.geojson --map-html data/ibx_walkshed.html`
- CLI prints baseline taxable value plus 4/6/8/10% uplift; tune with `--exclude-tax-classes` or `--uplift-rates`. `--map-html` needs `folium` and samples parcels to keep file size small.
- Swap in a different stations CSV to model other alignments.
- Prefer to work interactively? Open `notebooks/ibx_walkshed.ipynb` via `uv run jupyter lab` and follow the cells (it handles CSV→Parquet, runs the pipeline, and exports the map).

## Data references
- Valuation CSV: `data/Property_Valuation_and_Assessment_Data_Tax_Classes_1,2,3,4_20251120.csv`
- Stations: `data/stations_ibx.csv`
- Gupta paper: `data/Gupta - Take the Q Train.pdf`
- DOF dictionary: `data/Property_Assessment_Data_Dictionary.xlsx`
- Parcel geometries: `data/nyc_mappluto_25v3_fgdb` (folder) — MapPLUTO for walkshed selection.
