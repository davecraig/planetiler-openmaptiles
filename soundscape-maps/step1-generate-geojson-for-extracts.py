#!/usr/bin/env python3
"""
Build a GeoJSON FeatureCollection with:
- one Feature for most countries (Natural Earth admin 0)
- for countries listed in admin1-iso-a3 no country is added, but the provinces/states are instead
- optionally one Feature per *group* (cluster) of cities with pop >= threshold (Natural Earth populated places)

Examples:
    python generate-geojson-for-extracts.py \
        --input-data-path /path/to/natural_earth_data/

"""

from __future__ import annotations
import argparse
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
import copy
import geopandas as gpd
import numpy as np
from shapely.geometry import shape, Point, MultiPoint, mapping
from shapely.ops import unary_union
from shapely import make_valid  #
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, date
import antimeridian

def jsonable(x):
    """Recursively convert pandas/NumPy objects into plain Python JSON-serializable types."""
    # None / bool / int / float / str are fine
    if x is None or isinstance(x, (bool, int, float, str)):
        # Normalize NaNs to None
        if isinstance(x, float) and (x != x):  # NaN check
            return None
        return x

    # pandas NA-like
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # datetime-like
    if isinstance(x, (datetime, date, pd.Timestamp, np.datetime64)):
        try:
            return pd.to_datetime(x).isoformat()
        except Exception:
            return str(x)

    # NumPy scalar
    if isinstance(x, np.generic):
        return x.item()

    # pandas containers
    if isinstance(x, (pd.Series, pd.Index)):
        return [jsonable(v) for v in x.tolist()]

    # NumPy arrays
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]

    # Mappings
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    # Iterables (lists/tuples/sets)
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]

    # Fallback
    return str(x)

def pick_first_nonnull(df, candidates, default):
    """
    Return a Series with the first non-null across the columns in `candidates`.
    Falls back to `default` if none of the columns exist or all values are null.
    """
    existing = [c for c in candidates if c in df.columns]
    if existing:
        s = df[existing].bfill(axis=1).iloc[:, 0]
        return s.fillna(default)
    else:
        return pd.Series([default] * len(df), index=df.index)

from shapely.ops import transform
from shapely.geometry import Point, MultiPoint
from pyproj import Transformer, CRS

def _lonlat_to_local_transformer(lon, lat):
    """
    Build a local azimuthal-equidistant CRS centered on (lon, lat) and
    return forward/backward transformers between WGS84 <-> local meters.
    """
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform     # lon/lat -> meters
    inv = Transformer.from_crs(aeqd, wgs84, always_xy=True).transform     # meters -> lon/lat
    return fwd, inv

from shapely.ops import transform
from shapely.geometry import MultiPoint
from pyproj import Transformer, CRS

def buffered_polygon_from_points(points_ll, buffer_m=50_000, simplify_m=200):
    """
    Build an enclosing polygon as the union of buffers of radius `buffer_m` (meters)
    around all points. Works in a local azimuthal-equidistant CRS, then reprojects
    back to WGS84. May return MultiPolygon for sparse clusters.
    """
    if not points_ll:
        return None

    # Local CRS centered on cluster centroid to minimize distortion
    clons = [p[0] for p in points_ll]
    clats = [p[1] for p in points_ll]
    lon0 = sum(clons) / len(clons)
    lat0 = sum(clats) / len(clats)
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs")
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    inv = Transformer.from_crs(aeqd, wgs84, always_xy=True).transform

    mp_ll = MultiPoint(points_ll)
    mp_m  = transform(fwd, mp_ll)

    # Union-of-disks buffer (guarantees a border of buffer_m around every city)
    poly_m = mp_m.buffer(buffer_m, cap_style="square")

    # Optional simplification (in meters)
    if simplify_m and simplify_m > 0:
        poly_m = poly_m.simplify(simplify_m, preserve_topology=True)

    return transform(inv, poly_m)

def _find_first_shp_in_dir(d: Path) -> Path:
    shp_candidates = sorted(d.rglob("*.shp"))
    if not shp_candidates:
        raise FileNotFoundError(f"No .shp found inside: {d}")
    return shp_candidates[0]


def resolve_shp_path(input_path: Path) -> Path:
    """
    Accepts:
      - a .zip containing a shapefile set
      - a directory containing .shp
      - a direct .shp path
    Returns a filesystem path to a .shp (extracts zip to a temp dir if needed).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if input_path.suffix.lower() == ".shp":
        return input_path.resolve()

    if input_path.is_dir():
        return _find_first_shp_in_dir(input_path.resolve())

    if input_path.suffix.lower() == ".zip":
        # Extract to a temp dir unique per zip
        tmp_root = Path(tempfile.mkdtemp(prefix="ne_unzip_"))
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(tmp_root)
        return _find_first_shp_in_dir(tmp_root)

    raise ValueError(f"Unsupported path type: {input_path}")


def load_cities(pop_threshold: int, pop_path: Path) -> gpd.GeoDataFrame:
    shp_path = resolve_shp_path(pop_path)
    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)

    # Try common population field names
    pop_field = None
    for cand in ["POP_MAX", "pop_max", "POP_MIN", "pop_min", "POP_OTHER"]:
        if cand in gdf.columns:
            pop_field = cand
            break
    if pop_field is None:
        raise RuntimeError(
            f"Could not find a population field in {shp_path.name}. "
            "Expected something like POP_MAX."
        )

    gdf = gdf[gdf[pop_field].fillna(0) >= pop_threshold].copy()

    # Keep tidy columns (presence-checked)
    keep = [c for c in ["name", "name_ascii", "country_name", "adm0name", "adm0_a3", pop_field] if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]]

    # Normalize
    gdf = gdf.rename(columns={
        "NAME": "name",
        "NAMEASCII": "name_ascii",
        "ADM0NAME": "country_name",
        "ADM0_A3": "iso_a3",
        pop_field: "pop_max",
    })
    gdf.reset_index(drop=True, inplace=True)
    return gdf


def load_countries(admin0_path: Path) -> gpd.GeoDataFrame:
    shp_path = resolve_shp_path(admin0_path)
    gdf = gpd.read_file(shp_path).to_crs(epsg=4326)

    # Normalize column names across NE variants
    rename_map = {
        "NAME": "name",
        "ADMIN": "name_admin",
        "SOVEREIGNT": "sovereignty",
        "ISO_A3": "iso_a3_admin0",
        "WB_A2": "wb_a2",
        "CONTINENT": "continent",
        "SUBREGION": "subregion",
        "POP_EST": "pop_est",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in gdf.columns}
    gdf = gdf.rename(columns=rename_map)

    if "name" not in gdf.columns:
        # Fallbacks
        for cand in ["ADMIN", "NAME", "SOVEREIGNT"]:
            if cand in gdf.columns:
                gdf["name"] = gdf[cand]
                break
        else:
            gdf["name"] = "Unknown"

    if "iso_a3" not in gdf.columns:
        for cand in ["ADM0_A3", "ISO_A3"]:
            if cand in gdf.columns:
                gdf["iso_a3"] = gdf[cand]
                break
        else:
            gdf["iso_a3"] = None

    keep = [c for c in ["name", "iso_a3", "sovereignty", "continent", "subregion", "pop_est"] if c in gdf.columns]
    gdf = gdf[keep + ["geometry"]].copy()
    gdf.reset_index(drop=True, inplace=True)
    return gdf

def _first_col(gdf, choices):
    for c in choices:
        if c in gdf.columns:
            return c
    return None

def load_admin1(admin1_path: Path, only_iso_a3: set[str] | None = None) -> gpd.GeoDataFrame:
    """
    Load Natural Earth admin-1 states/provinces (10m recommended).
    Returns columns: admin1_name, admin1_type, iso_3166_2, hasc, country_name, iso_a3, geometry.
    Dissolves duplicate rows for the same unit id when possible.
    """
    shp = resolve_shp_path(admin1_path)
    gdf = gpd.read_file(shp).to_crs(epsg=4326)

    # Map common NE fields -> unified names
    name_col       = _first_col(gdf, ["name_en", "name", "NAME_EN", "NAME"])
    type_col       = _first_col(gdf, ["type_en", "type", "TYPE_EN", "TYPE"])
    iso2_col       = _first_col(gdf, ["iso_3166_2", "ISO_3166_2"])
    hasc_col       = _first_col(gdf, ["hasc", "HASC", "code_hasc", "CODE_HASC"])
    adm0_iso_col   = _first_col(gdf, ["adm0_a3", "ADM0_A3", "iso_a3", "ISO_A3"])
    country_col    = _first_col(gdf, ["admin", "ADMIN", "adm0_name", "ADM0NAME"])
    unit_id_col    = _first_col(gdf, ["adm1_code", "ADM1_CODE", "iso_3166_2", "HASC", "hasc"])

    # Build a tidy frame with unified columns
    tidy = gpd.GeoDataFrame({
        "name": gdf[name_col] if name_col else "Unknown",
        "admin1_type": gdf[type_col] if type_col else None,
        "iso_3166_2": gdf[iso2_col] if iso2_col else None,
        "hasc": gdf[hasc_col] if hasc_col else None,
        "country_name": gdf[country_col] if country_col else None,
        "iso_a3": gdf[adm0_iso_col] if adm0_iso_col else None,
        "_unit_id": gdf[unit_id_col] if unit_id_col else None,
        "geometry": gdf.geometry
    }, crs="EPSG:4326")

    # Optional country filter
    if only_iso_a3:
        tidy = tidy[tidy["iso_a3"].isin(only_iso_a3)].copy()

    # Dissolve duplicates for same admin-1 unit (if we have an id to group by)
    if "_unit_id" in tidy.columns and tidy["_unit_id"].notna().any():
        # Use 'first' to keep attributes from the first record; geometry gets dissolved
        tidy = tidy.dissolve(by="_unit_id", as_index=False, aggfunc="first")
    else:
        # Fall back: de-duplicate by (admin1_name, iso_a3)
        tidy["_fallback_id"] = tidy["admin1_name"].astype(str) + " | " + tidy["iso_a3"].astype(str)
        tidy = tidy.dissolve(by="_fallback_id", as_index=False, aggfunc="first")
        tidy.drop(columns=["_fallback_id"], inplace=True)

    # Final clean-up
    tidy.drop(columns=["_unit_id"], errors="ignore", inplace=True)
    tidy.reset_index(drop=True, inplace=True)
    return tidy

def _to_single_geometry(geojson_obj):
    """Accepts Geometry, Feature, or FeatureCollection and returns a single Shapely geometry and properties."""
    if geojson_obj.get("type") == "Feature":
        geom = shape(geojson_obj["geometry"])
        props = copy.deepcopy(geojson_obj.get("properties", {}))
        return geom, props
    elif geojson_obj.get("type") in ("Polygon", "MultiPolygon", "GeometryCollection", "MultiLineString", "MultiPoint", "LineString", "Point"):
        return shape(geojson_obj), {}
    elif geojson_obj.get("type") == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in geojson_obj.get("features", []) if f.get("geometry")]
        if not geoms:
            raise ValueError("Empty FeatureCollection")
        return unary_union(geoms), {}
    else:
        raise ValueError("Unsupported GeoJSON object type")

def simplify_superset(geojson_obj, tol=500, buf=500):
    """
    tol, buf in meters (after projection to EPSG:27700).
    Returns a GeoJSON Feature dict with simplified geometry that covers the original.
    """
    try:
        # 1) Get a single shapely geometry in WGS84
        if geojson_obj.get("type") == "FeatureCollection":
            gdf = gpd.GeoDataFrame.from_features(geojson_obj["features"], crs="EPSG:4326")
            country_wgs84 = gdf.unary_union  # shapely geometry
        elif geojson_obj.get("type") == "Feature":
            country_wgs84 = shape(geojson_obj["geometry"])
        else:  # bare geometry (e.g., MultiPolygon)
            country_wgs84 = shape(geojson_obj)

        if crosses_dateline(country_wgs84):
            crs_value = "+proj=laea +lat_0=0 +lon_0=180 +datum=WGS84 +units=m +no_defs"
        else:
            crs_value = 3857

        # 2) Re-project the geometry (not a FeatureCollection) to meters
        country_m = gpd.GeoSeries([country_wgs84], crs="EPSG:4326").to_crs(crs_value).iloc[0]

        # 3) Build the region you want
        region_m = country_m.buffer(buf).simplify(tol, preserve_topology=True)

        # 4) Back to lon/lat as a shapely geometry
        region_wgs84 = gpd.GeoSeries([region_m], crs=crs_value).to_crs("EPSG:4326").iloc[0]

        # 5) If you need GeoJSON **geometry**, do THIS (not .to_json())
        region_geojson_geometry = mapping(region_wgs84)

        if crosses_dateline(country_wgs84):
            # GeoJSON recommends splitting polygons which cross the antimeridian into two!
            region_geojson_geometry = antimeridian.fix_geojson(region_geojson_geometry, fix_winding=True)

        # OPTIONAL: wrap as a Feature (valid)
        region_feature = {
            "type": "Feature",
            "properties": geojson_obj["properties"],
            "geometry": region_geojson_geometry,
        }
        return region_feature
    
    except Exception as e:
        print(e)
        return geojson_obj

def cluster_cities(
    cities: gpd.GeoDataFrame,
    eps_km: float,
    min_samples: int,
    buffer_km: float = 50.0,
    max_cities_per_extract: int | None = None,
    proximity_radius_km: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Step 1: DBSCAN (haversine) to get coarse clusters.
    Step 2: For each DBSCAN cluster, repeatedly:
        - anchor = largest unassigned city by population
        - take nearest neighbors to the anchor (optionally within radius, then cap to N)
        - build polygon = union of buffers (buffer_km) around SELECTED cities
        - mark selected as assigned
    Produces multiple Features per DBSCAN cluster until ALL cities are assigned.
    """
    import pandas as pd
    import numpy as np
    from sklearn.cluster import DBSCAN

    # --- helpers (robust to Natural Earth column variants) ---
    def pick_first_nonnull(df, candidates, default):
        existing = [c for c in candidates if c in df.columns]
        if existing:
            s = df[existing].bfill(axis=1).iloc[:, 0]
            return s.fillna(default)
        else:
            return pd.Series([default] * len(df), index=df.index)

    def as_pop_series(df):
        if "pop_max" in df.columns:
            return pd.to_numeric(df["pop_max"], errors="coerce").fillna(0)
        return pd.to_numeric(
            pick_first_nonnull(df, ["POP_MAX", "POP_MIN", "pop_max", "pop_min", "POP_EST", "pop_est"], 0),
            errors="coerce"
        ).fillna(0)

    def haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0088
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    # --- DBSCAN over haversine ---
    lats = np.array([geom.y for geom in cities.geometry])
    lons = np.array([geom.x for geom in cities.geometry])
    coords_rad = np.radians(np.c_[lats, lons])

    labels = DBSCAN(
        eps=eps_km / 6371.0088,
        min_samples=min_samples,
        metric="haversine"
    ).fit_predict(coords_rad)

    df = cities.copy()
    df["cluster_id"] = labels
    df["_pop"] = as_pop_series(df)

    records = []
    for cid, block in df.groupby("cluster_id"):
        # Work only with needed cols; index will identify rows
        block = block.copy()
        unassigned = set(block.index)

        while unassigned:
            # Anchor = largest unassigned city
            anchor_idx = block.loc[list(unassigned), "_pop"].idxmax()
            anchor = block.loc[anchor_idx]
            a_lon, a_lat = float(anchor.geometry.x), float(anchor.geometry.y)

            # Distances from anchor to *unassigned* cities
            sub = block.loc[list(unassigned)].copy()
            sub["_dist_km"] = haversine_km(
                sub.geometry.x.values, sub.geometry.y.values, a_lon, a_lat
            )

            # Optional radius filter
            if proximity_radius_km is not None:
                sub = sub[sub["_dist_km"] <= float(proximity_radius_km)].copy()

            # Always include anchor
            if anchor_idx not in sub.index:
                sub = pd.concat([sub, anchor.to_frame().T])
                sub = sub.drop_duplicates(subset=sub.columns.difference(["geometry"]), keep="first")

            # Sort by distance and cap
            sub = sub.sort_values("_dist_km", ascending=True)
            if max_cities_per_extract and max_cities_per_extract > 0:
                used = sub.head(max_cities_per_extract).copy()
            else:
                used = sub

            # Geometry from selected cities (union of buffers)
            pts_used = list(zip(used.geometry.x.values, used.geometry.y.values))
            poly = buffered_polygon_from_points(
                pts_used,
                buffer_m=float(buffer_km) * 1000.0,
                simplify_m=200  # set 0 for full detail
            )

            # Aggregate props (selected set)
            names_sel = pick_first_nonnull(used, ["name", "NAME", "name_ascii", "NAMEASCII", "Name"], "Unknown")
            countries_sel = pick_first_nonnull(used, ["country_name", "adm0name", "SOVEREIGNT", "ADMIN"], "Unknown")
            iso_sel = pick_first_nonnull(used, ["iso_a3", "adm0_a3"], "")

            # Anchor props
            anchor_name = pick_first_nonnull(block.loc[[anchor_idx]], ["name", "NAME", "name_ascii", "NAMEASCII", "Name"], "Unknown").iloc[0]
            anchor_country = pick_first_nonnull(block.loc[[anchor_idx]], ["country_name", "adm0name", "SOVEREIGNT", "ADMIN"], "Unknown").iloc[0]
            anchor_iso = pick_first_nonnull(block.loc[[anchor_idx]], ["iso_a3", "adm0_a3"], "").iloc[0]

            total_pop_used = int(used["_pop"].sum())
            records.append({
                "dbscan_cluster_id": int(cid),
                "anchor_city": str(anchor_name),
                "anchor_country": str(anchor_country),
                "anchor_iso_a3": str(anchor_iso),
                "anchor_pop_max": int(float(anchor["_pop"])),
                "anchor_lon": a_lon,
                "anchor_lat": a_lat,
                "city_count": int(len(used)),
                "city_names": sorted({str(v) for v in names_sel.dropna().tolist()}),
                "countries": sorted({str(v) for v in countries_sel.dropna().tolist()}),
                "iso_a3s": sorted({str(v) for v in iso_sel.dropna().tolist()}),
                "total_pop_max": total_pop_used,
                "geometry": poly,
            })

            # Mark selected as assigned and continue
            unassigned -= set(used.index)

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def build_feature(geometry, properties: dict) -> dict:
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def crosses_dateline(geom):
    minx, miny, maxx, maxy = geom.bounds
    # We're a bit loose in our comparison, because by the time we've added our buffer
    # and simplification the resulting polygon can straddle the dateline
    return (maxx > 175) or (minx < -175)

def main():
    ap = argparse.ArgumentParser(description="Generate GeoJSON with countries and clustered cities (local files only).")
    ap.add_argument("--input-data-path", default="/home/dave/Downloads/", help="Path to Natural Earth zip files")
    ap.add_argument("--out", default="world_countries_and_city_groups.geojson", help="Output GeoJSON path")
    ap.add_argument("--pop-threshold", type=int, default=45000, help="Minimum city population to include")
    ap.add_argument("--eps-km", type=float, default=100.0, help="DBSCAN neighborhood radius in kilometers")
    ap.add_argument("--min-samples", type=int, default=1, help="DBSCAN min_samples (1 keeps singletons)")
    ap.add_argument("--cluster-sample", choices=["top", "random"], default="top", help="How to choose cities inside each cluster when limiting: 'top' by population or 'random'.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed used when --cluster-sample=random.")
    ap.add_argument("--buffer-km", type=float, default=50.0, help="Border distance around city clusters")
    ap.add_argument("--proximity-radius-km", type=float, default=80.0, help="Optional: include only cities within this radius of the anchor (largest city). Applied before --max-cities-per-cluster.")
    ap.add_argument("--max-cities-per-extract", type=int, default=20, help="Per-extract cap (anchor + nearest neighbors). If omitted, includes all neighbors.")
    ap.add_argument("--admin1-iso-a3", default="USA,CAN,CHN,JPN,IND,RUS,DEU,POL,BRA,AUS", help="Optional comma-separated ISO A3 list to include (e.g. 'USA,CAN,AUS'). The countries will be skipped if they are in this list. If omitted, include all available.")
    ap.add_argument("--exclude-cities", type=bool, default=False, help="Exclude cities from GeoJSO, defaults to true ")
    ap.add_argument("--exclude-countries", type=bool, default=False, help="Exclude countries and states from GeoJSO, defaults to true ")
    args = ap.parse_args()

    pop_path = Path(args.input_data_path) / "ne_10m_populated_places_simple.zip"
    admin0_path = Path(args.input_data_path) / "ne_10m_admin_0_countries.zip"
    admin1_path = Path(args.input_data_path) / "ne_10m_admin_1_states_provinces.zip"
    iso_filter = set([s.strip() for s in args.admin1_iso_a3.split(",")]) if args.admin1_iso_a3 else None

    countryToContinent = {}
    features = []

    if not args.exclude_countries:
        countries = load_countries(admin0_path)
        states = load_admin1(admin1_path, only_iso_a3=iso_filter)

        for _, row in countries.iterrows():
            # Countries
            raw_props = {k: row[k] for k in countries.columns if k != "geometry"}

            add_country = True
            iso_a3_value = raw_props["iso_a3"]

            # Fix South Sudan iso a3 value
            if iso_a3_value == "SDS":
                iso_a3_value = "SSD"
                raw_props["iso_a3"] = iso_a3_value

            countryToContinent[iso_a3_value] = raw_props["continent"]
            for v in iso_filter:
                if v == iso_a3_value:
                    add_country = False

            if raw_props["name"] == "Antarctica":
                add_country = False

            if add_country:
                raw_props["feature_type"] = "country"
                props = jsonable(raw_props)  # <-- sanitize
                features.append({
                    "type": "Feature",
                    "geometry": mapping(row.geometry),
                    "properties": props,
                })

        # States and provinces
        for _, row in states.iterrows():
            raw_props = {k: row[k] for k in states.columns if k != "geometry"}
            raw_props["feature_type"] = "admin1"
            raw_props["continent"] = countryToContinent[raw_props["iso_a3"]]
            features.append({
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": jsonable(raw_props),  # use your existing sanitizer
            })

        print(f"Loaded {len(countries)} countries and {len(states)} states.")

    # We want to simplify each of the countries, states and provinces. We don't mind
    # the polygons being larger, but we'd like them to have fewer points. This code adds
    # a buffer region of 100km round the edge and simplifies the polygon. This reduces
    # the size of the output GeoJSON greatly.
    simplified_features = [
        simplify_superset(f, tol=10000, buf=100000) 
        for f in features
    ]

    if not args.exclude_cities:
        cities = load_cities(args.pop_threshold, pop_path)
        print(f"Loaded {len(cities)} cities (pop ≥ {args.pop_threshold}).")

        clusters = cluster_cities(
            cities,
            eps_km=args.eps_km,
            min_samples=args.min_samples,
            buffer_km=args.buffer_km,
            max_cities_per_extract=args.max_cities_per_extract,
            proximity_radius_km=args.proximity_radius_km,
        )

        # City clusters
        for _, row in clusters.iterrows():
            raw_props = {k: row[k] for k in clusters.columns if k != "geometry"}
            raw_props["feature_type"] = "city_cluster"
            raw_props["continent"] = countryToContinent[raw_props["anchor_iso_a3"]]
            props = jsonable(raw_props)  # <-- sanitize
            simplified_features.append({
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": props,
            })
        
        print(f"Added {len(clusters)} city extracts.")

    fc = {"type": "FeatureCollection", "features": simplified_features}

    out_path = Path(args.out)
    out_path.write_text(json.dumps(fc, ensure_ascii=False))
    print(f"Wrote GeoJSON → {out_path.resolve()}")


if __name__ == "__main__":
    # Dependencies:
    #   pip install geopandas shapely pyproj scikit-learn tqdm antimeridian
    main()
