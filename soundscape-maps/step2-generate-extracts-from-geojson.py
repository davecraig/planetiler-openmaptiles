#!/usr/bin/env python3
"""
geojson_extract_pmtiles_region_with_sizes.py

For each polygonal feature in a GeoJSON, call:
  pmtiles extract <input.pmtiles> <outdir/<name>.pmtiles> --region=<feature.geojson>

Then write a new GeoJSON that mirrors the input, but adds a property to each
feature indicating the size of the generated extract.

Notes:
  - Only Polygon/MultiPolygon geometries are extracted with --region.
    Non-polygon features are preserved in the output GeoJSON; their size is set to 0 or null.
  - Requires a *clustered* source PMTiles for efficient extract.

Usage:
  python geojson_extract_pmtiles_region_with_sizes.py \
    --input-tiles world.pmtiles \
    --geojson areas.geojson \
    --outdir extracts \
    --output-geojson areas_with_sizes.geojson \
    --size-format bytes \
    --prefix area_ \
    --min-zoom 0 --max-zoom 14
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

POLY_TYPES = {"Polygon", "MultiPolygon"}

def slugify(value: str, max_len: int = 80) -> str:
    v = value.strip().lower()
    v = re.sub(r"[^\w\-\.]+", "-", v)
    v = re.sub(r"-{2,}", "-", v).strip("-")
    return v[:max_len] or "feature"

def ensure_pmtiles_available() -> None:
    try:
        subprocess.run(["./pmtiles", "--help"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        sys.stderr.write("ERROR: 'pmtiles' CLI not found in PATH.\n")
        sys.exit(1)

def feature_to_region_geojson(feat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    geom = feat.get("geometry")
    if not geom:
        return None
    if geom.get("type") in POLY_TYPES:
        return {"type": "Feature", "properties": feat.get("properties", {}), "geometry": geom}
    return None

def run_extract_region(
    input_tiles: str,
    out_path: str,
    region_geojson_obj: Dict[str, Any],
    min_zoom: Optional[int],
    max_zoom: Optional[int],
    extra_args: Optional[List[str]],
) -> int:
    if os.path.isfile(out_path):
        return 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as tf:
        json.dump(region_geojson_obj, tf)
        temp_path = tf.name
    try:
        cmd = ["./pmtiles", "extract", input_tiles, out_path, f"--region={temp_path}"]
        if min_zoom is not None:
            cmd.append(f"--min-zoom={min_zoom}")
        if max_zoom is not None:
            cmd.append(f"--max-zoom={max_zoom}")
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(cmd).returncode
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser(description="Extract per-feature PMTiles using --region and write sizes back to a new GeoJSON.")
    ap.add_argument("--input-tiles", default="/mnt/sdb/map-to-serve/world.pmtiles", help="Source .pmtiles file to extract from (must be clustered).")
    ap.add_argument("--geojson", default="world_countries_and_city_groups.geojson", help="GeoJSON FeatureCollection with polygon features.")
    ap.add_argument("--outdir", default="/mnt/sdb/map-to-serve/extracts", help="Directory to write extracts.")
    ap.add_argument("--output-geojson", default="manifest.geojson", help="Path to write augmented GeoJSON.")
    ap.add_argument("--prefix", default="", help="Prefix for output filenames.")
    ap.add_argument("--suffix", default="", help="Suffix (before .pmtiles) for output filenames.")
    ap.add_argument("--min-zoom", type=int, default=None, help="Optional min zoom for extracts.")
    ap.add_argument("--max-zoom", type=int, default=None, help="Optional max zoom for extracts.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Pass-through args to pmtiles after '--'.")
    args = ap.parse_args()

    ensure_pmtiles_available()

    if not os.path.isfile(args.input_tiles):
        sys.stderr.write(f"ERROR: input tiles not found: {args.input_tiles}\n")
        sys.exit(1)

    with open(args.geojson, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        sys.stderr.write("ERROR: GeoJSON must be a FeatureCollection.\n")
        sys.exit(1)

    features = data.get("features", [])
    if not features:
        sys.stderr.write("ERROR: FeatureCollection has no features.\n")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    successes = 0
    failures = 0
    skipped = 0

    augmented_features = []

    for idx, feat in enumerate(features):
        # Determine a filename for the extract
        filename = None
        if isinstance(feat.get("properties"), dict):
            t = feat["properties"].get("feature_type")
            v = feat["properties"].get("name")
            if t == "country":
                if v is not None:
                    filename = str(v)
            else:
                c = feat["properties"].get("iso_a2")
                if v is not None:
                    if c is not None:
                        filename = str(v) + "-" + str(c)
                    else:
                        filename = str(v)

        if filename is None:
            filename = str(feat.get("id", f"feature_{idx+1}"))

        filename = slugify(args.prefix + filename + args.suffix)
        filename += ".pmtiles"
        out_path = os.path.join(args.outdir, filename)

        # Prepare region GeoJSON if polygonal
        region = feature_to_region_geojson(feat)

        size_value: Optional[int] = None
        ran = False

        print(f"[{idx+1}/{len(features)}] {filename}")

        if region is None:
            skipped += 1
            print("  - non-polygon geometry: skipping extract")
        else:
            print("  - region extract planned")
            code = run_extract_region(
                input_tiles=args.input_tiles,
                out_path=out_path,
                region_geojson_obj=region,
                min_zoom=args.min_zoom,
                max_zoom=args.max_zoom,
                extra_args=(args.extra or []),
            )
            ran = True
            if code == 0 and os.path.isfile(out_path):
                successes += 1
                size_value = os.path.getsize(out_path)
                print(f"  - wrote {out_path} ({size_value} bytes)")
            else:
                failures += 1
                print(f"  - ERROR: pmtiles exited with code {code}")

        # Add/merge the size property on a deep-ish copy
        new_feat = {
            "type": "Feature",
            "geometry": feat.get("geometry"),
            "properties": dict(feat.get("properties", {})),
        }
        if "id" in feat:
            new_feat["id"] = feat["id"]

        # Set size and filename
        new_feat["properties"]["extract-size"] = int(size_value)
        new_feat["properties"]["filename"] = filename

        augmented_features.append(new_feat)

    # Write augmented GeoJSON
    augmented = {"type": "FeatureCollection", "features": augmented_features}
    with open(args.output_geojson, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False)

    print(f"\nDone. Success: {successes} | Failures: {failures} | Skipped (non-polygons): {skipped}")
    print(f"Wrote augmented GeoJSON -> {args.output_geojson}")

if __name__ == "__main__":
    main()
