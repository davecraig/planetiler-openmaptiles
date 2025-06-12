#!/bin/bash

usage() { echo "Usage: $0 -t temp_directory -o output_directory [-f]" 1>&2; echo "    -f forces build even if planet file hasn't changed" 1>&2; exit 1; }

build_map=0
while getopts ":t:o:f" opt; do
  case "${opt}" in
    t) tmp_dir=${OPTARG}
       ;;
    o) output_dir=${OPTARG}
       ;;
    f) build_map=1
       ;;
    *) usage
       ;;
  esac
done

if [ -z "${tmp_dir}" ] || [ -z "${output_dir}" ]; then
    usage
fi

printf "Temporary directory is %s\n" "$tmp_dir"
printf "Output directory is %s\n" "$output_dir"
printf "Force build map is  %d\n" "$build_map"

mkdir -p $output_dir/map-to-serve
mkdir -p $tmp_dir/tmp
mkdir -p data/sources

if [ $build_map != 1 ]; then
    printf "Calculating MD5 sum of current planet file...\n"

    # Check if there's a newer file than the one we have
    sum=`md5sum data/sources/planet.osm.pbf | awk '{print $1}'`

    # Download latest planet file md5sum via s3 to see if it's the one we already have
    printf "Downloading MD5 sum of latest planet file available...\n"
    latest_year=`aws s3 ls --no-sign-request s3://osm-planet-eu-central-1/planet/pbf/ | sort | tail -n 1 | awk '{print $2}'`
    latest_file=`aws s3 ls --no-sign-request s3://osm-planet-eu-central-1/planet/pbf/$latest_year | sort | tail -n 1 |  awk '{print $4}'`
    aws s3 cp --no-sign-request s3://osm-planet-eu-central-1/planet/pbf/$latest_year$latest_file.md5 latest.md5
    latest_sum=`cat latest.md5 | awk '{print $1}'`

    if [ "$latest_sum" != "$sum" ]; then
        # New file available, so download it
        printf "There's a new planet file available, so downloading and building new map."
        aws s3 cp --no-sign-request s3://osm-planet-eu-central-1/planet/pbf/$latest_year$latest_file data/sources/planet.osm.pbf
        build_map=1
    else 
        printf "MD5 sums match, so no new planet file available, work complete.\n"
    fi
fi

if [ $build_map == 1 ]; then
    printf "Building new map\n"
    java -Xmx30g -jar ../target/planetiler-openmaptiles-3.15.1-SNAPSHOT-with-deps.jar --download --force --area=planet --fetch-wikidata --tmpdir=$tmp_dir/tmp --output=$output_dir/map-to-serve/test.pmtiles --maxzoom=15 --render_maxzoom=15 --simplify-tolerance-at-max-zoom=-1 --nodemap-type=array --storage=mmap --min_feature_size_at_max_zoom=0 --languages=
fi
