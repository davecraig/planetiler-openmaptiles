tmp_dir=$1
output_dir=$2
mkdir -p $output_dir/map-to-serve
mkdir -p $tmp_dir/tmp
mkdir -p data/sources

# Download latest planet file via s3 which will max out my connection and is the fastest way to download
latest_year=`aws s3 ls --no-sign-request s3://osm-planet-eu-central-1/planet/pbf/ | sort | tail -n 1 | awk '{print $2}'`
latest_file=`aws s3 ls --no-sign-request s3://osm-planet-eu-central-1/planet/pbf/$latest_year | sort | tail -n 1 |  awk '{print $4}'`
#:aws s3 cp --no-sign-request s3://osm-planet-eu-central-1/planet/pbf/$latest_year/$latest_file data/sources/planet.osm.pbf

java -Xmx30g -jar ../target/planetiler-openmaptiles-3.15.1-SNAPSHOT-with-deps.jar --download --force --area=planet --fetch-wikidata --tmpdir=$tmp_dir/tmp --output=$output_dir/map-to-serve/test.pmtiles --maxzoom=15 --render_maxzoom=15 --simplify-tolerance-at-max-zoom=-1 --nodemap-type=array --storage=mmap --min_feature_size_at_max_zoom=0
