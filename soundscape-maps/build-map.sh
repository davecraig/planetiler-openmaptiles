mkdir -p map-to-serve
java -Xmx30g -jar ../target/planetiler-openmaptiles-3.15.1-SNAPSHOT-with-deps.jar --force --download --area=$1 --fetch-wikidata --output=map-to-serve/$1.pmtiles --maxzoom=15 --render_maxzoom=15 --simplify-tolerance-at-max-zoom=-1 --min_feature_size_at_max_zoom=0
