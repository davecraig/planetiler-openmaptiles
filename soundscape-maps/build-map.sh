mkdir -p map-to-serve

java -Xmx40g \
  -jar ../target/planetiler-openmaptiles-3.15.1-SNAPSHOT-with-deps.jar \
  --area=$1 --bounds=world --download \
  `# Accelerate the download by fetching the 10 1GB chunks at a time in parallel` \
  --download-threads=10 --download-chunk-size-mb=1000 \
  `# Also download name translations from wikidata` \
  --fetch-wikidata \
  --output=map-to-serve/$1.pmtiles \
  --maxzoom=15 \
  --render_maxzoom=15 \
  --simplify-tolerance-at-max-zoom=-1 \
  --tmpdir=/mnt/sdb/tmp \
  --nodemap-type=array --nodemap-storage=mmap
