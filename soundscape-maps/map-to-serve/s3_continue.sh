# Get the current UploadId - must only be one active multipart upload
upload_id=`aws --profile davesta s3api list-multipart-uploads --bucket sta-proto-maps-tiles | jq -r ".Uploads[0].UploadId"`

# Get the list of parts
aws --profile davesta s3api list-parts --bucket sta-proto-maps-tiles --key protomaps.pmtiles --upload-id $upload_id > multiparts.json

# Check that there are the right number of parts (1000)

# Edit the multipart.json to remove all `Size` and `LastModified` entries
# Edit the multipart.json to remove all \" and then the , after the `Etag` entries
# Then we only want the parts and not the trailing metadata of the JSON
cat multiparts.json | jq '.Parts | .[] |= {PartNumber, ETag}' | sed 's/\\"//g' > multiparts_trimmed.json

echo '{ "Parts":' > multiparts_complete.json
cat multiparts_trimmed.json >> multiparts_complete.json
echo '}' >> multiparts_complete.json

rm multiparts_trimmed.json

# To land it we run
aws --profile davesta s3api complete-multipart-upload --bucket sta-proto-maps-tiles --key protomaps.pmtiles --upload-id $upload_id --multipart-upload file://multiparts_complete.json

# To abort a multipart upload it would be
# aws --profile davesta s3api abort-multipart-upload --bucket sta-proto-maps-tiles --key protomaps.pmtiles --upload-id $upload_id
