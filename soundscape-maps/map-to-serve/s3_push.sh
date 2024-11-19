mkdir -p splitfiles
cd splitfiles

# Split file into 1000 parts
split -n1000 -a4 --numeric-suffixes=1 ../$1 part-

# Login to aws
aws --profile davesta sso login

# Create a multipart upload
upload_id=`aws --profile davesta s3api create-multipart-upload --bucket sta-proto-maps-tiles --key protomaps.pmtiles | jq -r '.UploadId'`
# Continue with original multipart upload (this can be used to continue an upload which timedout)
#upload_id=`aws --profile davesta s3api list-multipart-uploads --bucket sta-proto-maps-tiles | jq -r ".Uploads[0].UploadId"`

# Upload each part of the file (if continuing an upload, adjust the start value here)
for i in {1..1000}
do
    printf -v j "%04d" $i
    echo "Upload $j"
    aws --profile davesta s3api upload-part --bucket sta-proto-maps-tiles --key protomaps.pmtiles --part-number $i --body part-$j --upload-id $upload_id --output text
done

