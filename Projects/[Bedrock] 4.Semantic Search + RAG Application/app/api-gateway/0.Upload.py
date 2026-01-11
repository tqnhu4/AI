import json
import boto3
import os

s3 = boto3.client('s3')
BUCKET = os.environ.get("BUCKET_NAME", "semantic-search-input")

def lambda_handler(event, context):
    filename = json.loads(event['body'])['filename']
    
    presigned_url = s3.generate_presigned_url(
        ClientMethod='put_object',
        Params={'Bucket': BUCKET, 'Key': filename},
        ExpiresIn=3600
    )
    
    return {
        "statusCode": 200,
        "body": json.dumps({"upload_url": presigned_url})
    }
