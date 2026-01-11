import os
import boto3

s3 = boto3.client("s3")
REGION = os.environ.get("AWS_REGION", "us-east-1")

def lambda_handler(event, context):
    # event: { "bucket": "...", "key": "...", "documentId": "..." }

    # Case 1: EventBridge S3 event

    if "detail" in event:
        detail = event.get("detail", {})
        bucket = detail.get("bucket", {}).get("name")
        key = detail.get("object", {}).get("key")
    
    # Case 2: Manual input (for testing)
    else:

        bucket = event["bucket"]
        key = event["key"]

    resp = s3.head_object(Bucket = bucket, Key = key)
    metadata = {
        "contentLength": resp.get("ContentLength"),
        "contentType": resp.get("ContentType"),
        "lastModified": resp.get("LastModified").isoformat() if resp.get("LastModified") else None,
        "etag": resp.get("ETag"),
    }
    
    # pass through documentId
    return {
        "documentId": event.get("documentId"),
        "bucket": bucket,
        "key": key,
        "metadata": metadata
    }