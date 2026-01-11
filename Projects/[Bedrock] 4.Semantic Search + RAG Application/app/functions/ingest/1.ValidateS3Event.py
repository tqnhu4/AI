import os
import json
import uuid

ALLOWED = os.environ.get("ALLOWED_BUCKETS", "").split(",") if os.environ.get("ALLOWED_BUCKETS") else []

def lambda_handler(event, context):
    # Expect S3 put event (can be from EventBridge transformed input)
    # Support both s3 event and custom input with bucket/key
    # normalize input:

   # Case 1: S3 direct trigger
    if "Records" in event and event["Records"]:
        rec = event["Records"][0]
        s3 = rec.get("s3", {})
        bucket = s3.get("bucket", {}).get("name")
        key = s3.get("object", {}).get("key")

    # Case 2: EventBridge S3 event
    elif "detail" in event:
        detail = event.get("detail", {})
        bucket = detail.get("bucket", {}).get("name")
        key = detail.get("object", {}).get("key")

    # Case 3: Manual input (for testing)
    else:
        bucket = event.get("bucket")
        key = event.get("key")

    # Validate pdf, json, csv,...

    # Simple validate
    
    if not bucket or not key:
        raise Exception("InvalidEvent: missing bucket or key")

    if ALLOWED and bucket not in ALLOWED:
        raise Exception(f"BucketNotAllowed: {bucket}")

    document_id = str(uuid.uuid4())
    
    return {
        "bucket": bucket,
        "key": key,
        "documentId": document_id
    }