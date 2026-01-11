## ✅ **1️⃣ API: `POST /ingest` — Upload or provide an S3 key for processing & indexing**

### 🎯 **Objective**

Allow users to:

* Upload a new document to the system, **or**
* Provide an **S3 key** to trigger the processing pipeline (Step Functions).

---

### 📤 **Example request body**

```json
{
  "filename": "document1.pdf",
  "s3_key": "raw/document1.pdf"
}
```

Or (if the client has not yet uploaded the file):

```json
{
  "filename": "document1.pdf",
  "content_type": "application/pdf"
}
```


import json
import boto3
import base64
import uuid
import os
import cgi
from io import BytesIO

s3 = boto3.client("s3")
stepfunctions = boto3.client("stepfunctions")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
STEP_FUNCTION_ARN = os.environ.get("STEP_FUNCTION_ARN", None)

def lambda_handler(event, context):
    try:
        # Get header Content-Type
        content_type = event["headers"].get("Content-Type") or event["headers"].get("content-type")
        if not content_type or "multipart/form-data" not in content_type:
            return _response(400, {"error": "Content-Type must be multipart/form-data"})

        # Parse body (multipart)
        body = base64.b64decode(event["body"]) if event.get("isBase64Encoded", False) else event["body"].encode("utf-8")
        environ = {"REQUEST_METHOD": "POST"}
        fp = BytesIO(body)
        headers = {"content-type": content_type}
        form = cgi.FieldStorage(fp = fp, environ = environ, headers = headers)

        # Get file
        file_item = form["file"]
        if not file_item.filename:
            return _response(400, {"error": "No file uploaded"})

        filename = file_item.filename
        file_data = file_item.file.read()

        # Create document_id
        document_id = f"doc-{uuid.uuid4()}"
        s3_key = f"raw/{document_id}-{filename}"

        # Upload to S3
        s3.put_object(
            Bucket = BUCKET_NAME,
            Key = s3_key,
            Body = file_data,
            ContentType = file_item.type,
            Metadata={
                "original_filename": filename
            }
        )

        # (Optional) Send to Step Functions pipeline
        if STEP_FUNCTION_ARN:
            stepfunctions.start_execution(
                stateMachineArn = STEP_FUNCTION_ARN,
                input = json.dumps({
                    "document_id": document_id,
                    "bucket": BUCKET_NAME,
                    "s3_key": s3_key,
                    "filename": filename
                })
            )

        return _response(200, {
            "message": "File uploaded successfully",
            "document_id": document_id,
            "s3_key": s3_key,
            "status": "upload_received"
        })

    except Exception as e:
        print("Error:", str(e))
        return _response(500, {"error": str(e)})

def _response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body)
    }