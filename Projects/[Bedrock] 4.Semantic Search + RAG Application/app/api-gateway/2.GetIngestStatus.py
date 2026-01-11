## ✅ **2️⃣ API: `GET /status/{document_id}` — Get statis of ingestion**

### 🎯 **Target**

Display the current status of the document (processing, completed, error, indexed, etc.).


---

### 🧠 **Lambda Handler (GetIngestStatus)**

Get data is saved on DynamoDB table `Documents`:

* `document_id` (PK)
* `status` (“processing”, “completed”, “failed”)
* `timestamp`
* `metadata`

Pseudocode:

```python
import boto3, json
ddb = boto3.resource("dynamodb").Table("Documents")

def lambda_handler(event, context):
    doc_id = event["pathParameters"]["document_id"]
    item = ddb.get_item(Key={"document_id": doc_id}).get("Item")
    
    if not item:
        return {"statusCode": 404, "body": json.dumps({"error": "Document not found"})}
    
    return {"statusCode": 200, "body": json.dumps(item)}
```

---

### 📦 **Output example**

```json
{
  "document_id": "c1a2f88e-9a2b-4d42-8f76-9a85a1c1b9b9",
  "status": "COMPLETED",
  "pages": 12,
  "chunks": 145,
  "created_at": "2025-10-04T09:12:00Z"
}
```