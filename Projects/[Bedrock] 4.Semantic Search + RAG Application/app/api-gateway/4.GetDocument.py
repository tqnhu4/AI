## ✅ **API: `GET /document/{document_id}` — Get metadata & preview chunks**

### 🎯 **Target**

* Allow users to view an overview of the uploaded document.
* Retrieve a list of **indexed chunks** (content preview).

---

### 🧠 **Lambda Handler (GetDocument)**

Pseudocode:

```python
import boto3, json
ddb = boto3.resource("dynamodb").Table("Chunks")

def lambda_handler(event, context):
    doc_id = event["pathParameters"]["document_id"]
    items = ddb.query(
        KeyConditionExpression="document_id = :d",
        ExpressionAttributeValues={":d": doc_id}
    )["Items"]

    return {"statusCode": 200, "body": json.dumps({"document_id": doc_id, "chunks": items[:10]})}
```

---

### 📦 **Output example**

```json
{
  "document_id": "c1a2f88e-9a2b-4d42-8f76-9a85a1c1b9b9",
  "chunks": [
    {"chunk_id": "1", "text": "The company was founded in 2010...", "page": 1},
    {"chunk_id": "2", "text": "In 2024, revenue increased by 15%...", "page": 2}
  ]
}
```