## ✅ ** API: `POST /query` — Semantic query (Semantic Search / RAG)**

### 🎯 **Target**

* Receive the user query.
* Search within the vector index (OpenSearch or Kendra).
* Retrieve the most relevant chunks and send them to Bedrock to generate a summarized answer (RAG).

---

### 📤 **Request body example**

```json
{
  "query": "What are the financial policies for 2024?",
  "k": 5,
  "filters": {"document_type": "policy"}
}
```

---

### 🧠 **Lambda Handler (QueryHandler)**

Pseudocode:

```python
import boto3, json
from opensearchpy import OpenSearch

bedrock = boto3.client("bedrock-runtime")
os_client = OpenSearch(hosts=["https://my-domain.region.es.amazonaws.com"])
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v1")
BEDROCK_LLM_MODEL = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-v2")

def lambda_handler(event, context):
    body = json.loads(event["body"])
    query_text = body["query"]

    # Step 1: Create embedding for query
    embedding = bedrock.invoke_model(
        modelId = BEDROCK_EMBED_MODEL,
        body = json.dumps({"inputText": query_text})
    )["embedding"]

    # Step 2: Query OpenSearch (vector kNN)
    resp = os_client.search(
        body = {
            "size": body.get("k", 5),
            "query": {"knn": {"embedding": {"vector": embedding, "k": body.get("k", 5)}}}
        },
        index = "documents-index"
    )
    chunks = [hit["_source"] for hit in resp["hits"]["hits"]]

    # Step 3: Create prompt for Bedrock RAG
    context_text = "\n\n".join([c["text"] for c in chunks])
    prompt = f"Context:\n{context_text}\n\nQuestion: {query_text}\nAnswer concisely and cite sources."

    # Step 4: Call Bedrock model (Claude, Titan, v.v.)
    response = bedrock.invoke_model(
      modelId = BEDROCK_LLM_MODEL,
      body = json.dumps({"inputText": prompt, "maxTokens": 500})
    )

    answer = json.loads(response["body"].read())["outputText"]

    return {
        "statusCode": 200,
        "body": json.dumps({"answer": answer, "sources": chunks})
    }
```

---

### 📦 **Output example**

```json
{
  "answer": "According to the 2024 company policy document, all financial reports must be audited quarterly.",
  "sources": [
    {"document_id": "abc123", "page": 3, "score": 0.94},
    {"document_id": "def456", "page": 7, "score": 0.89}
  ]
}
```