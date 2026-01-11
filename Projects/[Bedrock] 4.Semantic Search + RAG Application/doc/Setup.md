1. S3

- Name: semantic-search-story

2. Bedrock (embedding)

- Model: amazon.titan-embed-text-v2:0

3. Bedrock (LLMs)

- Model: anthropic.claude-v2 -> anthropic.claude-3-5-haiku-20241022-v1:0 -> us.anthropic.claude-3-5-haiku-20241022-v1:0
-> arn:aws:bedrock:us-east-1:166261255518:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0
->arn:aws:bedrock:us-east-1:166261255518:inference-profile/us.anthropic.claude-3-5-haiku-20241022-v1:0
4. DynamoDB

- Name: semantic-search

- Partition key: documentId (String)
- Sort key: Optional

5. SNS

- Topic: semantic-search-topic
- Subscription: semantic-search-subscription

6. OpenSearch

- Domain: semantic-search

- Create Index:

PUT semantic-index
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024
      },
      "text": { "type": "text" },
      "metadata": { "type": "object" }
    }
  }
}



7. Lambda

- Functions name:
Ingest:
- ValidateS3Event
- GetObjectMetadata
- ExtractAndChunk
- GenerateEmbedding
- IndexChunkToOpenSearch
- FinalizeIngest
- FinalizeNoChucks

Query
- Ingest
- QueryHandler

8. Role - Policies

Ingest:

- ValidateS3EventRole, ValidateS3EventPolicy
- GetObjectMetadataRole, GetObjectMetadataPolicy
- ExtractAndChunkRole, ExtractAndChunkPolicy
- GenerateEmbeddingRole, GenerateEmbeddingPolicy
- IndexChunkToOpenSearchRole, IndexChunkToOpenSearchPolicy
- FinalizeIngestRole, FinalizeIngestPolicy
- FinalizeNoChucksRole, FinalizeNoChucksPolicy



Step Functions:
- StepFunctionsRole, StepFunctionsPolicy

Query:

- IngestRole, IngestPolicy
- QueryHandlerRole, QueryHandlerPolicy

9. Step Functions

- Name: semantic-search


10. Create AWS EventBridge

- Rule Name: S3-StepFunctions

10. Upload to S3



11. API Gateway

- Name: semantic-search
- Method: Ingest, query


**Method**  | `POST`                                                                  
**URL**     | `https://abcd1234.execute-api.ap-southeast-1.amazonaws.com/prod/search` 
**Headers** | `Content-Type: application/json`                                        
**Body**    | JSON

{
  "query": "What is semantic search?",
  "k": 3
}


Query Function

mkdir QueryLambda && cd QueryLambda
pip install requests -t .
cp ../'3.QueryHandler.py' .
zip -r function.zip .

==================
Import Libraries to IndexChunkToOpenSearch

  mkdir lambda_package && cd lambda_package
  pip install requests requests_aws4auth -t .
  cp ../lambda_function.py .


Compress :

zip -r9 function.zip .

Upload to Lambda

==================
OpenSearch Security

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "*"
      },
      "Action": "es:*",
      "Resource": "arn:aws:es:us-east-1:166261255518:domain/semantic-search-open-search/*"
    }
  ]
}
==================

## 🧭 **Goal**

Create an **Amazon OpenSearch Service domain** that:

* ✅ Enables **kNN (vector search)** for embeddings.
* ✅ Minimizes cost (for testing or development environments).
* ✅ Can be easily scaled up for production later.

---

## ⚙️ **1️⃣ Choose the Right Version & Region**

| Option              | Cost-efficient recommendation                                                       |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Version**         | `OpenSearch 2.11` or newer (supports native kNN vector search).                     |
| **Region**          | Choose a cheaper nearby region — e.g., `ap-southeast-1 (Singapore)` or `us-east-1`. |
| **Deployment type** | **Development and testing** (cheaper than production).                              |

---

## 🏗️ **2️⃣ Minimal Cluster Configuration**

### 🔸 **Cluster configuration**

| Setting               | Recommended value                                               |
| --------------------- | --------------------------------------------------------------- |
| **Instance type**     | `t3.small.search` or `t3.medium.search` *(very cost-effective)* |
| **Instance count**    | `1` (single node)                                               |
| **Storage**           | `10–20 GB` EBS (gp3)                                            |
| **Availability zone** | 1 (no Multi-AZ needed for dev/test)                             |

→ With this setup, the cost is only **around $25–35/month**, instead of hundreds.

---

## 🧩 **3️⃣ Enable the kNN Plugin**

When creating the domain (via AWS Console or CloudFormation):

* In the **“Network & security”** section, enable:

  ```bash
  Enable kNN
  ```
* This activates the kNN module, allowing you to create vector indexes like:

  ```json
  {
    "settings": {
      "index": { "knn": true }
    },
    "mappings": {
      "properties": {
        "embedding": {
          "type": "knn_vector",
          "dimension": 1536
        }
      }
    }
  }
  ```

---

## 🔐 **4️⃣ Simple Security Setup**

* **Disable fine-grained access control** if this is for internal or testing use (saves cost).
* For basic security: attach an **IAM role** to your Lambda or EC2 that grants access to OpenSearch.

---

## ☁️ **5️⃣ Network & Access**

| Scenario                     | Configuration                        |
| ---------------------------- | ------------------------------------ |
| Quick testing                | Public access (restrict to your IP)  |
| Lambda / Bedrock integration | Private VPC endpoint (internal only) |

---

## 🧮 **6️⃣ Estimated Monthly Cost**

| Component            | Type              | Cost estimate            |
| -------------------- | ----------------- | ------------------------ |
| OpenSearch instance  | `t3.small.search` | ~$25                     |
| EBS storage (20GB)   | gp3               | ~$2                      |
| Data transfer (~1GB) |                   | ~$0.10                   |
| **Total**            |                   | **~$27/month (~$1/day)** |

> ⚠️ Enabling Multi-AZ or using `m5.large.search` will increase the cost by 3–4x.

---

## 🧰 **7️⃣ (Optional) Create via AWS CLI**

You can quickly create the domain with:

```bash
aws opensearch create-domain \
  --domain-name my-knn-domain \
  --engine-version "OpenSearch_2.11" \
  --cluster-config InstanceType=t3.small.search,InstanceCount=1 \
  --ebs-options EBSEnabled=true,VolumeType=gp3,VolumeSize=20 \
  --node-to-node-encryption-options Enabled=false \
  --encryption-at-rest-options Enabled=false \
  --advanced-options "rest.action.multi.allow_explicit_index=true" \
  --region ap-southeast-1
```

Then enable kNN via API:

```bash
PUT semantic-index
{
  "settings": {"index.knn": true},
  "mappings": {
    "properties": {
      "embedding": {"type": "knn_vector", "dimension": 1536}
    }
  }
}
```

---

## 🧠 **8️⃣ Tips to Further Reduce Cost**

* Delete the domain when not in use (`Delete domain`) — data can stay in S3.
* Create manual snapshots (to S3) for backup.
* Turn the domain on only during test hours (automate with AWS CLI).
* Limit indexing requests (e.g., index only 10–20 docs for testing).

---

## ✅ **Quick Summary**

| Setting               | Cost-saving choice |
| --------------------- | ------------------ |
| Instance              | `t3.small.search`  |
| Count                 | `1`                |
| EBS                   | `20 GB gp3`        |
| kNN                   | Enabled            |
| AZ                    | 1                  |
| Fine-grained security | Disabled           |
| Estimated cost        | ~$25–30/month      |


