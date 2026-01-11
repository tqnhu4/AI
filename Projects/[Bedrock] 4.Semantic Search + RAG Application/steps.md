

### 💻 **#5. Setup the App**

* **Step-by-step Implementation**

  1. S3

  - Name: semantic-search-input

  2. Connect Bedrock model (embedding)

  - Model: amazon.titan-embed-text-v2:0

  3. Connect Bedrock model (LLMs)

  - Model: anthropic.claude-v2 -> anthropic.claude-3-5-haiku-20241022-v1:0

  4. DynamoDB

  - Name: semantic-search

  5. SNS

  - Topic: semantic-search-topic
  - Subscription: semantic-search-subscription

  6. Setup OpenSearch domain + index

  - Domain: semantic-search-open-search

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


  7. Lambda code for Ingest & Query

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


  11. API Gateway endpoint

  - Name: semantic-search
  - Method: Ingest, query  


* **Demo Flow**

  * Upload file → Query → Receive AI Answer
* Diagram: *System architecture (implementation-level)*
