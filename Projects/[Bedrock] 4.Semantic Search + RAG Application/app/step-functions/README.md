{
  "Comment": "State machine for semantic-search ingestion: S3 -> chunk -> embeddings -> index",
  "StartAt": "ValidateEvent",
  "States": {
    "ValidateEvent": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:ValidateS3Event",
      "InputPath": "$",
      "ResultPath": "$.validated",
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "Next": "Failure"
        }
      ],
      "Next": "GetObjectMetadata"
    },
    "GetObjectMetadata": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:GetObjectMetadata",
      "ResultPath": "$.metadata",
      "Next": "ExtractAndChunk",
      "Retry": [
        {
          "ErrorEquals": ["Lambda.ServiceException", "Lambda.AWSLambdaException", "States.TaskFailed"],
          "IntervalSeconds": 2,
          "MaxAttempts": 3,
          "BackoffRate": 2.0
        }
      ]
    },
    "ExtractAndChunk": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:ExtractAndChunk",
      "Parameters": {
        "bucket.$": "$.validated.bucket",
        "key.$": "$.validated.key",
        "maxTokens": 500,
        "chunkSizeSentences": 5
      },
      "ResultPath": "$.chunksResult",
      "Next": "IfNoChunks"
    },
    "IfNoChunks": {
      "Type": "Choice",
      "Choices": [
        {
          "And": [
            {
              "Variable": "$.chunksResult.chunks",
              "IsPresent": true
            },
            {
              "Variable": "$.chunksResult.chunkCount",
              "NumericGreaterThan": 0
            }
          ],
          "Next": "DistributedMapChunks"
        }
      ],
      "Default": "FinalizeNoChunks"
    },
    "DistributedMapChunks": {
      "Type": "Map",
      "InputPath": "$.chunksResult",
      "ItemsPath": "$.chunks",
      "MaxConcurrency": 5,
      "Retry": [
        {
          "ErrorEquals": ["Lambda.TooManyRequestsException"],
          "IntervalSeconds": 2,
          "MaxAttempts": 5,
          "BackoffRate": 2.0
        }
      ],      
      "ResultPath": "$.mapResult",
      "Iterator": {
        "StartAt": "GenerateEmbedding",
        "States": {
          "GenerateEmbedding": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:GenerateEmbedding",
            "Parameters": {
              "chunk.$": "$",
              "documentId.$": "$$.Map.Item.Value..documentId"
            },
            "ResultPath": "$.embeddingResult",
            "Next": "IndexChunk"
          },
          "IndexChunk": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:IndexChunkToOpenSearch",
            "Parameters": {
              "indexName": "your-opensearch-index",
              "chunk.$": "$",
              "embedding.$": "$.embeddingResult.embedding",
              "metadata.$": "$.metadata"
            },
            "End": true,
            "Retry": [
              {
                "ErrorEquals": ["OpenSearch.ServiceException", "States.TaskFailed"],
                "IntervalSeconds": 2,
                "MaxAttempts": 4,
                "BackoffRate": 2.0
              }
            ]
          }
        }
      },
      "Next": "Finalize"
    },
    "FinalizeNoChunks": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:FinalizeNoChunks",
      "End": true
    },
    "Finalize": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:REGION:ACCOUNT:function:FinalizeIngest",
      "Parameters": {
        "documentId.$": "$.chunksResult.documentId",
        "indexName": "your-opensearch-index",
        "processedBucket": "your-processed-bucket"
      },
      "End": true
    },
    "Failure": {
      "Type": "Fail",
      "Cause": "IngestionFailed"
    }
  }
}
