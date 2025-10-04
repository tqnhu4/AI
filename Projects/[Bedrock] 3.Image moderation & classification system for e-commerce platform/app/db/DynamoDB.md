TableName: image-metadata-table
PrimaryKey:
  PartitionKey: s3_key (String)
  SortKey: timestamp (Number)   # optional
Attributes:
  - s3_key: String
  - labels: List<String>
  - ocr_texts: List<String>
  - moderation: List<String>
  - bedrock: Map
  - final_status: String
  - timestamp: Number

GSIs:
  - Name: StatusIndex
    PartitionKey: final_status
    SortKey: timestamp
