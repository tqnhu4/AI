import boto3, json, uuid

s3 = boto3.client('s3')
OUTPUT_BUCKET = "semantic-search-processed"

def lambda_handler(event, context):
    # chunks
    chunks = ["chunk1 text...", "chunk2 text...", "..."]  

    # S3
    file_id = str(uuid.uuid4())
    output_key = f"chunks/{file_id}.json"
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=output_key,
        Body=json.dumps(chunks)
    )

    # metadata
    return {
        "output_bucket": OUTPUT_BUCKET,
        "output_key": output_key,
        "num_chunks": len(chunks)
    }
