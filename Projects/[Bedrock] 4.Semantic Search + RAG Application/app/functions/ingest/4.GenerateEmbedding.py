import os
import boto3
import json

REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL_ID"")

bedrock = boto3.client("bedrock-runtime", region_name = REGION)

def call_bedrock_embedding(text):
    # Example: model invocation shape may differ — adapt to your Bedrock model API
    prompt = {"inputText": text}
    resp = bedrock.invoke_model(
        modelId = BEDROCK_MODEL,
        contentType = "application/json",
        accept = "application/json",
        body = json.dumps(prompt)
    )
    body = resp['body'].read()
    parsed = json.loads(body)
    # parsed shape depends on model — adapt accordingly
    # assume parsed = {"embedding": [float,...]}
    embedding = parsed.get("embedding")
    if embedding is None:
        # fallback: maybe model returns embeddings under different key
        embedding = parsed
    return embedding

def lambda_handler(event, context):
    # event is one Map item: { "chunk": {...}, "documentId": "..." } depending on Step Fn config
    chunk = event.get("chunk") or event.get("chunkData") or event
    text = chunk.get("text") if isinstance(chunk, dict) else str(chunk)
    chunk_id = chunk.get("id") if isinstance(chunk, dict) else None

    embedding = call_bedrock_embedding(text)
    
    return {
        "chunkId": chunk_id,
        "embedding": embedding
    }