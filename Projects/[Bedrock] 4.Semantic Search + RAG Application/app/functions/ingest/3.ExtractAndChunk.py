import os
import boto3
import re
import uuid

s3 = boto3.client("s3")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "500"))
CHUNK_SENTENCES = int(os.environ.get("CHUNK_SENTENCES", "5"))

def naive_text_extractor(bucket, key):
    # Basic: for text/plain or small PDFs as fallback
    resp = s3.get_object(Bucket = bucket, Key = key)
    body = resp['Body'].read()
    ct = resp.get('ContentType', '')
    # NOTE: production: use PDF parser or Tika for PDFs and Office docs
    try:
        text = body.decode('utf-8', errors='replace')
    except:
        text = str(body)
    return text, ct

def sentence_split(text):
    # naive split on .,!? and newlines
    sents = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    return [s.strip() for s in sents if s.strip()]

def chunk_sentences(sentences, n = CHUNK_SENTENCES):
    chunks = []
    for i in range(0, len(sentences), n):
        part = sentences[i:i+n]
        chunks.append(" ".join(part))
        
    return chunks

def lambda_handler(event, context):
    # event contains bucket,key,documentId
    bucket = event['bucket']
    key = event['key']
    document_id = event.get('documentId') or str(uuid.uuid4())

    text, content_type = naive_text_extractor(bucket, key)
    sents = sentence_split(text)
    if not sents:
        return {"documentId": document_id, "chunks": []}

    texts = chunk_sentences(sents)
    result_chunks = []
    for idx, t in enumerate(texts):
        chunk_id = f"{document_id}__{idx}"
        result_chunks.append({
            "id": chunk_id,
            "text": t,
            "metadata": {
                "sourceBucket": bucket,
                "sourceKey": key,
                "contentType": content_type,
                "chunkIndex": idx
            }
        })

    return {"documentId": document_id, "chunks": result_chunks, "chunkCount": len(result_chunks)}