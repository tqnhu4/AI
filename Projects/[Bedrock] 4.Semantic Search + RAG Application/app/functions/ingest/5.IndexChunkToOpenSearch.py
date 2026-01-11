import os
import json
import boto3
import requests
#from requests_aws4auth import AWS4Auth
from requests.auth import HTTPBasicAuth

region = os.environ.get("AWS_REGION", "us-east-1")
opensearch_endpoint = os.environ.get("OPENSEARCH_ENDPOINT")
index_name = os.environ.get("OPENSEARCH_INDEX")

session = boto3.Session()
credentials = session.get_credentials().get_frozen_credentials()
#awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
#auth = HTTPBasicAuth(os.environ['OPENSEARCH_USER'], os.environ['OPENSEARCH_PASS'])
awsauth = HTTPBasicAuth(os.environ['OPENSEARCH_USER'], os.environ['OPENSEARCH_PASS'])
#awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'aoss', session_token=credentials.token)
def index_document(doc_id, body):
    url = f"{opensearch_endpoint}/{index_name}/_doc/{doc_id}"
    headers = {"Content-Type": "application/json"}
    r = requests.put(url, auth = awsauth, json = body, headers = headers)

    if not r.ok:
        print("OpenSearch Error:", r.text)    
        
    r.raise_for_status()
    return r.json()

def lambda_handler(event, context):
    # event contains chunk, embedding, metadata
    chunk = event.get("chunk") or {}
    embedding = event.get("embedding") or []
    metadata = event.get("metadata") or chunk.get("metadata", {})

    doc_id = chunk.get("id") or metadata.get("chunkId")
    body = {
        "text": chunk.get("text"),
        "metadata": metadata,
        "embedding": embedding
    }
    resp = index_document(doc_id, body)
    
    return {"indexed": True, "response": resp}