
import os
import boto3, json
import requests
from requests.auth import HTTPBasicAuth


bedrock = boto3.client("bedrock-runtime")
BEDROCK_EMBED_MODEL = os.environ.get("BEDROCK_EMBED_ID", "amazon.titan-embed-text-v1")
BEDROCK_LLM_MODEL = os.environ.get("BEDROCK_LLM_ID", "anthropic.claude-v2")
DOCUMENT_INDEX = os.environ.get("DOCUMENT_INDEX", "semantic-index")
OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT")
awsauth = HTTPBasicAuth(os.environ['OPENSEARCH_USER'], os.environ['OPENSEARCH_PASS'])

def lambda_handler(event, context):
    body = json.loads(event["body"])
    query_text = body["query"]
    
    # Step 1: Create embedding for query

    
    response = bedrock.invoke_model(
        modelId = BEDROCK_EMBED_MODEL,
        body = json.dumps({"inputText": query_text})
    )

    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]    
    
    # Step 2: Query OpenSearch (vector kNN)

    query_body = {
        "size": body.get("k", 5),
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": body.get("k", 5)
                }
            }
        }
    }

    url = f"{OPENSEARCH_ENDPOINT}/{DOCUMENT_INDEX}/_search"
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, auth=awsauth, json=query_body, headers=headers)
    resp_json = resp.json() 

    chunks = [hit["_source"] for hit in resp_json["hits"]["hits"]]

    # Step 3: Create prompt for Bedrock RAG
    context_text = "\n\n".join([c["text"] for c in chunks])
    

    # Step 4: Call Bedrock model (Claude, Titan, v.v.)
    #response = bedrock.invoke_model(
    #  modelId = BEDROCK_LLM_MODEL,
    #  body = json.dumps({"prompt": prompt, "max_tokens_to_sample": 500})
    #)
    
    response = bedrock.invoke_model(
        modelId=BEDROCK_LLM_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "temperature": 0.5,
            "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Context:\n{context_text}\n\nQuestion: {query_text}\nAnswer concisely and cite sources."}],
            }
            ]
        })
    )
    
    answer = json.loads(response["body"].read())["content"][0]['text']
    #answer = json.loads(response["body"].read())["outputText"]

    return {
        "statusCode": 200,
        #"body": json.dumps({"answer": answer, "sources": chunks})
        "body": json.dumps({"answer": answer})
    }
