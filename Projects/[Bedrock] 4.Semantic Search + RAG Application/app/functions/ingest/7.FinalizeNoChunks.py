import boto3
import os

sns = boto3.client("sns")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")

def lambda_handler(event, context):
    doc = event.get("documentId")
    if SNS_TOPIC_ARN:
        sns.publish(TopicArn = SNS_TOPIC_ARN, Message = f"No chunks created for {doc}")
        
    return {"documentId": doc, "status": "no_chunks"}