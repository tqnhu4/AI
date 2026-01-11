import os
import boto3

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")

TABLE = os.environ.get("DYNAMODB_TABLE")
PROCESSED_BUCKET = os.environ.get("PROCESSED_BUCKET")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")

def lambda_handler(event, context):
    # event: { "documentId": ..., "indexName": ..., "processedBucket": ... }
    document_id = event.get("documentId")
    source_bucket = event.get("bucket")
    source_key = event.get("key")

    # copy object to processed bucket (optional)
    if source_bucket and source_key and PROCESSED_BUCKET:
        copy_source = {"Bucket": source_bucket, "Key": source_key}
        new_key = f"{document_id}/{source_key.split('/')[-1]}"
        s3.copy_object(Bucket = PROCESSED_BUCKET, Key = new_key, CopySource = copy_source)
        # optional: delete source
        # s3.delete_object(Bucket=source_bucket, Key=source_key)

    # update DynamoDB document status
    if TABLE:
        table = dynamodb.Table(TABLE)
        table.update_item(
            Key={
                "documentId": document_id
            },
            UpdateExpression="SET #s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": "INDEXED"}
        )


    # publish SNS to notify consumers
    if SNS_TOPIC_ARN:
        sns.publish(TopicArn = SNS_TOPIC_ARN, Message=f"Document {document_id} indexed")

    return {"documentId": document_id, "status": "finalized"}