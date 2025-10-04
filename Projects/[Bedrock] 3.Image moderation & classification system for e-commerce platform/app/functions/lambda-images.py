import json
import os
import logging
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "us-east-1")
DDB_TABLE = os.environ.get("DDB_TABLE", "image-metadata-table")

dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(DDB_TABLE)

class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o) 
        return super(DecimalEncoder, self).default(o)

#GET /images?status=APPROVED
def lambda_handler(event, context):
    #status = event["queryStringParameters"].get("status", "DONE")  # optional
    query = event.get("queryStringParameters") or {}
    status = query.get("status", "DONE")


    resp = table.query(
        IndexName="StatusIndex",
        KeyConditionExpression=Key("final_status").eq(status)
    )

    return {
        "statusCode": 200,
        "body": json.dumps(resp["Items"], cls=DecimalEncoder)
    }
