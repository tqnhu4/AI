import json
import os
import logging
import boto3
from decimal import Decimal
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-text-express-v1")  
DDB_TABLE = os.environ.get("DDB_TABLE", "image-metadata-table")

rek = boto3.client("rekognition", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
dynamodb = boto3.resource("dynamodb", region_name=REGION)
table = dynamodb.Table(DDB_TABLE)

rules = ["nudity", "sexual", "violence", "drugs", "alcohol"]


def call_rekognition(bucket, key):
    """Call Rekognition for moderation, OCR, and labels."""
    s3_obj = {"S3Object": {"Bucket": bucket, "Name": key}}
    result = {}

    # 1. Moderation labels
    try:
        mod = rek.detect_moderation_labels(Image=s3_obj)
        result["moderation_labels"] = mod.get("ModerationLabels", [])
    except ClientError as e:
        logger.exception("Rekognition DetectModerationLabels failed")
        result["moderation_labels"] = []

    # 2. Detect text (OCR)
    try:
        text_resp = rek.detect_text(Image=s3_obj)
        # Collect unique detected strings
        texts = [t.get("DetectedText") for t in text_resp.get("TextDetections", []) if t.get("Type") == "LINE"]
        result["ocr_texts"] = texts
    except ClientError:
        logger.exception("Rekognition DetectText failed")
        result["ocr_texts"] = []

    # 3. Detect labels (object detection)
    try:
        labels_resp = rek.detect_labels(Image=s3_obj, MaxLabels=20, MinConfidence=50)
        # convert to list of (name, confidence)
        labels = [{"Name": l["Name"], "Confidence": l["Confidence"]} for l in labels_resp.get("Labels", [])]
        result["labels"] = labels
    except ClientError:
        logger.exception("Rekognition DetectLabels failed")
        result["labels"] = []

    return result


def call_bedrock_map_to_product(labels, ocr_texts, context_catalog=None):
    """
    Call Bedrock to map Rekognition outputs + OCR text into a single product label,
    and also ask for a 'match_score' (0-100).
    """
    prompt = {
        "inputText": (
            "You are an assistant that maps image detection results to a product label for an e-commerce catalog.\n\n"
            "Detected objects: {}\n"
            "Detected text in image: {}\n\n"
            "Given common e-commerce categories (Clothing, Electronics, Cosmetics, Food, Accessories), "
            "return a JSON with keys: product_label (short string), category, reasons (list), match_score (0-100).\n\n"
            "Be concise and only return valid JSON."
        ).format([l["Name"] for l in labels], ocr_texts)
    }

    # build body bytes per Bedrock invoke_model requirement
    body = json.dumps(prompt).encode("utf-8")

    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        # read response bytes
        payload = resp.get("body").read()
        text_response = payload.decode("utf-8")
        # Bedrock may return JSON or plain text; try parse JSON
        try:
            parsed = json.loads(text_response)
            return parsed
        except Exception:
            # If not JSON, return textual answer in 'explanation' field
            return {"product_label": None, "category": None, "reasons": [text_response], "match_score": 0}
    except ClientError:
        logger.exception("Bedrock invoke_model failed")
        return {"product_label": None, "category": None, "reasons": [], "match_score": 0}


def compute_final_status(moderation_labels, bedrock_result, threshold_safe=50):
    """
    Decide if image is allowed:
    - If any moderation label confidence high for disallowed types => reject.
    - Otherwise accept and use bedrock_result.match_score for ranking.
    """
    # simple rules: if Rekognition reports 'Explicit Nudity' or 'Violence' over 60% -> reject
    reject_reasons = []
    for ml in moderation_labels:
        name = ml.get("Name", "").lower()
        conf = ml.get("Confidence", 0)
        if any(x in name for x in rules) and conf >= 60:
            reject_reasons.append({"label": ml.get("Name"), "confidence": conf})

    status = "REJECTED" if reject_reasons else "APPROVED"
    score = bedrock_result.get("match_score", 0)
    return {"status": status, "reject_reasons": reject_reasons, "score": score}


def save_metadata_to_dynamodb(bucket, key, rek_result, bedrock_result, final_status):
    """Persist result summary to DynamoDB."""
    labels = rek_result.get("labels", [])
    for item in labels:
        if "Confidence" in item:
            item["Confidence"] = Decimal(str(item["Confidence"]))

    moderation_labels = rek_result.get("moderation_labels", [])
    for item in moderation_labels:
        if "Confidence" in item:
            item["Confidence"] = Decimal(str(item["Confidence"]))            

    item = {
        "s3_key": f"{bucket}/{key}",
        #"labels": rek_result.get("labels", []),
        "labels": labels,
        "ocr_texts": rek_result.get("ocr_texts", []),
        #"moderation": rek_result.get("moderation_labels", []),
        "moderation": moderation_labels,
        "bedrock": bedrock_result,
        "final_status": final_status.get("status", []),
        "timestamp": int(__import__("time").time())
    }
    try:
        table.put_item(Item=item)
    except ClientError:
        logger.exception("DynamoDB put_item failed")


def lambda_handler(event, context):
    """
    Expected: event from S3 put (ObjectCreated).
    """
    logger.info("Event: %s", json.dumps(event))
    # get bucket/key
    rec = event["Records"][0]
    bucket = rec["s3"]["bucket"]["name"]
    key = rec["s3"]["object"]["key"]

    # 1. Rekognition analyses
    rek_result = call_rekognition(bucket, key)

    # 2. Bedrock: map to product label
    bedrock_result = call_bedrock_map_to_product(rek_result.get("labels", []), rek_result.get("ocr_texts", []))

    # 3. Decide final status and compute score
    final_status = compute_final_status(rek_result.get("moderation_labels", []), bedrock_result)

    # 4. Save metadata to DynamoDB
    save_metadata_to_dynamodb(bucket, key, rek_result, bedrock_result, final_status)

    # 5. Return summary
    summary = {
        "s3_key": f"{bucket}/{key}",
        "status": final_status["status"],
        "product_label": bedrock_result.get("product_label"),
        "category": bedrock_result.get("category"),
        "score": final_status["score"],
        "reject_reasons": final_status["reject_reasons"]
    }
    logger.info("Summary: %s", json.dumps(summary))
    return {"statusCode": 200, "body": json.dumps(summary)}