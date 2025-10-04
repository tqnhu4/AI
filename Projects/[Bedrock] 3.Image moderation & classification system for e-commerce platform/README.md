## 📌 Project: Image Moderation & Classification System for E-commerce

### 🛒 Context

An e-commerce platform allows sellers to **upload product images**. However:

* Some images are invalid (contain competitor logos, sensitive text, violent content, or models wearing inappropriate clothing).
* Some images contain text (e.g., "SALE 70%", "Authentic Product") that needs to be analyzed for marketing purposes.
* Images need to be categorized into **product groups** (clothing, phones, cosmetics, food, etc.) to support automatic tagging.

---
### 🎯 Requirements

1. **Image Moderation**

   * Detect inappropriate content (violence, nudity, drugs, alcohol, etc.).
   * Reject images containing banned brand logos.

2. **Text Recognition in Images (OCR)**

   * Extract all text from images (e.g., "SALE", "100% Organic").
   * Analyze text meaning to detect spam or excessive promotions.

3. **Product Image Classification**

   * Use Rekognition to detect the main object in the image.
   * Combine with Bedrock to **interpret the object → assign product labels** (e.g., “men’s shirt,” “smartphone,” “lipstick”).

4. **Returned Results**

   * Image status: ✅ Valid / ❌ Invalid.
   * Product group: (Clothing, Electronics, Cosmetics, etc.).
   * Metadata text extracted from the image.

---

### 🔧 Technology Solution

* **AWS S3** → store uploaded images.
* **AWS Lambda** → process when a new image is uploaded.
* **AWS Rekognition** →
  * `DetectModerationLabels` (image moderation).
  * `DetectText` (OCR).
  * `DetectLabels` (object detection).
* **AWS Bedrock** →
  * Semantic analysis of OCR text.
  * Interpret detected objects into product labels aligned with the catalog.
* **AWS DynamoDB** → store result metadata.
* **API Gateway** → return results to web/app.

---

### 🚀 Workflow

1. Seller uploads image → S3.
2. Lambda triggers Rekognition:

   * Image moderation.
   * OCR text extraction.
   * Object detection.
3. Send results to Bedrock:

   * Interpret text to detect spam/promotions.
   * Map objects into valid product labels.
4. Store metadata → DynamoDB.
5. API Gateway returns results to the admin system.

---

### 📋 Example Output

Uploaded image: `somi123.jpg`

* **Moderation result**: ✅ Valid
* **OCR Text**: `"SALE 50% Office Shirt"`
* **Product Label (Bedrock)**: `"Men’s Shirt"`
* **Final Status**: Approved ✅

---

## 👉 Application deployment process

---

### Step 1: Create an S3 Bucket

* **Bucket Name:** `image-input`
* **Event Configuration:** Configure the bucket to trigger a Lambda function whenever a new image is uploaded.

---

### Step 2: Create a Lambda Function for Image Processing

* **Function Name:** `lambda-moderation-classification`
* **Purpose:** Process uploaded images (moderation, OCR, classification).
* **IAM Role:** Assign a role with the necessary permissions (S3 read access, Rekognition, Bedrock, DynamoDB, CloudWatch logging).

---

### Step 3: Configure Bedrock

* **Model Selection:** Use Amazon Titan model `amazon.titan-text-express-v1`.
* **Purpose:** Perform text analysis (OCR interpretation, spam/promotion detection, object-to-product label mapping).
* **Setup:** Request access to the model if required.

---

### Step 4: Create DynamoDB Table

* **Table Name:** `image-metadata-table`
* **Schema Definition:**

```schema
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
```

* **Purpose:** Store metadata results for moderation, OCR text, product labels, and final status.

---

### Step 5: Configure CloudWatch

* **Setup:** Ensure the Lambda function writes logs to a CloudWatch log group.
* **Purpose:** Monitor execution, debug errors, and validate processing during tests.

---

### Step 6: Create API Gateway for Retrieval

* **Endpoint:** `GET /images?status=APPROVED`
* **Lambda Integration:** Create a Lambda function that queries DynamoDB based on `final_status`.
* **IAM Role:** Assign a role with permission to query DynamoDB and log to CloudWatch.
* **Purpose:** Provide an API for the web or admin system to fetch approved images.

---

### Suggestions for Production Expansion

* Add a queue (SQS) if handling a large number of images or to prevent bottlenecks.
* Implement retry/backoff mechanisms for Bedrock/Rekognition calls.
* Add a whitelist/blacklist logo check (using image matching or embedding + similarity).
* Provide additional APIs (API Gateway + Lambda) to query image lists by score or with filters.
* Record events into an audit log and visualize them in a dashboard (CloudWatch / QuickSight).

---
[![Watch the video](https://img.youtube.com/vi/HDzT4YVIO-o/0.jpg)](https://www.youtube.com/watch?v=HDzT4YVIO-o)
