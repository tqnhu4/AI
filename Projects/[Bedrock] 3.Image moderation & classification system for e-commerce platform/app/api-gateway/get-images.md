---

## Query by Status 

```http
GET /images?status=APPROVED

* **Case 1: Using GSI with Sort Key = timestamp** 

**Example Lambda code (assuming a `StatusIndex` with Sort Key = timestamp):**

```python
# Code placeholder
```

---

## API Gateway Integration

* **Method:** `GET`
* **Integration Type:** Lambda Proxy
* **CORS:** Enabled (if accessed from a web frontend)
* **Authentication Options:**

  * **IAM Authentication** (for service-to-service communication).
  * **Cognito JWT Authentication** (for end users on web/app).

---

## Query Patterns

* **GetItem:** Retrieve an image by `s3_key`.
* **Query by Status:** Fetch images by processing status (`APPROVED`, `REJECTED`).
* **Query by Time Range:** Filter images by upload or processing timestamp.

---
