
---

## 🎙️ **YouTube Presentation Script — “Semantic Search + RAG Application (Bedrock + OpenSearch)”**

---

### 🧩 **#1. What is Semantic Search?**

> “Let’s start with the core concept — Semantic Search.”

Traditional search just matches **keywords**.
But **Semantic Search** understands **meaning** — it finds results that are *conceptually* similar, not just textually similar.

Here’s how it works:

> 🔍 Input query
> ⬇️
> 🧠 Encode to vector
> ⬇️
> 📊 Compare similarity with data vectors
> ⬇️
> 🎯 Return the most semantically relevant results

So when you ask a question, instead of matching exact words, it finds the content that **means the same thing** — that’s the magic of embeddings.

---

### 🧠 **#2. What is RAG (Retrieval-Augmented Generation)?**

> “Now, let’s talk about RAG — short for Retrieval-Augmented Generation.”

Imagine you have a smart assistant that can generate text, like ChatGPT —
but it doesn’t know about your private documents or recent data.

RAG fixes that.

Here’s the flow:

> 💬 Input query
> ⬇️
> 📚 Retrieve relevant information
> ⬇️
> 🧠 Feed into a language model
> ⬇️
> ✨ Generate an answer based on data and context

So, RAG combines **retrieval** (searching for facts)
and **generation** (producing natural language responses).

➡️ And remember: **Semantic Search** is the *retrieval brain* of RAG —
it helps the model find and use **relevant knowledge**,
so answers become factual and context-aware instead of vague.

---

### 🚀 **#3. What We Will Build**

> “Let’s see what we’re going to build together.”

Our application has two main functions:

1. **Ingest** — upload documents → split into chunks → generate embeddings → store in OpenSearch
2. **Query** — user sends a query → system retrieves semantic context → sends to Bedrock → generates an answer

We’ll build this entire flow using **AWS services**.

Our **system overview** includes:

* **S3** for document storage
* **Bedrock** for embeddings and LLMs
* **OpenSearch** for vector search
* **Lambda** for workflow logic
* **API Gateway** as the public entry point

This is the architecture behind our *“Semantic Search + RAG Application.”*

---

### ⚙️ **#4. Workflow in Action**

> “Let’s see how the workflow actually runs.”

#### **Step 1: Ingest Documents**

We start from document upload — that’s handled through **API Gateway**,
which triggers a series of **Lambda functions** and **Step Functions**:

Stage 1:

> Upload → Extract → Chunk → Generate Embedding (via Bedrock) → Store to OpenSearch

This is our **Document Ingestion Pipeline**.
It transforms your raw files into searchable semantic vectors.

#### **Step 2: Query + RAG**

When a user sends a query:

> Query → Embedding → Search in OpenSearch → Retrieve top results → Send to Bedrock LLM → Generate response

That’s our **RAG Query Pipeline** —
it finds the best-matching context and uses the LLM to produce an answer that’s both *relevant* and *informative*.

**Main AWS Services in Action:**

| Service       | Role               | Detail                           |
| ------------- | ------------------ | -------------------------------- |
| Bedrock (LLM) | Generate responses | Use Claude or Titan models       |
| OpenSearch    | Store embeddings   | Vector index + cosine similarity |
| Lambda        | Processing logic   | Orchestrates workflow steps      |
| S3            | Document source    | Raw data for ingestion           |
| API Gateway   | Entry point        | Connects client and Lambda       |

---

### 💻 **#5. Setup the App**

> “Now let’s walk through the setup, step by step.”

**1. Create an S3 bucket**

* Name it: `semantic-search-input`

**2. Connect the Bedrock embedding model**

* Use: `amazon.titan-embed-text-v2:0`

**3. Connect the Bedrock LLM**

* Start with: `anthropic.claude-v2`
* Then upgrade to: `anthropic.claude-3-5-haiku-20241022-v1:0`

**4. Create a DynamoDB table**

* Name: `semantic-search`

**5. Create an SNS Topic**

* Topic: `semantic-search-topic`
* Subscription: `semantic-search-subscription`

**6. Set up OpenSearch domain and index**

Example index mapping:

```json
PUT semantic-index
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": { "type": "knn_vector", "dimension": 1024 },
      "text": { "type": "text" },
      "metadata": { "type": "object" }
    }
  }
}
```

**7. Deploy Lambda functions**

* Ingest pipeline:
  `ValidateS3Event`, `GetObjectMetadata`, `ExtractAndChunk`,
  `GenerateEmbedding`, `IndexChunkToOpenSearch`,
  `FinalizeIngest`, `FinalizeNoChunks`

* Query pipeline:
  `Ingest`, `QueryHandler`

**8. Assign IAM Roles and Policies**
Each Lambda gets its own Role & Policy —
for example, `ValidateS3EventRole`, `GenerateEmbeddingPolicy`, and so on.

**9. Set up Step Functions**

* Name: `semantic-search`
  → This orchestrates all the ingest steps in sequence.

**10. Configure EventBridge**

* Rule: `S3-StepFunctions`
  → Automatically triggers your Step Function whenever a new file is uploaded.

**11. API Gateway**

* Name: `semantic-search`
* Methods: `/ingest`, `/query`

And that’s it — you now have a full **RAG application pipeline**,
from uploading documents to generating context-aware answers!

---

### 🎯 **Conclusion**

> “So to summarize…”

We combined **Semantic Search** and **RAG**, powered by **Amazon Bedrock + OpenSearch**,
to create a system that can:

* Understand your data,
* Retrieve the most relevant information,
* And generate precise, contextual responses.

This architecture is **scalable**, **serverless**, and **AI-ready** —
perfect for enterprise knowledge systems or intelligent chatbots.

---
