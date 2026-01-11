





==========================

DELETE semantic-index

PUT semantic-index
{
  "settings": {
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024
      },
      "text": { "type": "text" },
      "metadata": { "type": "object" }
    }
  }
}


GET semantic-index/_mapping

GET semantic-index/_doc/21f46072-5096-4d4e-9435-e64a28263621__2

GET semantic-index/_search
{
  "query": {
    "match_all": {}
  },
  "size": 10,
  "from": 0,
  "_source": ["id", "text"]
}


Model	Dimension	Index name
amazon.titan-embed-text-v2	1024	semantic-index-1024
amazon.titan-embed-text-v1	1536	semantic-index-1536
OpenAI text-embedding-3-large	3072	semantic-index-3072