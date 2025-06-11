SpaCy is an open-source library designed for advanced Natural Language Processing (NLP) in Python. It is known for its speed and efficiency in handling NLP tasks in production environments. SpaCy focuses on providing pre-trained models and an easy-to-use API for common NLP tasks.

### Key Features of SpaCy
- SpaCy offers a range of powerful features for text analysis:

- Tokenization: Breaking down text into smaller units like words, punctuation marks, etc. This is the first step in most NLP pipelines.

- Part-of-Speech Tagging (POS): Assigning grammatical labels to each word (e.g., noun, verb, adjective).

- Named Entity Recognition (NER): Identifying and classifying "named entities" in text, such as names of people, organizations, locations, dates, etc.

- Dependency Parsing: Analyzing the grammatical structure of a sentence by identifying "dependency" relationships between words.

- Lemmatization: Reducing different forms of a word to its base or root form (e.g., "running", "ran", "runs" all become "run").

- Rule-based Matching: Allows you to search for patterns of words, phrases, or attributes based on custom rules.

- Word Vectors/Embeddings: Provides numerical representations of words, enabling computers to understand their semantic meaning.

- Pipeline Processing: Allows you to build complex processing sequences by combining various NLP components.

### 1.How to Use SpaCy
To use SpaCy, you need to install the library and download language models.

```text
pip install spacy

```

### 2. Download Language Models
SpaCy uses pre-trained language models to perform NLP tasks. You need to download at least one model. For example, to download the small English model:

```text
python -m spacy download en_core_web_sm
```

You can download larger models or models for other languages depending on your needs (e.g., en_core_web_md, en_core_web_lg, vi_core_news_sm).


Here are some basic examples of how to use SpaCy for key NLP tasks:

[app.py](./app.py)