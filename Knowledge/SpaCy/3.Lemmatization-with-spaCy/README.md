-----

## Lemmatization with spaCy

**Lemmatization** is the process of reducing different inflected forms of a word to a common base form, known as the **lemma** or dictionary form. For example, the words "running," "runs," and "ran" all have the lemma "run." Lemmatization is crucial in NLP for tasks like text normalization, information retrieval, and text analysis, as it helps in treating different forms of the same word as a single unit.

Similar to POS tagging, spaCy automatically performs lemmatization when you process text with an `nlp` object.

### Step-by-Step Explanation

1.  **Accessing the Lemma Attribute:**
    When you process text with `nlp()`, each `Token` object in the resulting `Doc` object contains a `lemma_` attribute. This attribute stores the base form of the word.

2.  **Iterating and Printing:**
    You can iterate through the `doc` object to access each `token` and then print its `token.text` (the original word) and its `token.lemma_` (the base form).

### Full Code Example

Let's use our previous `doc` object (from the text "I love programming in Python.") to demonstrate lemmatization:

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Process the text to create a Doc object
doc_text = "I love programming in Python."
doc = nlp(doc_text)

# Iterate through the Doc object and print each token along with its lemma
print("Token | Lemma")
print("-" * 25)
for token in doc:
    print(f"{token.text:<11} | {token.lemma_}")
```

**Output of the code:**

```
Token       | Lemma
-------------------------
I           | I
love        | love
programming | program
in          | in
Python      | Python
.           | .
```

In this example, "programming" is lemmatized to "program." Notice that words already in their base form (like "I," "love," "in," "Python," and ".") remain unchanged.

-----

Would you like to explore **Named Entity Recognition (NER)** or another spaCy feature next?