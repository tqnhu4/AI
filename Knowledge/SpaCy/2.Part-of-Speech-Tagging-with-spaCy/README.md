
## Part-of-Speech Tagging with spaCy

Let's continue with the `doc` object we created in the previous step (from the text "I love programming in Python."):

### Step-by-Step Explanation

1.  **Accessing Token Attributes:**
    As we saw, when you process text with `nlp()`, spaCy creates a `Doc` object containing `Token` objects. Each `Token` object has several useful attributes. For POS tagging, we're interested in:

      * `token.text`: The original text of the token.
      * `token.pos_`: The **coarse-grained part-of-speech tag**. This is a simplified tag (e.g., `NOUN`, `VERB`, `ADJ`). It's generally more abstract and easier to understand.
      * `token.tag_`: The **fine-grained part-of-speech tag**. This provides more specific details based on the Universal Dependencies or Penn Treebank tag set (e.g., `NN` for singular noun, `VBP` for non-3rd person singular present verb).

2.  **Iterating and Printing Tags:**
    By iterating through the `doc` object, we can access each `token` and print these attributes.

### Full Code Example

Building on the previous example:

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Process the text to create a Doc object
doc_text = "I love programming in Python."
doc = nlp(doc_text)

# Iterate through the Doc object and print each token along with its POS tags
print("Token | Coarse-grained POS | Fine-grained POS")
print("-" * 50)
for token in doc:
    print(f"{token.text:<5} | {token.pos_:<18} | {token.tag_}")
```

**Output of the code:**

```
Token | Coarse-grained POS | Fine-grained POS
--------------------------------------------------
I     | PRON               | PRP
love  | VERB               | VBP
programming | VERB               | VBG
in    | ADP                | IN
Python | PROPN              | NNP
.     | PUNCT              | .
```

As you can see, for each token, spaCy has identified its part of speech. For instance, "I" is a pronoun (`PRON`), "love" is a verb (`VERB`), and "Python" is a proper noun (`PROPN`).

-----

Would you like to explore another spaCy feature, such as **Named Entity Recognition (NER)**?