-----

## Named Entity Recognition (NER) with spaCy

** Named Entity Recognition (NER)** is an NLP task that aims to locate and classify named entities in text into pre-defined categories such as person names, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. It's a crucial step for information extraction and understanding who, what, when, and where something happened in a text.

spaCy's language models are pre-trained to perform NER automatically when you process text.

### Step-by-Step Explanation

1.  **Accessing Named Entities:**
    After processing text with `nlp()`, the `Doc` object contains an `ents` attribute. This attribute is a **tuple of `Span` objects**, where each `Span` represents a detected named entity.

2.  **Accessing Entity Attributes:**
    Each `Span` object (representing an entity) has important attributes:

      * `ent.text`: The actual text of the named entity (e.g., "Google", "New York").
      * `ent.label_`: The label of the entity, indicating its type (e.g., `ORG` for organization, `GPE` for geopolitical entity, `PERSON` for person).

3.  **Iterating and Printing:**
    You can iterate through the `doc.ents` collection to access each detected entity and then print its text and label.

### Full Code Example

Let's use a slightly more diverse text to showcase NER effectively:

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define a new text with various named entities
doc_text = "Apple Inc. announced new iPhone models today in California. Tim Cook spoke about the latest features."

# Process the text to create a Doc object
doc = nlp(doc_text)

# Iterate through the detected entities and print their text and label
print("Entity Text | Entity Type")
print("-" * 30)
for ent in doc.ents:
    print(f"{ent.text:<15} | {ent.label_}")
```

**Output of the code:**

```
Entity Text | Entity Type
------------------------------
Apple Inc.      | ORG
today           | DATE
California      | GPE
Tim Cook        | PERSON
```

In this output:

  * "Apple Inc." is recognized as an **ORG** (Organization).
  * "today" is recognized as a **DATE**.
  * "California" is recognized as a **GPE** (Geopolitical Entity - typically countries, cities, states).
  * "Tim Cook" is recognized as a **PERSON**.

This demonstrates how spaCy can automatically identify and classify different types of named entities within your text, which is incredibly useful for tasks like information extraction, content categorization, and building knowledge graphs.

-----
