-----

## Dependency Parsing with spaCy

**Dependency Parsing** is a method in NLP that analyzes the grammatical structure of a sentence by showing the relationships between "head" words and the words that modify or depend on them. Essentially, it identifies how words in a sentence are syntactically related to each other. This helps in understanding the subject, object, and other modifiers within a sentence.

spaCy automatically performs dependency parsing as part of its standard NLP pipeline.

### Step-by-Step Explanation

When you process text with `nlp()`, spaCy generates a dependency parse tree for the sentence. Each `Token` object in the `Doc` object will have attributes related to its dependency relationship:

1.  **`token.text`**: This is the current word (token) we are examining.
2.  **`token.dep_`**: This attribute holds the **dependency label** (or relationship type) of the current token to its head. Common dependency labels include `nsubj` (nominal subject), `dobj` (direct object), `amod` (adjectival modifier), etc.
3.  **`token.head.text`**: This attribute points to the **head word** of the current token. The head is the word that the current token depends on, or that it modifies.

By iterating through the `doc` object, you can access these attributes for each token and print out the dependency relationship.

### Full Code Example

Let's use an example sentence to illustrate dependency parsing:

```python
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define a sentence for dependency parsing
doc_text = "Apple bought a small company in London."

# Process the text to create a Doc object
doc = nlp(doc_text)

# Iterate through each token and print its text, dependency label, and its head word
print("Token    -> Dependency -> Head Word")
print("-" * 40)
for token in doc:
    print(f"{token.text:<8} -> {token.dep_:<10} -> {token.head.text}")
```

**Output of the code:**

```
Token    -> Dependency -> Head Word
----------------------------------------
Apple    -> nsubj      -> bought
bought   -> ROOT       -> bought
a        -> det        -> company
small    -> amod       -> company
company  -> dobj       -> bought
in       -> prep       -> bought
London   -> pobj       -> in
.        -> punct      -> bought
```

### Understanding the Output

Let's break down a few lines from the output:

  * **`Apple -> nsubj -> bought`**: This means "Apple" is the **nominal subject** (`nsubj`) of the verb "bought".
  * **`bought -> ROOT -> bought`**: The word "bought" is the **root** of the sentence, meaning it's the main verb around which other words are structured. A word is its own head if it's the root.
  * **`small -> amod -> company`**: "small" is an **adjectival modifier** (`amod`) of "company".
  * **`company -> dobj -> bought`**: "company" is the **direct object** (`dobj`) of the verb "bought".
  * **`in -> prep -> bought`**: "in" is a **preposition** (`prep`) whose head is "bought" (it forms a prepositional phrase modifying the action of buying).
  * **`London -> pobj -> in`**: "London" is the **object of the preposition** (`pobj`) "in".

Dependency parsing is a powerful tool for understanding sentence structure and can be invaluable for tasks like relation extraction, question answering, and improving search relevance.
