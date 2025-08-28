
## Text Tokenization with spaCy

**Tokenization** is the process of breaking down a text into smaller units called **tokens**. These tokens can be words, punctuation marks, numbers, or even parts of words. It's often the first step in many Natural Language Processing (NLP) tasks, as it helps structure the text for further analysis.

We'll use **spaCy**, a popular and efficient NLP library in Python, for this task.

### Step-by-Step Explanation

1.  **Import spaCy:**
    First, you need to import the spaCy library.

    ```python
    import spacy
    ```

2.  **Load a Language Model:**
    spaCy uses pre-trained language models to process text. For English, a common small model is `en_core_web_sm`. You need to load this model to create an `nlp` object, which is essentially a processing pipeline.
    *(If you haven't installed the model yet, you'll need to run `python -m spacy download en_core_web_sm` in your terminal.)*

    ```python
    nlp = spacy.load("en_core_web_sm")
    ```

3.  **Process the Text:**
    Pass your text string to the `nlp` object. This processes the text through the loaded pipeline (including tokenization, part-of-speech tagging, etc.) and returns a **Doc object**. The `Doc` object is a container for all the processed information about your text, including its tokens.

    ```python
    text = "I love programming in Python."
    doc = nlp(text)
    ```

4.  **Iterate and Print Tokens:**
    The `Doc` object behaves like a sequence of `Token` objects. You can iterate over it to access each individual token. Each `token` object has various attributes, and `token.text` gives you the original string representation of the token.

    ```python
    for token in doc:
        print(token.text)
    ```

### Full Code Example

```python
import spacy

# 1. Load the English language model
# Make sure you have installed it: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# 2. Define the text to be tokenized
doc_text = "I love programming in Python."

# 3. Process the text to create a Doc object
doc = nlp(doc_text)

# 4. Iterate through the Doc object and print each token
print("Tokens:")
for token in doc:
    print(token.text)
```

**Output of the code:**

```
Tokens:
I
love
programming
in
Python
.
```