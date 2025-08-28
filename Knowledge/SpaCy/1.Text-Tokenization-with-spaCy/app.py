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
