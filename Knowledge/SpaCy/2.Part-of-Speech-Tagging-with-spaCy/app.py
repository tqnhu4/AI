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
