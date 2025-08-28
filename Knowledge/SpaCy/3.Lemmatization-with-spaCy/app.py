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