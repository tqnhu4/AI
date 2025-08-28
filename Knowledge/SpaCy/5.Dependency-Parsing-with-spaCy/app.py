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