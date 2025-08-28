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