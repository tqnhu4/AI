import spacy

# 1. Load the language model
# Ensure you have run 'python -m spacy download en_core_web_sm' beforehand
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model 'en_core_web_sm' not downloaded. Please run: python -m spacy download en_core_web_sm")
    exit()

text = "Apple is looking at buying U.K. startup for $1 billion. Tim Cook is the CEO."

# 2. Process the text
doc = nlp(text)

print("--- 1. Tokenization ---")
# Tokens (words and punctuation)
for token in doc:
    print(f"'{token.text}'")

print("\n--- 2. Part-of-Speech Tagging (POS) ---")
# POS tag and Lemma for each token
for token in doc:
    print(f"'{token.text}': POS={token.pos_}, Lemma={token.lemma_}")

print("\n--- 3. Named Entity Recognition (NER) ---")
# Named entities and their types
for ent in doc.ents:
    print(f"'{ent.text}': Entity Type={ent.label_}")

print("\n--- 4. Dependency Parsing ---")
# Dependency relationships between words
for token in doc:
    print(f"'{token.text}': Dep={token.dep_}, Head='{token.head.text}'")

print("\n--- 5. Lemmatization ---")
# Base form of words (already shown in POS section, but this focuses on lemma)
print("Lemmas of words:")
for token in doc:
    print(f"'{token.text}' -> '{token.lemma_}'")

print("\n--- 6. Rule-based Matching ---")
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)

# Example: Find phrases like "iPhone" followed by a number and "billion"
# Or "Apple" followed by "looking" and "at"
pattern1 = [{"LOWER": "iphone"}, {"IS_DIGIT": True}, {"LOWER": "billion"}]
pattern2 = [{"LOWER": "apple"}, {"LOWER": "looking"}, {"LOWER": "at"}]

matcher.add("MY_PATTERN", [pattern1, pattern2])

matches = matcher(doc)

print("Matching results:")
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Match ID: {nlp.vocab.strings[match_id]}, Position: ({start}, {end}), Matched Text: '{span.text}'")