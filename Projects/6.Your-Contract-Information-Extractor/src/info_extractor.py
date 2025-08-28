import spacy
from typing import List, Dict, Optional

# Load a pre-trained spaCy English model
# This model includes a default Named Entity Recognition (NER) pipeline.
# For contract-specific entities, a custom fine-tuned model would be needed.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    # Exit or handle error appropriately in production
    nlp = None # Set to None if model loading fails

class InformationExtractor:
    """
    A basic information extractor using spaCy's default NER model.
    """
    def __init__(self):
        if nlp is None:
            raise RuntimeError("spaCy model 'en_core_web_sm' could not be loaded. Please ensure it's downloaded.")
        self.nlp = nlp

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts named entities from the given text using spaCy's default NER.

        Args:
            text (str): The input text (e.g., contract content).

        Returns:
            List[Dict[str, str]]: A list of dictionaries, where each dictionary represents
                                  an extracted entity with 'text' and 'label'.
                                  Example: [{'text': 'Acme Corp', 'label': 'ORG'}, ...]
        """
        if not text:
            return []

        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return entities

    def get_structured_contract_info(self, text: str) -> Dict[str, List[str]]:
        """
        Attempts to extract and structure common contract information using
        a combination of spaCy NER and simple keyword/pattern matching (basic).

        Args:
            text (str): The full text of the contract.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are information categories
                                  and values are lists of extracted strings.
        """
        extracted_data = {
            "Party Name": [],
            "Signing Date": [],
            "Contract Term": [],
            "Payment Amount": [],
            "Payment Clause": []
        }

        if not text:
            return extracted_data

        doc = self.nlp(text)

        # 1. Extract Party Names (Organizations and Persons often involved)
        for ent in doc.ents:
            if ent.label_ == "ORG" or ent.label_ == "PERSON":
                # Simple heuristic: filter out common non-party ORGs like "Inc." "Ltd." if they are not part of a larger name
                if len(ent.text.split()) > 1 and ent.text.lower() not in ["inc.", "ltd.", "corp.", "llc"]: # Basic filter
                    extracted_data["Party Name"].append(ent.text)

        # Remove duplicates and sort for cleaner output
        extracted_data["Party Name"] = sorted(list(set(extracted_data["Party Name"])))

        # 2. Extract Signing Date (Dates)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                # Simple check for common date phrases
                if any(phrase in ent.sent.text.lower() for phrase in ["signed on", "dated", "effective as of", "made on"]):
                    extracted_data["Signing Date"].append(ent.text)
        extracted_data["Signing Date"] = sorted(list(set(extracted_data["Signing Date"])))
        if len(extracted_data["Signing Date"]) > 1:
            # If multiple dates found, pick the earliest or use context
            # For this basic version, we might just take the first or assume latest relevant
            pass # Keep all for user review in basic version

        # 3. Extract Payment Amounts (Money)
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                extracted_data["Payment Amount"].append(ent.text)
        extracted_data["Payment Amount"] = sorted(list(set(extracted_data["Payment Amount"])))


        # 4. Basic Keyword/Pattern Matching for Contract Term & Payment Clause (very basic)
        # These are usually not direct NER entities, so a simple pattern search is used.
        # For real extraction, this would be a custom NER or rule-based system.
        lines = text.split('\n')
        for line in lines:
            lower_line = line.lower()
            if "term of this agreement" in lower_line or "contract term" in lower_line or "duration of this agreement" in lower_line:
                # Attempt to find common time units near these phrases
                import re
                match = re.search(r'(\d+)\s+(year|month|day)s?', lower_line)
                if match:
                    extracted_data["Contract Term"].append(match.group(0))
                else: # Add the whole line if a specific term isn't found
                    extracted_data["Contract Term"].append(line.strip())
            
            if "payment shall be" in lower_line or "payment terms" in lower_line or "due within" in lower_line or "invoice will be paid" in lower_line:
                extracted_data["Payment Clause"].append(line.strip())

        # Clean up lists if empty
        for key in extracted_data:
            if not extracted_data[key]:
                extracted_data[key] = ["N/A"] # Indicate no data found


        return extracted_data

if __name__ == "__main__":
    # --- Basic Test for info_extractor.py ---
    print("--- Testing src/info_extractor.py ---")
    if nlp is None:
        print("Skipping tests as spaCy model could not be loaded.")
    else:
        extractor = InformationExtractor()
        sample_contract_text = """
        This Agreement is made on January 1st, 2023, by and between Alpha Corporation (referred to as "Alpha")
        and Beta Solutions Limited (referred to as "Beta").
        The term of this Agreement shall commence on the Effective Date and continue for a period of 12 months.
        Beta shall pay Alpha an amount of $50,000 USD. Payment terms require 50% upfront, with the remaining
        50% due within 30 days of project completion.
        This contract was signed in New York.
        """

        print("\n--- Raw Named Entities (spaCy default) ---")
        raw_entities = extractor.extract_entities(sample_contract_text)
        for ent in raw_entities:
            print(f"  Text: '{ent['text']}', Label: {ent['label']}")

        print("\n--- Structured Contract Information (Basic Logic) ---")
        structured_info = extractor.get_structured_contract_info(sample_contract_text)
        for key, values in structured_info.items():
            print(f"  {key}: {values}")