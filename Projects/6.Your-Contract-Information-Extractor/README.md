
# 🤖 Contract Information Extractor: Basic Version

This project implements a fundamental version of a virtual assistant that extracts information from PDF contracts. This basic iteration focuses on demonstrating the core workflow, including PDF parsing and a simple (rule-based or default spaCy) Named Entity Recognition (NER) for information extraction, presented via a user-friendly Streamlit interface.

**Please Note:** This "basic" version uses spaCy's pre-trained `en_core_web_sm` model, which is general-purpose. For highly accurate and contract-specific information extraction (like "Contract Term" or "Payment Clause"), **fine-tuning a specialized model (e.g., BERT as in the previous discussion) on annotated contract data is essential.** This basic version serves as a foundational blueprint.

### ✅ Description:
Build an application that reads a contract PDF file (or legal document) and extracts information such as:

Parties' names

Signing date

Contract duration

Amount or payment terms

#### ✅ Techniques used:
spaCy for grammar processing (tokenize, POS, NER)

BERT fine-tuned model from HuggingFace to extract information according to specific requirements

#### ✅ Extended features:
Simple UI for PDF upload

Extracted results are displayed as editable tables
-----

## 📁 Project Structure

```
contract_extractor_basic/
├── venv/                       # Python virtual environment
├── data/                       # Stores sample data
│   └── sample_contracts/       # Place your sample PDF contracts here (e.g., dummy_contract.pdf)
├── src/                        # Source code for the application logic
│   ├── pdf_parser.py           # Handles PDF to text conversion
│   ├── info_extractor.py       # Basic information extraction logic (using spaCy default NER)
│   └── __init__.py             # Makes 'src' a Python package (empty file)
├── app.py                      # Main Streamlit application file
├── requirements.txt            # Lists all Python dependencies
├── README.md                   # Project description and instructions
└── .gitignore                  # Specifies files/directories to be ignored by Git
```

-----

## 🛠️ Setup and Installation

Follow these steps to set up your project environment:

1.  **Create Project Directory & Navigate:**

    ```bash
    mkdir contract_extractor_basic
    cd contract_extractor_basic
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

3.  **Create Project Files and Directories:**

    ```bash
    mkdir data data/sample_contracts
    mkdir src
    touch src/__init__.py
    touch src/pdf_parser.py
    touch src/info_extractor.py
    touch app.py
    touch requirements.txt
    touch README.md
    touch .gitignore
    ```

4.  **Install Dependencies:**
    First, populate `requirements.txt` (see next section), then run:

    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm # Download the small English language model for spaCy
    ```

-----

## 📦 `requirements.txt`

This file lists all the Python packages required for the project.

```
# requirements.txt
pypdf==4.2.0    # For reading PDF files
spacy==3.7.4    # For Natural Language Processing (NER, POS, etc.)
streamlit==1.36.0 # For building the web-based user interface
pandas==2.2.2   # For data manipulation and tabular display
```

-----

## 🙈 `.gitignore`

This file tells Git which files and directories to ignore when tracking changes.

```
# .gitignore
venv/
__pycache__/
*.pyc
.streamlit/ # Streamlit's internal cache directory
data/sample_contracts/ # If your sample contracts are large or sensitive, consider ignoring
```

-----




# 🤖 Contract Information Extractor (Basic Version)

This project demonstrates a basic application for extracting key information from PDF contract documents using Python, spaCy, and Streamlit.

## ✨ Features

-   **PDF Text Extraction:** Reads text content from uploaded PDF files.
-   **Basic Information Extraction:** Uses spaCy's pre-trained NER model to identify entities like organizations, dates, and monetary values.
-   **Interactive UI:** A simple web interface for uploading PDFs and viewing extracted information in an editable table.

## 🛠️ Technologies Used

-   **Python 3.x**
-   **`pypdf`**: For parsing PDF documents.
-   **`spaCy`**: For Natural Language Processing (tokenization, POS tagging, Named Entity Recognition).
-   **`Streamlit`**: For building the interactive web user interface.
-   **`pandas`**: For managing and displaying data in a tabular format.

## 🚀 Setup and Installation

1.  **Clone this repository (or create the project structure manually):**
    ```bash
    git clone ..git
    cd 6.Your-Contract-Information-Extractor
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    
    # On Windows:
    # .\venv\Scripts\activate

    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy English language model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## 🏃 How to Run the Application

1.  Ensure your virtual environment is activated (`source venv/bin/activate`).
2.  Run the Streamlit application from the root project directory:
    ```bash
    streamlit run app.py
    ```
3.  Your web browser should automatically open to the Streamlit app (usually `http://localhost:8501`).
4.  Upload a PDF contract and see the extracted information.

## 📁 Project Structure Explained

-   `venv/`: Isolated Python environment with project dependencies.
-   `data/`: Stores any data, e.g., `sample_contracts/` for test PDFs.
-   `src/`: Contains core Python modules for application logic.
    -   `pdf_parser.py`: Logic for extracting text from PDFs.
    -   `info_extractor.py`: Logic for basic information extraction using spaCy.
-   `app.py`: The main Streamlit script that orchestrates the UI and calls functions from `src/`.
-   `requirements.txt`: Lists all Python package dependencies.
-   `README.md`: This file, providing project overview and instructions.
-   `.gitignore`: Specifies files/directories to be ignored by Git.

## ⚠️ Limitations of this Basic Version

-   **Extraction Accuracy:** This version relies on spaCy's general pre-trained NER model (`en_core_web_sm`), which is not specifically trained for legal contracts. It will perform well on general entities (dates, organizations, money) but may not accurately identify contract-specific terms like "Contract Term" or "Payment Clause" as distinct entities.
-   **No Custom Model Training:** It does not include the complex process of fine-tuning a BERT model on custom annotated contract data, which is essential for high-precision, domain-specific extraction.

## 💡 Future Enhancements

-   Integrate a **fine-tuned BERT model** (as discussed previously) for highly accurate, contract-specific information extraction.
-   Implement more robust **post-processing rules** to refine extracted data (e.g., standardizing date formats, currency detection).
-   Add **export functionality** (CSV, JSON).
-   Improve error handling for various PDF types (e.g., scanned PDFs).
````

-----

## 🐍 `src/pdf_parser.py`

```python
# src/pdf_parser.py

import io
from pypdf import PdfReader
from typing import Optional

def extract_text_from_pdf(pdf_file: io.BytesIO) -> Optional[str]:
    """
    Extracts all text content from a PDF file-like object.

    Args:
        pdf_file (io.BytesIO): A BytesIO object containing the PDF's binary content.

    Returns:
        Optional[str]: A single string containing all extracted text,
                       or None if an error occurs during extraction.
    """
    try:
        reader = PdfReader(pdf_file)
        text_content = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # Ensure text was actually extracted from the page
                text_content += page_text + "\n" # Add a newline between pages
        return text_content
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

if __name__ == "__main__":
    # --- Basic Test for pdf_parser.py ---
    # To run this test:
    # 1. Create a dummy_contract.pdf file (even a simple one with text) in your project root.
    # 2. Run: python src/pdf_parser.py
    print("--- Testing src/pdf_parser.py ---")
    try:
        # Create a dummy PDF in memory for demonstration purposes
        # In a real scenario, you'd load a file from disk or an upload
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=72 * 8.5, height=72 * 11)
        writer.pages[0].add_js(
            f"this.addAnnot({{page: 0, rect: [72, 720, 520, 750], type: 'FreeText', contents: 'This is a sample contract. Parties: Alpha Corp and Beta Ltd. Date: 2023-01-01. Amount: $10,000.', font: 'Helvetica', point: [72, 720]}})"
        )
        dummy_pdf_io = io.BytesIO()
        writer.write(dummy_pdf_io)
        dummy_pdf_io.seek(0) # Reset stream position to the beginning

        extracted_text = extract_text_from_pdf(dummy_pdf_io)

        if extracted_text:
            print("\nSuccessfully extracted text. First 500 characters:")
            print(extracted_text[:500])
        else:
            print("\nFailed to extract text from dummy PDF.")
    except ImportError:
        print("pypdf might not be fully installed or its features vary across versions.")
        print("Please ensure 'pypdf' is installed via `pip install pypdf`.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

```

-----

## 🐍 `src/info_extractor.py`

```python
# src/info_extractor.py

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

```

-----

## 🐍 `app.py`

```python
# app.py

import streamlit as st
import pandas as pd
import io
from src.pdf_parser import extract_text_from_pdf
from src.info_extractor import InformationExtractor
from typing import Dict, List

# --- Configuration ---
# Use st.cache_resource to load heavy objects like NLP models only once.
# This prevents reloading the model every time the Streamlit app reruns.
@st.cache_resource
def load_extractor_model():
    """Loads the InformationExtractor and its spaCy model."""
    try:
        extractor = InformationExtractor()
        return extractor
    except RuntimeError as e:
        st.error(f"Error loading NLP model: {e}")
        st.stop() # Stop the app if model loading fails
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()

extractor = load_extractor_model()

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Contract Info Extractor")

st.title("📄 Contract Information Extractor (Basic)")
st.markdown("Upload a PDF contract below, and the AI will attempt to extract key details.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload a Contract PDF", type="pdf")

if uploaded_file is not None:
    # Convert uploaded file to BytesIO object for pdf_parser
    pdf_content_io = io.BytesIO(uploaded_file.read())

    with st.spinner("Extracting text from PDF..."):
        contract_text = extract_text_from_pdf(pdf_content_io)

    if contract_text:
        st.success("Text extracted successfully!")
        st.markdown("---")
        st.header("✨ Extracted Information")

        with st.spinner("Analyzing text and extracting information..."):
            # Use the basic information extraction logic
            extracted_structured_data: Dict[str, List[str]] = extractor.get_structured_contract_info(contract_text)

        # Prepare data for pandas DataFrame (for the editable table)
        df_rows = []
        for field, values in extracted_structured_data.items():
            # For simplicity, if multiple values, join them or list them
            # For a basic editable table, each item will be a row.
            if values:
                for value in values:
                    df_rows.append({"Field": field, "Value": value})
            else:
                df_rows.append({"Field": field, "Value": "N/A"}) # No value extracted

        if not df_rows:
            st.warning("No specific information could be extracted based on the current model and rules.")
        else:
            df_extracted = pd.DataFrame(df_rows)

            st.markdown("#### Review and Edit Extracted Data:")
            # `st.data_editor` allows users to interactively edit the DataFrame
            edited_df = st.data_editor(
                df_extracted,
                num_rows="dynamic", # Allows adding/deleting rows
                hide_index=True,    # Hides pandas DataFrame index
                use_container_width=True, # Makes table span full width
                key="extracted_data_editor" # Unique key for the widget
            )

            st.markdown("---")
            st.markdown("### 📝 Original Contract Text (for reference)")
            with st.expander("Click to view full text"):
                st.text_area(
                    "Full Contract Text:",
                    contract_text,
                    height=400,
                    key="full_text_area",
                    help="This is the raw text extracted from your PDF."
                )

            st.markdown("---")
            if st.button("Save Edited Data (as JSON)", help="In a real application, this would save to a database or file."):
                # For this basic version, we just display the edited data as JSON
                st.success("Data saved (displayed below as JSON for demonstration).")
                st.json(edited_df.to_dict(orient="records"))
    else:
        st.error("Failed to extract text from the PDF. This might happen if the PDF is an image-only (scanned) document without selectable text.")
        st.info("Please try uploading a text-searchable PDF.")

```