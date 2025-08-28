import streamlit as st
import pandas as pd
import io
from src.pdf_parser import extract_text_from_pdf
from src.info_extractor import InformationExtractor
from typing import Dict, List

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Contract Info Extractor")

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