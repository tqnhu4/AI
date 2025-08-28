# app.py

import streamlit as st
from transformers import pipeline
import PyPDF2
import io

# Initialize the summarization pipeline
# Using 'sshleifer/distilbart-cnn-12-6' which is a good general-purpose summarization model.
# This will download the model the first time it runs.
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

def summarize_text(text, max_length=150, min_length=30):
    if not text.strip():
        return "No text provided to summarize."
    try:
        # The summarizer often takes a list of texts
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {e}"

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        # PyPDF2 expects a binary file, so use io.BytesIO for uploaded files
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

st.title("🧠 AI Text Summarizer")
st.markdown("---")

st.subheader("1. Summarize from Text Input")
text_input = st.text_area("Paste your text here:", height=300, help="Enter the long text you want to summarize.")

if st.button("Summarize Text"):
    if text_input:
        with st.spinner("Summarizing... Please wait."):
            summary = summarize_text(text_input)
            st.success("Summary from Text:")
            st.write(summary)
    else:
        st.warning("Please paste some text to summarize.")

st.markdown("---")

st.subheader("2. Summarize from PDF Upload")
pdf_file = st.file_uploader("Upload a PDF file:", type=["pdf"], help="Upload a PDF document to get its summary.")

if pdf_file:
    with st.spinner("Extracting text from PDF and summarizing..."):
        extracted_text = extract_text_from_pdf(pdf_file)
        if "Error" in extracted_text:
            st.error(extracted_text) # Display PDF extraction error
        elif extracted_text.strip():
            summary = summarize_text(extracted_text)
            st.success("Summary from PDF:")
            st.write(summary)
            # Optionally show extracted text for debugging/transparency
            # with st.expander("View Extracted Text from PDF"):
            #     st.text(extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text)
        else:
            st.warning("Could not extract any meaningful text from the PDF.")
else:
    st.info("Upload a PDF file above to get its summary.")

st.markdown("---")
st.caption("Powered by Hugging Face Transformers & Streamlit")        