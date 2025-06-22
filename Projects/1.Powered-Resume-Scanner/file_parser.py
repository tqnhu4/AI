# file_parser.py
import docx
import PyPDF2
import io

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

def extract_text_from_docx(docx_file):
    """Extracts text from a DOCX file."""
    text = ""
    try:
        doc = docx.Document(docx_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return None
    return text

def extract_text_from_txt(txt_file):
    """Extracts text from a TXT file."""
    try:
        return txt_file.read().decode("utf-8")
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return None

def parse_resume(uploaded_file):
    """
    Parses an uploaded resume file and returns its text content.
    Supports PDF, DOCX, and TXT formats.
    """
    if uploaded_file is None:
        return None

    file_extension = uploaded_file.name.split('.')[-1].lower()
    text = None

    # Streamlit files are already file-like objects, but PyPDF2/docx might need io.BytesIO
    file_stream = io.BytesIO(uploaded_file.getvalue())

    if file_extension == "pdf":
        text = extract_text_from_pdf(file_stream)
    elif file_extension == "docx":
        text = extract_text_from_docx(file_stream)
    elif file_extension == "txt":
        text = extract_text_from_txt(file_stream)
    else:
        return None, "Unsupported file format. Please upload PDF, DOCX, or TXT."

    return text, None # Return text and no error, or None and an error message