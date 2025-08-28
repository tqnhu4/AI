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