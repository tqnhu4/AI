# src/cv_parser.py

import io
import re
import pdfplumber
from docx import Document

class CVParser:
    def __init__(self):
        pass

    def parse_pdf(self, file_content: bytes) -> str:
        """
        Phân tích nội dung từ file PDF.
        """
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() if page.extract_text() else ""
        except Exception as e:
            print(f"Lỗi khi phân tích PDF: {e}")
            return f"Không thể phân tích PDF: {e}"
        return text

    def parse_docx(self, file_content: bytes) -> str:
        """
        Phân tích nội dung từ file DOCX.
        """
        text = ""
        try:
            doc = Document(io.BytesIO(file_content))
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Lỗi khi phân tích DOCX: {e}")
            return f"Không thể phân tích DOCX: {e}"
        return text

    def parse_cv(self, file_content: bytes, file_type: str) -> str:
        """
        Phân tích CV dựa trên loại file.
        """
        if file_type == "pdf":
            return self.parse_pdf(file_content)
        elif file_type == "docx":
            return self.parse_docx(file_content)
        else:
            return "Loại file không được hỗ trợ. Vui lòng tải lên PDF hoặc DOCX."