Welcome to the "AI Resume Analyzer & Career Assistant" project!

This project is designed to help users:

Analyze their CVs (PDF/DOCX files) using AI.

Evaluate skills and suggest improvements for their CVs.

Suggest suitable job openings based on their capabilities.

Engage in a mock interview chatbot for practice.

We will use Python for the backend logic (Flask), Streamlit for a simple web interface, and Terraform for deploying the infrastructure on AWS (Lambda, API Gateway, S3). You will need a Gemini API key for the generative AI functionalities.

Let's get started!

ai-resume-analyzer/
├── src/
│   ├── __init__.py        # Tệp rỗng để Python nhận diện đây là một package
│   ├── config.py          # Cấu hình chung
│   ├── cv_parser.py       # Chứa lớp CVParser
│   ├── skill_extractor.py # Chứa lớp SkillExtractor và khởi tạo embedding model
│   ├── ai_advisor.py      # Chứa lớp AICareerAdvisor
│   ├── job_matcher.py     # Chứa lớp JobMatcher
│   ├── interview_chatbot.py # Chứa lớp InterviewChatbot
│   ├── app.py             # Ứng dụng Flask chính (trước đây là flask_app.py)
│   └── wsgi_handler.py    # Handler cho AWS Lambda
├── frontend/
│   └── app.py             # Ứng dụng Streamlit
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── requirements.txt       # Danh sách thư viện Python
└── lambda_package.zip     # Tệp zip cuối cùng để triển khai lên Lambda


Chào bạn,

Bạn đang hỏi về hướng dẫn cài đặt dựa trên các thư viện đã chọn trong Canvas "Tệp requirements.txt".

Để cài đặt các thư viện này, bạn cần thực hiện các bước sau trong môi trường Python của mình:

1.  **Tạo một tệp `requirements.txt`:**
    Nếu bạn chưa có, hãy tạo một tệp có tên `requirements.txt` trong thư mục dự án của bạn và dán nội dung đã chọn vào đó:

    ```
    streamlit
    requests
    pdfplumber
    python-docx
    sentence-transformers
    faiss-cpu
    google-generativeai
    Flask
    Flask-Cors
    awsgi # Bộ chuyển đổi WSGI cho Lambda
    gunicorn # Để chạy Flask cục bộ một cách ổn định (tùy chọn)
    ```

2.  **Kích hoạt môi trường ảo (khuyến nghị):**
    Để tránh xung đột thư viện với các dự án Python khác, bạn nên sử dụng môi trường ảo.

      * Tạo môi trường ảo:
        ```bash
        python -m venv venv
        ```
      * Kích hoạt môi trường ảo:
          * Trên Windows:
            ```bash
            .\venv\Scripts\activate
            ```
          * Trên macOS/Linux:
            ```bash
            source venv/bin/activate
            ```

3.  **Cài đặt các thư viện:**
    Sau khi môi trường ảo đã được kích hoạt, chạy lệnh sau để cài đặt tất cả các thư viện được liệt kê trong `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

Lệnh này sẽ tự động tải xuống và cài đặt tất cả các thư viện cần thiết cho ứng dụng của bạn.
