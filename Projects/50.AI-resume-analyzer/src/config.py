# src/config.py

import os
import google.generativeai as genai

# --- Cấu hình API Key ---
# LƯU Ý: Trong môi trường sản phẩm, hãy sử dụng AWS Secrets Manager
# để lưu trữ khóa API thay vì hardcode hoặc biến môi trường trực tiếp.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

# --- Cấu hình Mô hình AI ---
GEMINI_MODEL_NAME = "gemini-1.5-flash" # Hoặc "gemini-1.5-pro"