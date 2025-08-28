# src/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import json

# Import các cấu hình và lớp logic từ các module mới
from .config import GEMINI_MODEL_NAME
from .cv_parser import CVParser
from .skill_extractor import SkillExtractor, embedding_model
from .ai_advisor import AICareerAdvisor
from .job_matcher import JobMatcher
from .interview_chatbot import InterviewChatbot

import google.generativeai as genai

# Khởi tạo mô hình Gemini
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

app = Flask(__name__)
CORS(app) # Cho phép CORS cho tất cả các route

# Khởi tạo các đối tượng xử lý
# Các đối tượng này sẽ được khởi tạo một lần khi ứng dụng Flask khởi động
cv_parser = CVParser()
skill_extractor = SkillExtractor(embedding_model)
job_matcher = JobMatcher(embedding_model) # JobMatcher sử dụng Singleton để giữ FAISS index
ai_advisor = AICareerAdvisor(model)
interview_chatbot = InterviewChatbot(model) # InterviewChatbot quản lý phiên chat qua session_id

@app.route('/analyze-cv', methods=['POST'])
def analyze_cv():
    try:
        data = request.get_json()
        file_content_base64 = data.get('file_content')
        file_type = data.get('file_type')
        session_id = request.headers.get('X-Session-Id', 'default') # Lấy session_id từ header

        if not file_content_base64 or not file_type:
            return jsonify({'error': 'Missing file_content or file_type'}), 400
        
        file_content = base64.b64decode(file_content_base64)
        cv_text = cv_parser.parse_cv(file_content, file_type)

        if "Không thể phân tích" in cv_text:
            return jsonify({'error': cv_text}), 500

        extracted_skills = skill_extractor.extract_skills(cv_text)
        cv_embedding = skill_extractor.get_embedding(cv_text)
        
        career_advice = ai_advisor.analyze_resume(cv_text, extracted_skills)
        career_suggestions = ai_advisor.suggest_careers(extracted_skills)
        matching_jobs = job_matcher.find_matching_jobs(cv_embedding)

        return jsonify({
            'parsed_text': cv_text,
            'extracted_skills': extracted_skills,
            'career_advice': career_advice,
            'career_suggestions': career_suggestions,
            'matching_jobs': matching_jobs
        })

    except Exception as e:
        print(f"Lỗi xử lý /analyze-cv: {e}")
        return jsonify({'error': f'Internal server error: {e}'}), 500

@app.route('/interview', methods=['POST'])
def interview():
    try:
        data = request.get_json()
        action = data.get('action')
        session_id = request.headers.get('X-Session-Id', 'default') # Lấy session_id từ header

        if action == 'start':
            cv_text = data.get('cv_text')
            job_description = data.get('job_description')
            if not cv_text:
                return jsonify({'error': 'Missing cv_text to start interview'}), 400
            response = interview_chatbot.start_interview(cv_text, job_description, session_id)
            return jsonify({'response': response})
        elif action == 'continue':
            user_response = data.get('user_response')
            if not user_response:
                return jsonify({'error': 'Missing user_response to continue interview'}), 400
            response = interview_chatbot.continue_interview(user_response, session_id)
            return jsonify({'response': response})
        elif action == 'end':
            response = interview_chatbot.end_interview(session_id)
            return jsonify({'response': response})
        else:
            return jsonify({'error': 'Invalid interview action'}), 400

    except Exception as e:
        print(f"Lỗi xử lý /interview: {e}")
        return jsonify({'error': f'Internal server error: {e}'}), 500

# Endpoint cho CORS preflight requests (OPTIONS)
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    response = jsonify({'message': 'OK'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, X-Session-Id')
    response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    response.headers.add('Access-Control-Max-Age', '86400')
    return response

# Nếu chạy cục bộ (không phải Lambda)
if __name__ == '__main__':
    # Đảm bảo GEMINI_API_KEY được đặt trong biến môi trường cục bộ
    # hoặc thay thế "YOUR_GEMINI_API_KEY_HERE" trong src/config.py
    app.run(debug=True, host='0.0.0.0', port=5000)
