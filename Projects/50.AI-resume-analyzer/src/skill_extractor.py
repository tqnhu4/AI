# src/skill_extractor.py

import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Khởi tạo mô hình Embedding (cần tải xuống lần đầu)
# Mô hình nhỏ để giảm kích thước gói Lambda, nhưng vẫn đủ tốt cho demo
# 'all-MiniLM-L6-v2' là một lựa chọn tốt về hiệu suất/kích thước
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Lỗi khi tải mô hình embedding: {e}. Đảm bảo có kết nối internet hoặc tải thủ công.")
    # Fallback cho trường hợp không tải được mô hình (ví dụ: trong Lambda nếu không có Layer phù hợp)
    class MockEmbeddingModel:
        def encode(self, texts, convert_to_tensor=False):
            print("Sử dụng mô hình embedding giả lập.")
            # all-MiniLM-L6-v2 có 384 chiều
            return np.random.rand(len(texts), 384).astype(np.float32)
    embedding_model = MockEmbeddingModel()

class SkillExtractor:
    def __init__(self, model):
        self.embedding_model = model
        # Danh sách kỹ năng mẫu (có thể mở rộng từ cơ sở dữ liệu)
        self.known_skills = [
            "Python", "Java", "C++", "JavaScript", "React", "Angular", "Vue.js",
            "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "DynamoDB",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CloudFormation",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Science",
            "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
            "Git", "CI/CD", "Agile", "Scrum", "Project Management",
            "Communication", "Teamwork", "Problem Solving", "Leadership", "Critical Thinking"
        ]
        self.skill_embeddings = self.embedding_model.encode(self.known_skills, convert_to_tensor=False)

    def extract_skills(self, text: str) -> list:
        """
        Trích xuất kỹ năng từ văn bản CV bằng cách so khớp từ khóa đơn giản.
        Trong phiên bản nâng cao, có thể dùng NER model (spaCy, HuggingFace).
        """
        extracted = set()
        text_lower = text.lower()
        for skill in self.known_skills:
            # Sử dụng regex để tìm kiếm từ khóa đầy đủ
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                extracted.add(skill)
        return list(extracted)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Tạo embedding cho văn bản.
        """
        # Đảm bảo trả về numpy array float32
        return self.embedding_model.encode(text, convert_to_tensor=False).astype(np.float32)