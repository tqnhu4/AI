# src/job_matcher.py

import faiss
import numpy as np

class JobMatcher:
    _instance = None # Singleton pattern để giữ FAISS index trong bộ nhớ Lambda
    _initialized = False

    def __new__(cls, embedding_model):
        if cls._instance is None:
            cls._instance = super(JobMatcher, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_model):
        if not self._initialized:
            self.embedding_model = embedding_model
            self.job_descriptions = []
            self.job_embeddings = None
            self.faiss_index = None
            self._load_mock_jobs() # Tải dữ liệu việc làm mẫu
            JobMatcher._initialized = True

    def _load_mock_jobs(self):
        """
        Tải dữ liệu việc làm mẫu và tạo embedding/chỉ mục FAISS.
        Trong thực tế, dữ liệu này sẽ được tải từ DB hoặc API.
        """
        mock_jobs = [
            {"id": "job1", "title": "Software Engineer", "description": "Phát triển ứng dụng web với Python, JavaScript, SQL. Kinh nghiệm AWS, Docker là lợi thế."},
            {"id": "job2", "title": "Data Scientist", "description": "Phân tích dữ liệu lớn, xây dựng mô hình Machine Learning với Python, Pandas, TensorFlow. Kinh nghiệm SQL, thống kê."},
            {"id": "job3", "title": "DevOps Engineer", "description": "Quản lý hạ tầng đám mây AWS, triển khai CI/CD với Docker, Kubernetes, Terraform. Kinh nghiệm Linux, scripting."},
            {"id": "job4", "title": "Frontend Developer", "description": "Xây dựng giao diện người dùng với React, HTML, CSS, JavaScript. Kinh nghiệm UI/UX."},
            {"id": "job5", "title": "Backend Developer (Java)", "description": "Phát triển API với Java, Spring Boot, PostgreSQL. Kinh nghiệm Microservices."},
            {"id": "job6", "title": "Machine Learning Engineer", "description": "Triển khai mô hình ML vào sản phẩm, tối ưu hiệu suất với Python, PyTorch, AWS SageMaker. Có kinh nghiệm MLOps."},
            {"id": "job7", "title": "Cloud Architect", "description": "Thiết kế kiến trúc giải pháp đám mây trên AWS, Azure. Am hiểu mạng, bảo mật, serverless."},
            {"id": "job8", "title": "Mobile Developer (Android)", "description": "Phát triển ứng dụng Android với Kotlin, Java. Kinh nghiệm RESTful API, Firebase."},
            {"id": "job9", "title": "QA Engineer", "description": "Kiểm thử phần mềm, viết test case, tự động hóa kiểm thử với Selenium, Python. Có kinh nghiệm Agile."},
            {"id": "job10", "title": "Product Manager", "description": "Quản lý vòng đời sản phẩm, thu thập yêu cầu người dùng, làm việc với đội ngũ kỹ thuật. Kinh nghiệm Agile, Scrum, giao tiếp tốt."}
        ]
        self.job_descriptions = mock_jobs
        job_texts = [job["title"] + " " + job["description"] for job in mock_jobs]
        self.job_embeddings = self.embedding_model.encode(job_texts, convert_to_tensor=False)

        # Tạo chỉ mục FAISS
        dimension = self.job_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
        self.faiss_index.add(self.job_embeddings.astype('float32')) # FAISS cần float32

    def find_matching_jobs(self, cv_embedding: np.ndarray, top_k: int = 5) -> list:
        """
        Tìm kiếm các công việc phù hợp nhất dựa trên embedding của CV.
        """
        if self.faiss_index is None:
            return []

        # FAISS cần input 2D array, ngay cả khi chỉ có 1 vector
        D, I = self.faiss_index.search(cv_embedding.astype('float32').reshape(1, -1), top_k)
        
        matching_jobs = []
        for i, idx in enumerate(I[0]):
            job = self.job_descriptions[idx]
            score = D[0][i] # Điểm khoảng cách (thấp hơn tốt hơn cho L2)
            matching_jobs.append({
                "id": job["id"],
                "title": job["title"],
                "description": job["description"],
                "score": float(score) # Chuyển đổi sang float để có thể serialize JSON
            })
        return matching_jobs