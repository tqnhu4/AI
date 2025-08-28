# src/ai_advisor.py

class AICareerAdvisor:
    def __init__(self, model):
        self.llm = model

    def analyze_resume(self, cv_text: str, extracted_skills: list) -> str:
        """
        Phân tích CV và đưa ra lời khuyên cải thiện.
        """
        prompt = f"""Bạn là một chuyên gia tư vấn nghề nghiệp AI.
        Dựa trên nội dung CV sau và các kỹ năng được trích xuất, hãy đưa ra phân tích và lời khuyên chi tiết để cải thiện CV.
        Tập trung vào:
        1. Đánh giá tổng thể CV (cấu trúc, rõ ràng, điểm mạnh).
        2. Phân tích các kỹ năng hiện có và đề xuất kỹ năng cần bổ sung/nâng cao cho các vai trò phổ biến trong ngành IT (ví dụ: Software Engineer, Data Scientist, DevOps Engineer).
        3. Gợi ý cách trình bày kinh nghiệm, dự án để nổi bật hơn.
        4. Các lỗi phổ biến cần tránh.

        Nội dung CV:
        ```
        {cv_text[:4000]} # Giới hạn độ dài để tránh vượt token limit
        ```
        Các kỹ năng được trích xuất: {', '.join(extracted_skills)}

        Phân tích và lời khuyên của bạn (viết bằng tiếng Việt, khoảng 300-500 từ):
        """
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Lỗi khi gọi AI Career Advisor: {e}")
            return f"Xin lỗi, tôi không thể phân tích CV lúc này: {e}"

    def suggest_careers(self, extracted_skills: list) -> str:
        """
        Gợi ý các ngành nghề phù hợp dựa trên kỹ năng.
        """
        prompt = f"""Bạn là một chuyên gia tư vấn nghề nghiệp AI.
        Dựa trên danh sách kỹ năng sau, hãy gợi ý 3-5 ngành nghề/vị trí công việc phù hợp trong lĩnh vực công nghệ thông tin.
        Đối với mỗi ngành nghề, hãy giải thích ngắn gọn tại sao nó phù hợp và những kỹ năng nào trong danh sách là điểm mạnh.

        Các kỹ năng: {', '.join(extracted_skills)}

        Gợi ý ngành nghề của bạn (viết bằng tiếng Việt):
        """
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Lỗi khi gợi ý ngành nghề: {e}")
            return f"Xin lỗi, tôi không thể gợi ý ngành nghề lúc này: {e}"