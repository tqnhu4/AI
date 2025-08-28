# src/interview_chatbot.py

class InterviewChatbot:
    # Sử dụng một dictionary để lưu trữ phiên chat cho từng session_id
    _chat_sessions = {} # session_id -> chat_session

    def __init__(self, model):
        self.llm = model

    def start_interview(self, cv_text: str, job_description: str = None, session_id: str = "default") -> str:
        """
        Bắt đầu phiên phỏng vấn mô phỏng.
        """
        system_instruction = f"""Bạn là một nhà tuyển dụng AI chuyên nghiệp.
        Bạn sẽ phỏng vấn ứng viên dựa trên CV của họ.
        Hãy đặt câu hỏi từng câu một, bắt đầu bằng câu hỏi giới thiệu bản thân.
        Mục tiêu của bạn là đánh giá kỹ năng, kinh nghiệm và sự phù hợp với vị trí.
        Giữ vai trò là một nhà tuyển dụng, không đưa ra lời khuyên hay phân tích.
        Kết thúc phỏng vấn khi bạn cảm thấy đủ thông tin.

        Đây là CV của ứng viên:
        ```
        {cv_text[:3000]} # Giới hạn độ dài
        ```
        """
        if job_description:
            system_instruction += f"""
            Đây là mô tả công việc (nếu có):
            ```
            {job_description[:1000]}
            ```
            """
        
        # Tạo một phiên chat mới và lưu trữ
        chat_session = self.llm.start_chat(history=[
            {"role": "user", "parts": [system_instruction]},
            {"role": "model", "parts": ["Chào bạn, hãy giới thiệu một chút về bản thân bạn nhé!"]}
        ])
        self._chat_sessions[session_id] = chat_session
        return "Chào bạn, hãy giới thiệu một chút về bản thân bạn nhé!"

    def continue_interview(self, user_response: str, session_id: str = "default") -> str:
        """
        Tiếp tục phiên phỏng vấn với phản hồi của người dùng.
        """
        chat_session = self._chat_sessions.get(session_id)
        if not chat_session:
            return "Vui lòng bắt đầu phỏng vấn trước."
        
        try:
            response = chat_session.send_message(user_response)
            return response.text
        except Exception as e:
            print(f"Lỗi khi tiếp tục phỏng vấn: {e}")
            return f"Xin lỗi, có lỗi xảy ra trong quá trình phỏng vấn: {e}"

    def end_interview(self, session_id: str = "default") -> str:
        """
        Kết thúc phỏng vấn và đưa ra nhận xét ngắn gọn.
        """
        chat_session = self._chat_sessions.get(session_id)
        if not chat_session:
            return "Không có phiên phỏng vấn nào đang diễn ra."
        
        # Lấy lịch sử trò chuyện để tóm tắt
        history = chat_session.history
        
        # Tạo prompt tóm tắt
        summary_prompt = "Dựa trên lịch sử phỏng vấn sau đây, hãy đưa ra một nhận xét tổng quan ngắn gọn về ứng viên (khoảng 100 từ). Tập trung vào điểm mạnh và điểm cần cải thiện trong buổi phỏng vấn."
        
        # Gửi lịch sử và prompt tóm tắt đến LLM
        try:
            # Tạo một phiên chat mới cho việc tóm tắt để không làm ảnh hưởng phiên phỏng vấn chính
            summary_model = self.llm.__class__(self.llm.model_name) # Tạo instance mới của model
            summary_response = summary_model.generate_content(history + [{"role": "user", "parts": [summary_prompt]}])
            del self._chat_sessions[session_id] # Kết thúc phiên
            return summary_response.text
        except Exception as e:
            print(f"Lỗi khi kết thúc phỏng vấn và tóm tắt: {e}")
            del self._chat_sessions[session_id]
            return f"Phỏng vấn kết thúc. Xin lỗi, không thể tóm tắt do lỗi: {e}"