# frontend/app.py
import streamlit as st
import requests
import base64
import json
import os
import uuid # Để tạo session_id cho chatbot

# --- Cấu hình API Endpoint (Thay thế bằng URL API Gateway của bạn) ---
# LƯU Ý: Trong môi trường sản phẩm, hãy sử dụng biến môi trường hoặc cấu hình an toàn hơn.
# Ví dụ: https://xxxxxxxxx.execute-api.ap-southeast-1.amazonaws.com/dev
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "YOUR_API_GATEWAY_INVOKE_URL_HERE") 

st.set_page_config(layout="wide", page_title="AI Resume Analyzer & Career Assistant")

st.title("🧠 AI Resume Analyzer & Career Assistant 🚀")
st.markdown("Tải CV của bạn lên để nhận phân tích, lời khuyên và luyện phỏng vấn AI.")

# --- Khởi tạo session_id cho chatbot nếu chưa có ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Tab Navigation ---
tab1, tab2 = st.tabs(["Phân tích CV", "Chatbot Phỏng vấn"])

with tab1:
    st.header("Upload CV của bạn")
    uploaded_file = st.file_uploader("Chọn file CV (PDF hoặc DOCX)", type=["pdf", "docx"])

    if uploaded_file is not None:
        file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        st.write(file_details)

        # Đọc nội dung file và mã hóa Base64
        file_content = uploaded_file.read()
        encoded_content = base64.b64encode(file_content).decode('utf-8')
        
        file_extension = uploaded_file.name.split('.')[-1].lower()

        st.info("Đang phân tích CV của bạn, vui lòng chờ trong giây lát...")

        try:
            # Gửi yêu cầu đến API Gateway Lambda
            response = requests.post(
                f"{API_GATEWAY_URL}/analyze-cv",
                json={
                    "file_content": encoded_content,
                    "file_type": file_extension
                },
                headers={
                    "X-Session-Id": st.session_state.session_id # Gửi session_id cho backend
                }
            )
            response.raise_for_status() # Ném lỗi cho các mã trạng thái HTTP xấu
            result = response.json()

            st.success("Phân tích hoàn tất!")

            st.subheader("Nội dung CV đã phân tích:")
            st.text_area("Văn bản CV", result.get('parsed_text', 'Không có nội dung'), height=300)

            st.subheader("Kỹ năng được trích xuất:")
            st.write(result.get('extracted_skills', ['Không tìm thấy kỹ năng']))

            st.subheader("Lời khuyên từ AI Career Advisor:")
            st.markdown(result.get('career_advice', 'Không có lời khuyên.'))

            st.subheader("Gợi ý ngành nghề phù hợp:")
            st.markdown(result.get('career_suggestions', 'Không có gợi ý ngành nghề.'))

            st.subheader("Các công việc phù hợp:")
            matching_jobs = result.get('matching_jobs', [])
            if matching_jobs:
                for job in matching_jobs:
                    st.write(f"**{job['title']}** (Điểm phù hợp: {job['score']:.2f})")
                    st.markdown(f"_{job['description']}_")
            else:
                st.write("Không tìm thấy công việc phù hợp.")

        except requests.exceptions.RequestException as e:
            st.error(f"Lỗi khi gọi API: {e}. Vui lòng kiểm tra API Gateway URL và kết nối mạng.")
            if e.response is not None:
                st.error(f"Mã lỗi HTTP: {e.response.status_code}")
                st.error(f"Phản hồi từ API: {e.response.text}")
        except json.JSONDecodeError:
            st.error("Lỗi: Phản hồi từ API không phải là JSON hợp lệ.")
            if response is not None:
                st.error(f"Phản hồi thô: {response.text}")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi không mong muốn: {e}")

with tab2:
    st.header("Chatbot Phỏng vấn AI")

    # Lưu trữ lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cv_text_for_interview" not in st.session_state:
        st.session_state.cv_text_for_interview = ""
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False

    # Hiển thị tin nhắn cũ
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    col1, col2 = st.columns([3, 1])
    with col1:
        cv_text_input = st.text_area("Dán nội dung CV của bạn vào đây để chatbot tham khảo (tùy chọn, nhưng khuyến khích để phỏng vấn sát hơn):", height=200, key="cv_text_for_interview_input")
    with col2:
        job_description_input = st.text_area("Mô tả công việc (tùy chọn):", height=200, key="job_description_for_interview_input")

    if st.button("Bắt đầu Phỏng vấn", key="start_interview_button"):
        if not cv_text_input:
            st.warning("Vui lòng dán nội dung CV để bắt đầu phỏng vấn.")
        else:
            st.session_state.cv_text_for_interview = cv_text_input
            st.session_state.job_description_for_interview = job_description_input
            st.session_state.messages = [] # Xóa lịch sử chat cũ
            st.session_state.interview_started = True
            
            with st.spinner("Đang khởi tạo phiên phỏng vấn..."):
                try:
                    response = requests.post(
                        f"{API_GATEWAY_URL}/interview",
                        json={
                            "action": "start",
                            "cv_text": st.session_state.cv_text_for_interview,
                            "job_description": st.session_state.job_description_for_interview
                        },
                        headers={
                            "X-Session-Id": st.session_state.session_id # Gửi session_id cho backend
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.session_state.messages.append({"role": "assistant", "content": result.get('response', 'Lỗi khởi tạo.')})
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Lỗi khi bắt đầu phỏng vấn: {e}")
                    if e.response is not None:
                        st.error(f"Mã lỗi HTTP: {e.response.status_code}")
                        st.error(f"Phản hồi từ API: {e.response.text}")
                    st.session_state.interview_started = False
                except Exception as e:
                    st.error(f"Lỗi không mong muốn khi bắt đầu phỏng vấn: {e}")
                    st.session_state.interview_started = False

    if st.session_state.interview_started:
        user_input = st.chat_input("Trả lời câu hỏi của nhà tuyển dụng:")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("AI đang suy nghĩ..."):
                try:
                    response = requests.post(
                        f"{API_GATEWAY_URL}/interview",
                        json={
                            "action": "continue",
                            "user_response": user_input
                        },
                        headers={
                            "X-Session-Id": st.session_state.session_id # Gửi session_id cho backend
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    ai_response = result.get('response', 'Lỗi phản hồi.')
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    with st.chat_message("assistant"):
                        st.markdown(ai_response)
                except requests.exceptions.RequestException as e:
                    st.error(f"Lỗi khi tiếp tục phỏng vấn: {e}")
                    if e.response is not None:
                        st.error(f"Mã lỗi HTTP: {e.response.status_code}")
                        st.error(f"Phản hồi từ API: {e.response.text}")
                except Exception as e:
                    st.error(f"Lỗi không mong muốn khi tiếp tục phỏng vấn: {e}")

        if st.button("Kết thúc Phỏng vấn", key="end_interview_button"):
            with st.spinner("Đang kết thúc phỏng vấn và tóm tắt..."):
                try:
                    response = requests.post(
                        f"{API_GATEWAY_URL}/interview",
                        json={
                            "action": "end"
                        },
                        headers={
                            "X-Session-Id": st.session_state.session_id # Gửi session_id cho backend
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.session_state.messages.append({"role": "assistant", "content": result.get('response', 'Lỗi kết thúc.')})
                    st.session_state.interview_started = False
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Lỗi khi kết thúc phỏng vấn: {e}")
                    if e.response is not None:
                        st.error(f"Mã lỗi HTTP: {e.response.status_code}")
                        st.error(f"Phản hồi từ API: {e.response.text}")
                except Exception as e:
                    st.error(f"Lỗi không mong muốn khi kết thúc phỏng vấn: {e}")
