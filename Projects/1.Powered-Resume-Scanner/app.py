# app.py
import streamlit as st
from file_parser import parse_resume
from nlp_processor import generate_suggestions, extract_skills

st.set_page_config(page_title="AI-Powered Resume Scanner 🚀", layout="wide")

st.title("🚀 AI-Powered Resume Scanner")
st.markdown("Upload your resume (PDF, DOCX, TXT) to get AI-powered suggestions for improvement!")

uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", "docx", "txt"])

# Option to input job description
st.markdown("---")
st.header("✨ Optional: Job Description for Targeted Analysis")
job_description = st.text_area(
    "Paste the Job Description here (optional, for matching skills)",
    height=200,
    placeholder="e.g., 'Seeking a Python Developer with strong NLP skills and experience with cloud platforms...'"
)
st.markdown("---")


if uploaded_file is not None:
    with st.spinner("Analyzing your resume..."):
        resume_text, error_message = parse_resume(uploaded_file)

        if error_message:
            st.error(error_message)
        elif resume_text:
            st.success("Resume uploaded and parsed successfully!")
            # st.subheader("Parsed Text:")
            # st.text_area("Full text from your resume", resume_text, height=300, disabled=True)

            st.subheader("📊 Analysis Results & Suggestions:")
            suggestions = generate_suggestions(resume_text, job_description)

            # Display suggestions
            if suggestions:
                for i, suggestion in enumerate(suggestions):
                    st.write(f"- {suggestion}")
            else:
                st.info("No specific suggestions generated. Your resume looks good!")

            st.markdown("---")
            st.subheader("💡 Identified Skills:")
            identified_skills = extract_skills(resume_text)
            if identified_skills:
                st.write(f"Your resume mentions: **{', '.join(identified_skills)}**")
            else:
                st.write("No specific skills could be automatically identified. Make sure they are clearly listed.")

        else:
            st.warning("Could not extract text from the uploaded file. Please check the file content.")

else:
    st.info("Please upload a resume to begin the analysis.")

st.markdown("---")
st.markdown("Developed with ❤️ using Python, SpaCy, and Streamlit.")