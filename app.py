import streamlit as st
st.title("AI Resume Analyzer")
uploaded_file = st.file_uploader(
    "Upload your Resume",
    type=["pdf", "docx"]
)
if uploaded_file is not None:
    st.success("Resume uploaded successfully!")
    # Call your resume analysis function
    # result = analyze_resume(uploaded_file)
    # Display results
    # st.write(result)