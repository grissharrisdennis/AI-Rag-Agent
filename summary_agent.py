import streamlit as st
import os
from langchain_community.chat_models import ChatPerplexity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import pdfplumber
import io

# Set up Streamlit page
st.set_page_config(page_title="Video Summarizer", layout="wide")
st.title("AI Video Summarizer")

# Initialize session state for storing summaries
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}

# Function to get or create summary
def get_or_create_summary(video_id, transcript):
    if video_id in st.session_state.summaries:
        return st.session_state.summaries[video_id]

    # Set up ChatPerplexity
    try:
        chat_model = ChatPerplexity(model="sonar-reasoning", pplx_api_key=os.getenv("PPLX_API_KEY"), temperature=0.7)
    except Exception as e:
        st.error(f"Error initializing ChatPerplexity: {e}")
        return None # Crucial:  Return None if chat_model fails

    # Create prompt template
    prompt_template = """
    Summarize the following video transcript in a structured manner:

    {transcript}

    Provide a summary with the following sections:
    1. Main Topic
    2. Key Points (bullet points)
    3. Conclusion

    Summary:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["transcript"])

    # Create LLMChain
    chain = LLMChain(llm=chat_model, prompt=prompt)

    # Generate summary
    summary = chain.run(transcript=transcript)

    # Store summary in session state
    st.session_state.summaries[video_id] = summary

    return summary

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}") # Report the error
        return None # Important: Return None if extraction fails
    return text

# Main app logic
input_type = st.radio("Choose input type:", ("YouTube URL", "Upload Transcript"))

# Initialize transcript and video_id variables
transcript = None
video_id = None

if input_type == "YouTube URL":
    youtube_url = st.text_input("Enter YouTube URL:")
    if youtube_url:
        try:
            loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
            documents = loader.load()  # Load the documents

            if documents: # Check if documents list is not empty
                transcript = documents[0].page_content
                video_id = hashlib.md5(youtube_url.encode()).hexdigest()
            else:
                st.error("No transcript found for this YouTube video.") # Inform the user
                transcript = None # set transcript to none for other checks to continue.
                video_id = None

        except Exception as e:
            st.error(f"Error loading YouTube video: {str(e)}")
            transcript = None # set transcript to none for other checks to continue.
            video_id = None

else:
    uploaded_file = st.file_uploader("Upload transcript document", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            transcript = extract_text_from_pdf(uploaded_file)
        else:
            transcript = uploaded_file.getvalue().decode("utf-8")

        if transcript:  # Check if transcript is not empty after PDF extraction
            video_id = hashlib.md5(transcript.encode()).hexdigest()
        else:
            st.error("Could not extract any text from the uploaded file.")
            video_id = None

if transcript and video_id:
    if st.button("Summarize"):
        with st.spinner("Generating summary..."):
            summary = get_or_create_summary(video_id, transcript)

            if summary:  # Only display if summary was successfully generated
                st.subheader("Video Summary")
                st.write(summary)
            else:
                st.error("Failed to generate summary.")



