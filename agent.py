import os
import streamlit as st
import tempfile
import re  # Import the regular expression module

from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv
import json

# Load variables from .env file
load_dotenv()




def init_session_state():
    """Initialize Streamlit session state variables."""
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("memory", ConversationBufferMemory(memory_key="chat_history", return_messages=True))

def extract_timestamp(text: str):
    """Extracts timestamp from text if present."""
    match = re.search(r"\(Refer Slide Time: (\d{1,2}:\d{2})\)", text)
    return match.group(1) if match else None

def process_uploaded_file(uploaded_file):
    """Loads and splits file into documents while preserving timestamps."""
    if not uploaded_file:
        return None

    try:
        # ✅ Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())  # Write the uploaded file to temp file
            temp_file.flush()  # Ensure data is written before reading
            loader = PDFPlumberLoader(temp_file.name)  # Load the PDF from temp file
            documents = loader.load()
        return documents

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

    finally:
        # ✅ Ensure file cleanup
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def get_vs_retriever_from_docs(documents):
    """Creates a vector store retriever from documents."""
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    chroma_db = Chroma.from_documents(docs, embedding=embeddings)
    return chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# def get_chat_model():
#     """Returns a GROQ LLM model for generating responses."""
#     return ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), temperature=0.2, max_tokens=1024)

def get_chat_model():
    """Returns a GROQ LLM model for generating responses."""
    return ChatPerplexity(
            temperature=0, pplx_api_key=os.getenv("PPLX_API_KEY"), model="llama-3-sonar-small-32k-online")


def get_related_content_query(retriever):
    """Creates a history-aware retriever to refine queries based on chat history."""
    llm = get_chat_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a search query based on the conversation so far. Chat history: {chat_history}"),
        ("human", "{input}"),
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_context_aware_prompt():
    """Creates a retrieval chain with context-aware responses."""
    llm = get_chat_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the context provided:\n\n{context}"),
        ("human", "{input}"),
    ])
    return create_stuff_documents_chain(llm, prompt)

def get_response(query: str) -> str:
    """Generates a response based on user query and document context."""
    if not st.session_state.vector_store:
        return "Please upload a document first."
    
    try:
        retriever = st.session_state.retriever
        history_aware_retriever = get_related_content_query(retriever)
        docs_chain = get_context_aware_prompt()
        rag_chain = create_retrieval_chain(history_aware_retriever, docs_chain)
        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})["answer"]
        st.session_state.memory.chat_memory.add_user_message(query)
        st.session_state.memory.chat_memory.add_ai_message(response)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def init_chat_interface():
    """Initializes the Streamlit chat interface."""
    st.title("Academic Guidance AI Agent")
    init_session_state()
    uploaded_file = st.file_uploader("Upload a PDF or text document", type=["pdf", "txt"])
    
    if uploaded_file:
        documents = process_uploaded_file(uploaded_file)
        if documents:
            st.success(f"File processed successfully: {uploaded_file.name}")
            st.session_state.vector_store = get_vs_retriever_from_docs(documents)
            st.session_state.retriever = st.session_state.vector_store
            st.success("Document uploaded and processed successfully!")
    
    for message in st.session_state.memory.chat_memory.messages:
        st.write(f"**{'User' if isinstance(message, HumanMessage) else 'System'}:** {message.content}")
    
    prompt = st.chat_input("Ask a question", disabled=st.session_state.vector_store is None)
    if prompt:
        response = get_response(prompt)
        st.write(f"**System:** {response}")

if __name__ == "__main__":
    init_chat_interface()











