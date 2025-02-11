import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma


# Initialize Streamlit session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Initialize Conversation Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def process_uploaded_file(uploaded_file):
    """Loads and splits PDF file into documents."""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())  # Write the uploaded file to temp file
            temp_file.flush()  # Ensure data is written before reading
            loader = PDFPlumberLoader(temp_file.name)  # Load the PDF from temp file
            documents = loader.load()
        return documents
    return None

def get_vs_retriever_from_docs(documents):
    """Creates a vector store retriever from documents."""
    embeddings = HuggingFaceEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    chroma_db = Chroma.from_documents(docs, embedding=embeddings)
    return chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def get_chat_model():
    """Returns an LLM model for generating responses."""
    return ChatOpenAI(
        model="gpt-3.5-turbo",  # Use a valid OpenAI model
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2,  # Keeps responses factual
        max_tokens=1024
    )

def get_related_content_query(retriever):
    """Creates a history-aware retriever to refine queries based on chat history."""
    llm = get_chat_model()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a search query based on the conversation so far. Chat history: {chat_history}"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)
    return history_aware_retriever

def get_context_aware_prompt():
    """Creates a retrieval chain with context-aware responses."""
    llm = get_chat_model()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the context provided:\n\n{context}"),
        ("human", "{input}")
    ])
    
    docs_chain = create_stuff_documents_chain(llm, prompt)
    return docs_chain

def get_response(query: str) -> str:
    """Generates a response based on user query and document context."""
    if st.session_state.vector_store is None:
        return "Please upload a document first."
    
    try:
        retriever = st.session_state.retriever
        history_aware_retriever = get_related_content_query(retriever)
        docs_chain = get_context_aware_prompt()
        
        # Create the retrieval chain
        rag_chain = create_retrieval_chain(history_aware_retriever, docs_chain)
        
        # Get chat history
        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        
        # Invoke the chain
        response = rag_chain.invoke({"input": query, "chat_history": chat_history})['answer']
        
        # Store in chat memory
        st.session_state.memory.chat_memory.add_user_message(query)
        st.session_state.memory.chat_memory.add_ai_message(response)
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

def init_chat_interface():
    """Initializes the Streamlit chat interface."""
    st.title("Academic Guidance AI Agent")

    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        documents = process_uploaded_file(uploaded_file)
        if documents:
            st.success(f"PDF processed successfully: {uploaded_file.name}")
        st.session_state.vector_store = get_vs_retriever_from_docs(documents)
        st.session_state.retriever = st.session_state.vector_store  # Set retriever
        st.success("Document uploaded and processed successfully!")

    # Display chat history
    for message in st.session_state.memory.chat_memory.messages:
        if isinstance(message, HumanMessage):
            st.write(f"**User:** {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"**System:** {message.content}")

    # Chat Input
    prompt = st.chat_input("Ask a question", disabled=st.session_state.vector_store is None)
    
    if prompt:
        response = get_response(prompt)
        st.write(f"**System:** {response}")

if __name__ == "__main__":
    init_chat_interface()










