# AI-RAG-Agent

AI-RAG-Agent is a powerful Retrieval-Augmented Generation (RAG) system leveraging the **LangChain** framework and the **Perplexity API** for efficient document-based querying and summarization. This repository includes multiple components:

- **Agent**: Handles document-based queries.
- **Summary Agent**: Summarizes text and video content.
- **API**: Provides FastAPI-based endpoints for video summarization.

---

## Installation

### 1. Create a Virtual Environment
```sh
python -m venv env
```
Activate the environment:
- **Windows**:
  ```sh
  env\Scripts\activate
  ```
- **Mac/Linux**:
  ```sh
  source env/bin/activate
  ```

### 2. Set Up API Keys
Create a `.env` file in the root directory and add your **PERPLEXITY API Key** to access the LLM.
```sh
PPLX_API_KEY=your_api_key_here
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

---

## Running the Agents

### 1. AI RAG Agent
This agent allows users to upload documents (such as PDFs or transcripts) and ask questions based on their content.
```sh
streamlit run agent.py
```

### 2. Summary Agent
This agent summarizes text-based or video-based content using the **Perplexity API**.
```sh
streamlit run summary_agent.py
```

---

## API for Video Summarization

The API uses **FastAPI** to provide video summarization services. It extracts key points from YouTube videos.

### Running the API Server
```sh
fastapi dev api.py
```

### API Documentation
Once the server is running, you can access the API documentation at:
```
http://127.0.0.1:8000/docs
```

---

## Features
- **Document-based QA**: Upload PDFs and query them using AI.
- **Text Summarization**: Extract key information from uploaded documents.
- **YouTube Video Summarization**: Input a YouTube link and get a summarized output.
- **FastAPI Integration**: Provides REST API endpoints for easy access.

---

## License
This project is open-source. Feel free to modify and extend it as needed.

---


For any issues or feature requests, feel free to open an **issue** or submit a **pull request**.




