# AI-Rag-Agent  -  Agent.py

### Uses Langchain Framework for the agents and PERPLEXITY API for the LLM.

## create a env
<code>python -m venv env </code>
<code>env\Scripts\activate</code>

## create a .env file to save OpenAI Api Key to access the llm for the agent

## Install Depedencies
<code>pip install requirements.txt</code>

## Run Agent
<code>streamlit run agent.py </code> Run this command.

### Uses PERPLEXITY API
### upload documents like transcripts or any other pdf  on the server to test the agent.
### Ask a about anything on the documents uploaded.


#  Summary Agent  - summary-agent.py

### Uses PERPLEXITY API

## Run Agent
<code>streamlit run summary_agent.py </code> Run this command.

# API  - api.py

### Uses PERPLEXITY API and FastApi Framework.
<code>fastapi dev api.py </code> Run this command.
paste this to see the docuemntation <code> http://127.0.0.1:8000/docs </code>

### API for video summmarisation.Uses youtbe video link to summarise the video.


