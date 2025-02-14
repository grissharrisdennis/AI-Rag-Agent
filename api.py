from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatPerplexity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
import hashlib
import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()

class YouTubeInput(BaseModel):
    url: str

# Function to get or create summary
def get_or_create_summary(video_id, transcript):
    # Set up ChatPerplexity
    try:
        chat_model = ChatPerplexity(model="sonar-reasoning", pplx_api_key=os.getenv("PPLX_API_KEY"), temperature=0.7)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing ChatPerplexity: {str(e)}")

    # Create prompt template.Edit This to get a structured summary
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

    chain = LLMChain(llm=chat_model, prompt=prompt)

    # Generate summary
    summary = chain.run(transcript=transcript)

    return summary

@app.post("/summarize")
async def summarize_video(input: YouTubeInput):
    try:
        loader = YoutubeLoader.from_youtube_url(input.url, add_video_info=False)
        documents = loader.load()

        if not documents:
            raise HTTPException(status_code=404, detail="No transcript found for this YouTube video.")

        transcript = documents[0].page_content
        video_id = hashlib.md5(input.url.encode()).hexdigest()

        summary = get_or_create_summary(video_id, transcript)
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
