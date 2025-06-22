from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import requests
import uuid

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Check API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in .env")

# Define input model for debate parameters


class DebateInput(BaseModel):
    topic: str
    agent1_position: str
    agent2_position: str
    agent1_llm: str
    agent2_llm: str
    language: str


# Define prompt template for debate arguments
debate_prompt = PromptTemplate(
    input_variables=["topic", "position", "language"],
    template="In {language}, argue the position '{position}' on the topic '{topic}' in a concise, persuasive manner (100-150 words)."
)

# Eleven Labs audio generation


def generate_audio(text: str, voice_id: str) -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate audio: {response.text}")

    # Save audio to temporary file
    filename = f"audio_{uuid.uuid4()}.mp3"
    filepath = os.path.join("/tmp", filename)
    with open(filepath, "wb") as f:
        f.write(response.content)
    return f"/audio/{filename}"


@app.get("/")
async def root():
    return {"message": "AI Debate Talk Show Backend"}


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    filepath = os.path.join("/tmp", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(filepath, media_type="audio/mpeg")


@app.post("/start-debate")
async def start_debate(input: DebateInput):
    # Validate LLM selection
    valid_llms = ["gpt-3.5-turbo", "gpt-4"]
    if input.agent1_llm not in valid_llms or input.agent2_llm not in valid_llms:
        raise HTTPException(
            status_code=400, detail=f"Selected LLM not supported. Choose from {valid_llms}")

    # Initialize LLMs
    llm1 = ChatOpenAI(model=input.agent1_llm, api_key=OPENAI_API_KEY)
    llm2 = ChatOpenAI(model=input.agent2_llm, api_key=OPENAI_API_KEY)

    # Simulate 4 turns (2 per agent)
    script = []
    audio_urls = []
    for i in range(2):
        # Agent 1 turn - FIXED: Removed await and properly access content
        agent1_response_obj = llm1.invoke(
            debate_prompt.format(
                topic=input.topic,
                position=input.agent1_position,
                language=input.language
            )
        )
        agent1_response = agent1_response_obj.content
        script.append(
            f"Agent 1 ({input.agent1_llm}) Turn {i+1}: {agent1_response}")
        audio_urls.append(generate_audio(
            agent1_response, "ordbVDppyuwp96ZjvQOM"))  # Hakeem

        # Agent 2 turn - FIXED: Removed await and properly access content
        agent2_response_obj = llm2.invoke(
            debate_prompt.format(
                topic=input.topic,
                position=input.agent2_position,
                language=input.language
            )
        )
        agent2_response = agent2_response_obj.content
        script.append(
            f"Agent 2 ({input.agent2_llm}) Turn {i+1}: {agent2_response}")
        audio_urls.append(generate_audio(
            agent2_response, "QRq5hPRAKf5ZhSlTBH6r"))  # Yahya

    return {
        "script": script,
        "audio_urls": audio_urls
    }
