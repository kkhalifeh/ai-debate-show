from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.prompts import PromptTemplate

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow front-end origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

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
    template="[{language}] Agent argues {position} on {topic}: This is a mock argument for {position}."
)

# Placeholder function for Eleven Labs text-to-speech


def generate_audio(text: str, voice_id: str) -> str:
    # Temporary placeholder returning sample audio paths
    return "/audio/sample_agent1.mp3" if voice_id == "agent1_voice" else "/audio/sample_agent2.mp3"


@app.get("/")
async def root():
    return {"message": "AI Debate Talk Show Backend"}


@app.post("/start-debate")
async def start_debate(input: DebateInput):
    # Simulate 4 turns (2 per agent)
    script = []
    audio_urls = []
    for i in range(2):
        # Agent 1 turn
        agent1_response = debate_prompt.format(
            topic=input.topic,
            position=input.agent1_position,
            language=input.language
        )
        script.append(
            f"Agent 1 ({input.agent1_llm}) Turn {i+1}: {agent1_response}")
        audio_urls.append(generate_audio(agent1_response, "agent1_voice"))

        # Agent 2 turn
        agent2_response = debate_prompt.format(
            topic=input.topic,
            position=input.agent2_position,
            language=input.language
        )
        script.append(
            f"Agent 2 ({input.agent2_llm}) Turn {i+1}: {agent2_response}")
        audio_urls.append(generate_audio(agent2_response, "agent2_voice"))

    return {
        "script": script,
        "audio_urls": audio_urls
    }
