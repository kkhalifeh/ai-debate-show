from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define input model for debate parameters


class DebateInput(BaseModel):
    topic: str
    agent1_position: str
    agent2_position: str
    agent1_llm: str
    agent2_llm: str
    language: str


@app.get("/")
async def root():
    return {"message": "AI Debate Talk Show Backend"}


@app.post("/start-debate")
async def start_debate(input: DebateInput):
    # Mock response for now
    return {
        "script": [
            f"Agent 1 ({input.agent1_llm}): Argues {input.agent1_position} on {input.topic} in {input.language}",
            f"Agent 2 ({input.agent2_llm}): Argues {input.agent2_position} on {input.topic} in {input.language}"
        ],
        "audio_urls": []
    }
