from fastapi import FastAPI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate

app = FastAPI()

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


@app.get("/")
async def root():
    return {"message": "AI Debate Talk Show Backend"}


@app.post("/start-debate")
async def start_debate(input: DebateInput):
    # Simulate 4 turns (2 per agent)
    script = []
    for i in range(2):
        # Agent 1 turn
        agent1_response = debate_prompt.format(
            topic=input.topic,
            position=input.agent1_position,
            language=input.language
        )
        script.append(
            f"Agent 1 ({input.agent1_llm}) Turn {i+1}: {agent1_response}")

        # Agent 2 turn
        agent2_response = debate_prompt.format(
            topic=input.topic,
            position=input.agent2_position,
            language=input.language
        )
        script.append(
            f"Agent 2 ({input.agent2_llm}) Turn {i+1}: {agent2_response}")

    return {
        "script": script,
        "audio_urls": []  # Placeholder for Eleven Labs
    }
