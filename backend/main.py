from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import OpenAI
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
    turns_per_agent: int
    voice_provider: str
    min_words: int
    max_words: int
    debate_intensity: str = "heated"  # mild, heated, explosive
    use_personas: bool = True
    use_emotional_language: bool = True
    use_debate_tactics: bool = True

# Define prompt templates for customizable heated debate


def get_opening_prompt(min_words: int, max_words: int, debate_config):
    # Build intensity language
    intensity_map = {
        "mild": "civilly discuss and argue",
        "heated": "passionately and assertively argue",
        "explosive": "AGGRESSIVELY and FIERCELY argue"
    }
    intensity_phrase = intensity_map.get(
        debate_config.debate_intensity, "argue")

    # Build persona instructions
    persona_text = ""
    if debate_config.use_personas:
        persona_text = "You are a passionate expert who gets fired up about this topic. "

    # Build emotional language instructions
    emotional_text = ""
    if debate_config.use_emotional_language:
        emotional_text = "Show strong conviction and don't hold back your emotions. "

    # Build debate tactics instructions
    tactics_text = ""
    if debate_config.use_debate_tactics:
        tactics_text = "Use rhetorical questions, challenge assumptions, and demand evidence. "

    template = f"In {{language}}, {persona_text}{emotional_text}You are in a debate. {intensity_phrase.upper()} the position '{{position}}' on the topic '{{topic}}'. {tactics_text}Be confident and use strong language. ({min_words}-{max_words} words)."

    return PromptTemplate(
        input_variables=["topic", "position", "language"],
        template=template
    )


def get_rebuttal_prompt(min_words: int, max_words: int, debate_config):
    # Build intensity language for rebuttals
    intensity_map = {
        "mild": "respectfully counter their argument",
        "heated": "AGGRESSIVELY counter their argument",
        "explosive": "DEMOLISH their argument with FURY"
    }
    intensity_phrase = intensity_map.get(
        debate_config.debate_intensity, "counter their argument")

    # Build persona instructions
    persona_text = ""
    if debate_config.use_personas:
        persona_text = "You are a passionate expert who gets fired up about this topic. "

    # Build emotional language instructions
    emotional_text = ""
    if debate_config.use_emotional_language:
        emotional_text = "Express frustration with your opponent's weak arguments. Show that you're getting more heated as the debate continues. "

    # Build debate tactics instructions
    tactics_text = ""
    if debate_config.use_debate_tactics:
        tactics_text = "Use rhetorical questions, challenge their sources, demand evidence, and point out logical fallacies. "

    template = f"In {{language}}, {persona_text}{emotional_text}Your opponent just said: '{{opponent_argument}}'. {intensity_phrase.upper()} while defending your position '{{position}}' on '{{topic}}'. {tactics_text}Be confrontational but professional. ({min_words}-{max_words} words)."

    return PromptTemplate(
        input_variables=["topic", "position", "language", "opponent_argument"],
        template=template
    )

# Audio generation


def generate_audio(text: str, voice_id: str, provider: str) -> str:
    filename = f"audio_{uuid.uuid4()}.mp3"
    filepath = os.path.join("/tmp", filename)

    if provider == "eleven_labs":
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        response = requests.post(url, json=data, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=500, detail=f"Eleven Labs failed: {response.text}")
        with open(filepath, "wb") as f:
            f.write(response.content)

    elif provider == "openai":
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice_id.lower(),  # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        response.stream_to_file(filepath)

    else:
        raise HTTPException(status_code=400, detail="Invalid voice provider")

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

    # Validate turns_per_agent
    if input.turns_per_agent < 1 or input.turns_per_agent > 5:
        raise HTTPException(
            status_code=400, detail="Turns per agent must be between 1 and 5")

    # Validate word count
    if input.min_words < 50 or input.max_words > 500 or input.min_words > input.max_words:
        raise HTTPException(
            status_code=400, detail="Word count must be between 50-500, with min <= max")

    # Validate voice provider
    valid_providers = ["eleven_labs", "openai"]
    if input.voice_provider not in valid_providers:
        raise HTTPException(
            status_code=400, detail=f"Voice provider not supported. Choose from {valid_providers}")

    # Initialize LLMs
    try:
        llm1 = ChatOpenAI(model=input.agent1_llm, api_key=OPENAI_API_KEY)
        llm2 = ChatOpenAI(model=input.agent2_llm, api_key=OPENAI_API_KEY)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize LLM: {str(e)}")

    # Get prompts with custom word count and debate configuration
    opening_prompt = get_opening_prompt(
        input.min_words, input.max_words, input)
    rebuttal_prompt = get_rebuttal_prompt(
        input.min_words, input.max_words, input)

    # Generate debate turns
    script = []
    audio_urls = []
    agent1_last_response = ""
    agent2_last_response = ""

    for i in range(input.turns_per_agent):
        try:
            # Agent 1 turn
            if i == 0:
                # First turn - opening statement
                prompt_to_use = opening_prompt.format(
                    topic=input.topic,
                    position=input.agent1_position,
                    language=input.language
                )
            else:
                # Subsequent turns - rebuttal to agent 2's last response
                prompt_to_use = rebuttal_prompt.format(
                    topic=input.topic,
                    position=input.agent1_position,
                    language=input.language,
                    opponent_argument=agent2_last_response
                )

            agent1_response_obj = llm1.invoke(prompt_to_use)
            agent1_response = agent1_response_obj.content
            agent1_last_response = agent1_response

            script.append(
                f"Agent 1 ({input.agent1_llm}) Turn {i+1}: {agent1_response}")
            voice_id = "ordbVDppyuwp96ZjvQOM" if input.voice_provider == "eleven_labs" else "onyx"
            audio_urls.append(generate_audio(
                agent1_response, voice_id, input.voice_provider))

            # Agent 2 turn
            if i == 0:
                # First turn - opening statement (but can reference agent 1's opening)
                prompt_to_use = rebuttal_prompt.format(
                    topic=input.topic,
                    position=input.agent2_position,
                    language=input.language,
                    opponent_argument=agent1_response
                )
            else:
                # Subsequent turns - rebuttal to agent 1's response
                prompt_to_use = rebuttal_prompt.format(
                    topic=input.topic,
                    position=input.agent2_position,
                    language=input.language,
                    opponent_argument=agent1_response
                )

            agent2_response_obj = llm2.invoke(prompt_to_use)
            agent2_response = agent2_response_obj.content
            agent2_last_response = agent2_response

            script.append(
                f"Agent 2 ({input.agent2_llm}) Turn {i+1}: {agent2_response}")
            voice_id = "QRq5hPRAKf5ZhSlTBH6r" if input.voice_provider == "eleven_labs" else "echo"
            audio_urls.append(generate_audio(
                agent2_response, voice_id, input.voice_provider))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"LLM or audio generation failed: {str(e)}")

    return {
        "script": script,
        "audio_urls": audio_urls
    }
