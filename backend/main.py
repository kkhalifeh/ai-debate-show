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
import random

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in .env")


class VoiceSettings(BaseModel):
    speed: float = 1.0
    pitch: float = 1.0
    stability: float = 0.5
    similarity_boost: float = 0.75


class DebateInput(BaseModel):
    topic: str
    moderator_name: str
    agent1_name: str
    agent2_name: str
    agent1_position: str
    agent2_position: str
    agent1_llm: str
    agent2_llm: str
    agent1_tone: str = "neutral"
    agent2_tone: str = "neutral"
    language: str
    turns_per_agent: int
    voice_provider: str
    min_words: int
    max_words: int
    moderator_prompt: str
    agent1_prompt: str
    agent2_prompt: str
    moderator_intervention: str = "frequent"
    agent1_voice: VoiceSettings = VoiceSettings()
    agent2_voice: VoiceSettings = VoiceSettings()
    moderator_voice: VoiceSettings = VoiceSettings()
    # Optional: Add context length control (defaults to 2 for backwards compatibility)
    context_length: int = 2


class ConversationEntry(BaseModel):
    speaker: str
    content: str
    turn_number: int
    entry_type: str  # "opening", "argument", "intervention"


def clean_agent_response(response: str, agent_name: str) -> str:
    """Remove agent name from the beginning of response to avoid duplication"""
    response = response.strip()

    # Common patterns to remove (English names):
    patterns_to_remove = [
        f"{agent_name}: ",
        f"{agent_name.lower()}: ",
        f"{agent_name.upper()}: ",
        f"**{agent_name}**: ",
        f"**{agent_name.lower()}**: ",
        f"**{agent_name.upper()}**: ",
    ]

    # Check English patterns first
    for pattern in patterns_to_remove:
        if response.startswith(pattern):
            response = response[len(pattern):].strip()
            return response

    # If not found, use more generic approach for translated names
    # Look for any name followed by colon at the start
    import re

    # Pattern: Any word(s) followed by colon and space at the beginning
    # This catches translated names like "أحمد: " or "ليلى: "
    pattern = r'^[^\:]+:\s*'

    # Check if the response starts with "Name: " pattern
    match = re.match(pattern, response)
    if match:
        # Extract what comes before the colon
        potential_name = match.group(0)[:-1].strip()  # Remove ": " part

        # Only remove if it's a reasonable length for a name (2-20 characters)
        # This prevents removing things like "Question: " or long sentences
        if 2 <= len(potential_name) <= 20 and not any(char in potential_name for char in ['?', '!', '.', '،', '؟']):
            response = response[len(match.group(0)):].strip()

    return response


def generate_audio(text: str, voice_id: str, provider: str, settings: VoiceSettings, tone: str = "neutral") -> str:
    filename = f"audio_{uuid.uuid4()}.mp3"
    filepath = os.path.join("/tmp", filename)

    # Adjust voice settings based on tone
    adjusted_settings = VoiceSettings(
        speed=settings.speed,
        pitch=settings.pitch,
        stability=settings.stability,
        similarity_boost=settings.similarity_boost
    )

    # Tone-based adjustments
    if tone == "assertive":
        adjusted_settings.speed = min(
            settings.speed * 1.1, 2.0)  # Slightly faster
        adjusted_settings.stability = min(
            settings.stability + 0.2, 1.0)  # More stable
        if provider == "openai":
            adjusted_settings.pitch = min(
                settings.pitch * 0.95, 2.0)  # Slightly lower pitch

    elif tone == "emotional":
        adjusted_settings.speed = max(
            settings.speed * 0.9, 0.5)  # Slightly slower
        adjusted_settings.stability = max(
            settings.stability - 0.2, 0.0)  # Less stable (more variation)
        if provider == "openai":
            adjusted_settings.pitch = max(
                settings.pitch * 1.05, 0.5)  # Slightly higher pitch

    elif tone == "dramatic":
        adjusted_settings.speed = max(
            settings.speed * 0.85, 0.5)  # Slower for drama
        adjusted_settings.stability = max(
            settings.stability - 0.3, 0.0)  # Much less stable
        adjusted_settings.similarity_boost = min(
            settings.similarity_boost + 0.1, 1.0)  # More expressive
        if provider == "openai":
            adjusted_settings.pitch = settings.pitch * 0.9  # Lower pitch for drama

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
                "stability": adjusted_settings.stability,
                "similarity_boost": adjusted_settings.similarity_boost,
                "style": 0.2 if tone == "dramatic" else 0.1 if tone in ["assertive", "emotional"] else 0.0,
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
            voice=voice_id.lower(),
            input=text,
            speed=adjusted_settings.speed
            # Note: OpenAI TTS doesn't support pitch parameter in the current API
            # The pitch adjustment would need to be done post-processing
        )
        response.stream_to_file(filepath)

    else:
        raise HTTPException(status_code=400, detail="Invalid voice provider")

    return f"/audio/{filename}"


def get_conversation_context(conversation_history: list, context_length: int) -> str:
    """Get recent conversation context for agents"""
    if not conversation_history or context_length <= 0:
        return ""

    # Get the last 'context_length' entries
    recent_entries = conversation_history[-context_length:]
    context_parts = []

    for entry in recent_entries:
        context_parts.append(f"{entry.speaker}: {entry.content}")

    return "\n".join(context_parts)


def build_agent_prompt(
    base_prompt: str,
    topic: str,
    position: str,
    language: str,
    opponent_last_response: str,
    moderator_question: str,
    conversation_context: str,
    min_words: int,
    max_words: int,
    tone: str
) -> str:
    """Build enhanced agent prompt with conversation context and moderator questions"""

    # Start with the base prompt template
    enhanced_prompt = base_prompt.format(
        topic=topic,
        position=position,
        language=language,
        opponent_argument=opponent_last_response,
        min_words=min_words,
        max_words=max_words,
        tone=tone
    )

    # Add moderator question if present
    if moderator_question.strip():
        enhanced_prompt += f"\n\nIMPORTANT: The moderator has asked you this specific question: \"{moderator_question}\"\nPlease address this question in your response while also responding to your opponent's argument."

    # Add conversation context if available
    if conversation_context.strip():
        enhanced_prompt += f"\n\nFor additional context, here are the recent exchanges in this debate:\n{conversation_context}"

    return enhanced_prompt


def build_moderator_prompt(
    base_prompt: str,
    context: dict,
    conversation_history: list,
    target_agent: str,
    last_speaker: str,
    last_response: str
) -> str:
    """Build moderator prompt with full conversation context"""

    # Build conversation summary for moderator
    if conversation_history:
        conv_summary = "\n".join([f"{entry.speaker}: {entry.content[:100]}..."
                                 for entry in conversation_history[-3:]])  # Last 3 entries
    else:
        conv_summary = "No previous conversation."

    # Create enhanced moderator prompt
    enhanced_prompt = f"""
{base_prompt.format(**context)}

CONVERSATION SO FAR:
{conv_summary}

{last_speaker} just said: "{last_response[:200]}..."

As the moderator, provide a brief transition and ask {target_agent} a relevant follow-up question that:
1. Acknowledges what {last_speaker} just argued
2. Poses a specific question to {target_agent} 
3. Helps advance the debate

Keep your response under 100 words and in {context['language']}.
"""

    return enhanced_prompt


@app.get("/")
async def root():
    return {"message": "AI Debate Talk Show Backend with Enhanced Moderator Interactions"}


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    filepath = os.path.join("/tmp", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(filepath, media_type="audio/mpeg")


@app.post("/start-debate")
async def start_debate(input: DebateInput):
    valid_llms = ["gpt-3.5-turbo", "gpt-4"]
    if input.agent1_llm not in valid_llms or input.agent2_llm not in valid_llms:
        raise HTTPException(
            status_code=400, detail=f"Selected LLM not supported. Choose from {valid_llms}")

    if input.turns_per_agent < 1 or input.turns_per_agent > 5:
        raise HTTPException(
            status_code=400, detail="Turns per agent must be between 1 and 5")

    if input.min_words < 50 or input.max_words > 500 or input.min_words > input.max_words:
        raise HTTPException(
            status_code=400, detail="Word count must be between 50-500, with min <= max")

    valid_providers = ["eleven_labs", "openai"]
    if input.voice_provider not in valid_providers:
        raise HTTPException(
            status_code=400, detail=f"Voice provider not supported. Choose from {valid_providers}")

    try:
        moderator = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
        llm1 = ChatOpenAI(model=input.agent1_llm, api_key=OPENAI_API_KEY)
        llm2 = ChatOpenAI(model=input.agent2_llm, api_key=OPENAI_API_KEY)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize LLM: {str(e)}")

    # Initialize conversation tracking
    script = []
    audio_urls = []
    conversation_history = []
    agent1_last_response = ""
    agent2_last_response = ""
    moderator_last_question = ""
    turn_counter = 0

    # Moderator opening
    mod_context = {
        'topic': input.topic,
        'agent1': input.agent1_name,
        'agent2': input.agent2_name,
        'language': input.language,
        'moderator_name': input.moderator_name,
        'min_words': input.min_words,
        'max_words': input.max_words
    }

    mod_opening_prompt = input.moderator_prompt.format(**mod_context)
    mod_opening = moderator.invoke(mod_opening_prompt).content
    # Clean moderator response too
    mod_opening = clean_agent_response(mod_opening, input.moderator_name)

    # Record in conversation history
    opening_entry = ConversationEntry(
        speaker=input.moderator_name,
        content=mod_opening,
        turn_number=turn_counter,
        entry_type="opening"
    )
    conversation_history.append(opening_entry)

    script.append(f"{input.moderator_name}: {mod_opening}")
    audio_urls.append(generate_audio(
        mod_opening,
        "fable" if input.voice_provider == "openai" else "rfkTsdZrVWEVhDycUYn9",
        input.voice_provider,
        input.moderator_voice,
        "neutral"
    ))

    for turn in range(input.turns_per_agent):
        try:
            turn_counter += 1

            # === AGENT 1 TURN ===
            conversation_context = get_conversation_context(
                conversation_history, input.context_length)

            # Build enhanced prompt for Agent 1
            agent1_enhanced_prompt = build_agent_prompt(
                base_prompt=input.agent1_prompt,
                topic=input.topic,
                position=input.agent1_position,
                language=input.language,
                opponent_last_response=agent2_last_response,
                moderator_question=moderator_last_question if turn > 0 else "",
                conversation_context=conversation_context,
                min_words=input.min_words,
                max_words=input.max_words,
                tone=input.agent1_tone
            )

            agent1_response = llm1.invoke(agent1_enhanced_prompt).content
            # Clean response: remove agent name if it appears at the start
            agent1_response = clean_agent_response(
                agent1_response, input.agent1_name)
            agent1_last_response = agent1_response

            # Record Agent 1's response
            agent1_entry = ConversationEntry(
                speaker=input.agent1_name,
                content=agent1_response,
                turn_number=turn_counter,
                entry_type="argument"
            )
            conversation_history.append(agent1_entry)

            script.append(f"{input.agent1_name}: {agent1_response}")
            audio_urls.append(generate_audio(
                agent1_response,
                "alloy" if input.voice_provider == "openai" else "UgBBYS2sOqTuMpoF3BR0",
                input.voice_provider,
                input.agent1_voice,
                input.agent1_tone
            ))

            # Clear previous moderator question since it was addressed
            moderator_last_question = ""

            # === MODERATOR INTERVENTION (before Agent 2) ===
            if input.moderator_intervention == "frequent" or (input.moderator_intervention == "occasional" and turn % 2 == 1):
                # Build context-aware moderator prompt
                mod_followup_prompt = build_moderator_prompt(
                    base_prompt=input.moderator_prompt,
                    context=mod_context,
                    conversation_history=conversation_history,
                    target_agent=input.agent2_name,
                    last_speaker=input.agent1_name,
                    last_response=agent1_response
                )

                mod_followup = moderator.invoke(mod_followup_prompt).content
                # Clean moderator response
                mod_followup = clean_agent_response(
                    mod_followup, input.moderator_name)
                moderator_last_question = mod_followup  # Store for Agent 2

                # Record moderator intervention
                mod_entry = ConversationEntry(
                    speaker=input.moderator_name,
                    content=mod_followup,
                    turn_number=turn_counter,
                    entry_type="intervention"
                )
                conversation_history.append(mod_entry)

                script.append(f"{input.moderator_name}: {mod_followup}")
                audio_urls.append(generate_audio(
                    mod_followup,
                    "shimmer" if input.voice_provider == "openai" else "rfkTsdZrVWEVhDycUYn9",
                    input.voice_provider,
                    input.moderator_voice,
                    "neutral"
                ))

            # === AGENT 2 TURN ===
            conversation_context = get_conversation_context(
                conversation_history, input.context_length)

            # Build enhanced prompt for Agent 2
            agent2_enhanced_prompt = build_agent_prompt(
                base_prompt=input.agent2_prompt,
                topic=input.topic,
                position=input.agent2_position,
                language=input.language,
                opponent_last_response=agent1_last_response,
                moderator_question=moderator_last_question,
                conversation_context=conversation_context,
                min_words=input.min_words,
                max_words=input.max_words,
                tone=input.agent2_tone
            )

            agent2_response = llm2.invoke(agent2_enhanced_prompt).content
            # Clean response: remove agent name if it appears at the start
            agent2_response = clean_agent_response(
                agent2_response, input.agent2_name)
            agent2_last_response = agent2_response

            # Record Agent 2's response
            agent2_entry = ConversationEntry(
                speaker=input.agent2_name,
                content=agent2_response,
                turn_number=turn_counter,
                entry_type="argument"
            )
            conversation_history.append(agent2_entry)

            script.append(f"{input.agent2_name}: {agent2_response}")
            audio_urls.append(generate_audio(
                agent2_response,
                "ash" if input.voice_provider == "openai" else "uju3wxzG5OhpWcoi3SMy",
                input.voice_provider,
                input.agent2_voice,
                input.agent2_tone
            ))

            # Clear moderator question since it was addressed
            moderator_last_question = ""

            # === OPTIONAL MODERATOR INTERVENTION (after Agent 2) ===
            # Only intervene occasionally after Agent 2 to avoid too much moderation
            if input.moderator_intervention == "frequent" and turn < input.turns_per_agent - 1:  # Don't intervene on last turn
                mod_followup_prompt = build_moderator_prompt(
                    base_prompt=input.moderator_prompt,
                    context=mod_context,
                    conversation_history=conversation_history,
                    target_agent=input.agent1_name,
                    last_speaker=input.agent2_name,
                    last_response=agent2_response
                )

                mod_followup = moderator.invoke(mod_followup_prompt).content
                # Clean moderator response
                mod_followup = clean_agent_response(
                    mod_followup, input.moderator_name)
                moderator_last_question = mod_followup  # Store for Agent 1's next turn

                # Record moderator intervention
                mod_entry = ConversationEntry(
                    speaker=input.moderator_name,
                    content=mod_followup,
                    turn_number=turn_counter,
                    entry_type="intervention"
                )
                conversation_history.append(mod_entry)

                script.append(f"{input.moderator_name}: {mod_followup}")
                audio_urls.append(generate_audio(
                    mod_followup,
                    "fable" if input.voice_provider == "openai" else "rfkTsdZrVWEVhDycUYn9",
                    input.voice_provider,
                    input.moderator_voice,
                    "neutral"
                ))

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"LLM or audio generation failed: {str(e)}")

    # === CLOSING MODERATOR STATEMENT ===
    try:
        closing_context = mod_context.copy()
        closing_context['conversation_summary'] = get_conversation_context(
            conversation_history, len(conversation_history))

        closing_prompt = f"""
{input.moderator_prompt}

The debate between {input.agent1_name} and {input.agent2_name} on '{input.topic}' has concluded.

Here's what was discussed:
{closing_context['conversation_summary'][:500]}...

As {input.moderator_name}, provide a brief closing statement in {input.language} that:
1. Thanks both participants
2. Briefly summarizes the key points raised
3. Concludes the debate professionally

Keep it under 150 words.
"""

        closing_statement = moderator.invoke(closing_prompt).content
        # Clean closing statement
        closing_statement = clean_agent_response(
            closing_statement, input.moderator_name)

        script.append(f"{input.moderator_name}: {closing_statement}")
        audio_urls.append(generate_audio(
            closing_statement,
            "fable" if input.voice_provider == "openai" else "rfkTsdZrVWEVhDycUYn9",
            input.voice_provider,
            input.moderator_voice,
            "neutral"
        ))

    except Exception as e:
        print(f"Warning: Could not generate closing statement: {e}")

    return {
        "script": script,
        "audio_urls": audio_urls,
        "conversation_metadata": {
            "total_turns": len(conversation_history),
            "agent1_turns": len([e for e in conversation_history if e.speaker == input.agent1_name]),
            "agent2_turns": len([e for e in conversation_history if e.speaker == input.agent2_name]),
            "moderator_interventions": len([e for e in conversation_history if e.speaker == input.moderator_name and e.entry_type == "intervention"]),
            "context_length_used": input.context_length
        }
    }
