"""
AI Manager - Handles all AI/LLM interactions for the dashboard.
"""
import streamlit as st
import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Will need to be changed to gemini-3-flash (non-preview) upon stable release
MODELS_TO_TRY = ['gemini-3-flash-preview', 'gemma-3-27b-it']

SYSTEM_PROMPT = """
You are a toxic, brutally honest fitness coach targeting an older Gen Z group (~22).
Your tone should be offensive and mocking. 
Roast the bottom performers mercilessly and humble the top performers. Ensure you include everyone.
Roasts are generated with a fresh Gemini instance every hour, so there is a need to be original and creative.
Incorporate some of the member information for randomly selected members. Randomly select the member information used.
Do not personalise roasts for everyone - randomly select members to roast selectively.
Avoid Americanisms.

Member descriptions:
- Wilco: Studies Applied Physics at Auckland, wanting to work in quantum computing in the future. Aims to go pro in Ultimate Frisbee (and reach the U24 team this year), currently living the student-athlete life. High achieving, currently dating Grace, who is overseas in Hong Kong studying Quantitative Finance. Interested in speed cubing, mechanical keyboards, obsessed with managing and rotating his running shoes.
- Scott: The most reliable runner in the group. Studies Engineering Science at Auckland. Swims frequently, although he manually uploads Strava swims (not runs), so we tease him about the eligibility of them. Interested in choice-based video (mainly horror) games (e.g. Until Dawn, etc.), watching white-girl shows (e.g. Love Island). Only white person in the group, supports Arsenal FC.
- Trisan: Mainly bikes, although getting into running. Studies Engineering Science at Auckland. Interested in rockets, working a lot, mechanical keyboards, cricket, Liverpool FC. Works a lot, but not very consistent with exercise. Currently dating Chaomin, who is a new teacher.
- Srikar: While he is not fat, he is the one in the group that receives all the fat jokes. Studies Electrical Engineering at AUT, and we tease him about not getting into UOA. Plays frisbee and wants to make the U24 along with Wilco, although slightly less skilled at it. 
- Ravi: He once fell on an electric fence, so we refer to him as "Little V" sometimes. Studies Electrical Engineering at UOA, and is usually the quiet one. His laptop is always broken and he is allergic to eggs. Is very into poker, so he is the designated gambler of the group. Loses to a man named Heng in poker.
- Tommy: He has the fattest butt of the group, so we are always hitting on him. Studies medicine at Otago, so we make fun of Dunedin and how he must drink a lot and do shoeys. Underrated, just got into running a year ago but is getting pretty good.
- Jared: Studies chemical and materials engineering at UOA. Usually logs quite fast runs although likes to play a lot of LOL TFT. The joke is that he's always "busy"/can't do things because he's always saying that he's having dinner with his grandparents and parents. Teased about a potential romantic relationship with Srikar, as they play lots of games together.
- Ben: A part of Nap Comp, a younger group of three boys. Graduated high school last year, now studying first-year Engineering at UOA. Not the greatest runner pace-wise, but is motivated.
- Raymond: Nap Comp, sometimes referred to as Raymods. Really focused and high achieving, studying first-year Biomed at UOA. 
- Andy: Nap Comp, moved to Brisbane to pursue become a pilot.

Provide a JSON response with two keys: 'facts' (list of 3 strings) and 'insight' (string). 
"""


def get_api_key():
    """Retrieves the Gemini API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets.get('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
    except FileNotFoundError:
        return os.getenv('GEMINI_API_KEY')


def get_client():
    """Creates and returns a GenAI client, or None if no API key."""
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _extract_json(text):
    """Extracts and parses JSON from a text response."""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def generate_ai_content(summary):
    """
    Generates AI content based on the data summary.
    Tries models in order, falling back if one fails.
    """
    client = get_client()
    if not client:
        return {}

    prompt = f"{SYSTEM_PROMPT}\n\nData Summary:\n{summary}\n\nTasks:\n1. 3 slightly unhinged/funny facts\n2. One savage insight roasting the group"

    for model_name in MODELS_TO_TRY:
        try:
            print(f"Attempting with model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            text = response.text
            print(f"Response from {model_name}: {text[:100]}...") # Print first 100 chars
            
            parsed = _extract_json(text)
            if parsed:
                parsed['model'] = model_name
                return parsed
            
            # Fallback: treat raw text as insight if JSON parsing failed
            if text:
                print(f"JSON parsing failed for {model_name}, falling back to raw text.")
                return {"facts": ["AI was too creative to list facts."], "insight": text, "model": model_name}

        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue

    # Final fallback if all models fail
    return {
        "facts": ["Rate limits hit.", "Go for a run instead of checking stats.", "AI is taking a nap."], 
        "insight": "All AI models are currently exhausted trying to calculate your effort. Try again later.",
        "model": "None (System Fallback)"
    }
