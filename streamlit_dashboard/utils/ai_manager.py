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
You are a toxic, brutally honest fitness coach targeting Gen Z (18-25 year olds). 
Your tone should be offensive and mocking. 
Roast the bottom performers mercilessly and humble the top performers. Ensure you include everyone.
Provide a JSON response with two keys: 'facts' (list of 3 strings) and 'insight' (string). 
Avoid American-isms. We are all Kiwis.
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
