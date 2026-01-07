"""
AI Manager - Handles all AI/LLM interactions for the dashboard.
"""
import streamlit as st
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODELS_TO_TRY = ['gemini-3-flash', 'gemma-3-27b-it']

SYSTEM_PROMPT = """You are a toxic, brutally honest fitness coach targeting Gen Z (18-25 year olds). 
Your tone should be offensive and mocking. 
Roast the bottom performers mercilessly and humble the top performers. Ensure you include everyone.
Provide a JSON response with two keys: 'facts' (list of 3 strings) and 'insight' (string)."""


def get_api_key():
    """Retrieves the Gemini API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets.get('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
    except FileNotFoundError:
        return os.getenv('GEMINI_API_KEY')


def get_model(model_name):
    """Configures and returns a generative model instance, or None if no API key."""
    api_key = get_api_key()
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


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
    if not get_api_key():
        return {}

    prompt = f"{SYSTEM_PROMPT}\n\nData Summary:\n{summary}\n\nTasks:\n1. 3 slightly unhinged/funny facts\n2. One savage insight roasting the group"

    for model_name in MODELS_TO_TRY:
        try:
            model = get_model(model_name)
            if not model:
                continue
            
            response = model.generate_content(prompt)
            text = response.text
            
            parsed = _extract_json(text)
            if parsed:
                return parsed
            
            # Fallback: treat raw text as insight if JSON parsing failed
            if text:
                return {"facts": ["AI was too creative to list facts."], "insight": text}

        except Exception:
            continue

    # Final fallback if all models fail
    return {
        "facts": ["Rate limits hit.", "Go for a run instead of checking stats.", "AI is taking a nap."], 
        "insight": "All AI models are currently exhausted trying to calculate your effort. Try again later."
    }
