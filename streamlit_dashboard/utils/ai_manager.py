"""
AI Manager - Handles all AI/LLM interactions for the dashboard.
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from config import FALLBACK_RESPONSE, MODELS_TO_TRY, SYSTEM_PROMPT

load_dotenv()


# --- Helper Functions ---
def get_api_key() -> str | None:
    """Retrieve the Gemini API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    except FileNotFoundError:
        return os.getenv("GEMINI_API_KEY")


def get_client() -> genai.Client | None:
    """Create and return a GenAI client, or None if no API key is available."""
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _extract_json(text: str) -> dict | None:
    """Extract and parse JSON from a text response."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _build_prompt(summary: str) -> str:
    """Build the full prompt for AI content generation."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Data Summary:\n{summary}\n\n"
        "Tasks:\n"
        "1. For 'insight': One savage, brutal roast of the group (use your toxic coach persona)\n"
        "2. For 'facts': 3 genuine, data-driven insights about trends, comparisons, or patterns (be analytical, not snarky)"
    )


# --- Public Functions ---
def generate_ai_content(summary: str) -> dict:
    """
    Generate AI content based on the data summary.

    Tries models in order, falling back if one fails.

    Args:
        summary: Text summary of the activity data.

    Returns:
        Dictionary with 'facts', 'insight', and 'model' keys.
    """
    client = get_client()
    if not client:
        return {}

    prompt = _build_prompt(summary)

    for model_name in MODELS_TO_TRY:
        try:
            print(f"Attempting with model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = response.text
            print(f"Response from {model_name}: {text[:100]}...")

            parsed = _extract_json(text)
            if parsed:
                parsed["model"] = model_name
                return parsed

            # Fallback: treat raw text as insight if JSON parsing failed
            if text:
                print(
                    f"JSON parsing failed for {model_name}, falling back to raw text."
                )
                return {
                    "facts": ["AI was too creative to list facts."],
                    "insight": text,
                    "model": model_name,
                }

        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            continue

    return FALLBACK_RESPONSE
