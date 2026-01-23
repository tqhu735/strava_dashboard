"""
Data Manager - Handles data loading, transformation, and summarization.
"""

import pandas as pd
import streamlit as st

# --- Configuration ---
SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vRePCvC9b_RY80n7ulOgVQQwKEWi5GZm8gDeyl7UTaTBONtAOqOsNgGGRm5R9vQtoospZ7RaPbIupBp/"
    "pub?gid=0&single=true&output=csv"
)

REQUIRED_COLUMNS = ["Team", "Date", "Name"]
STRING_COLUMNS = ["Team", "Type", "Name"]
NUMERIC_COLUMNS = ["Distance (km)", "Effort", "Time (min)", "Elevation (m)"]


# --- Data Loading ---
@st.cache_data(ttl=60 * 30)
def load_data() -> pd.DataFrame:
    """Load and clean data from the Google Sheet."""
    try:
        df = pd.read_csv(SHEET_URL)
        df = df.dropna(subset=REQUIRED_COLUMNS)

        df["Date"] = pd.to_datetime(df["Date"])

        for col in STRING_COLUMNS:
            df[col] = df[col].astype(str)

        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# --- Data Summarization ---
def get_data_summary(dataframe: pd.DataFrame) -> str:
    """Create a concise text summary of the dataframe for AI consumption."""
    if dataframe.empty:
        return "No activity data available."

    total_km = dataframe["Distance (km)"].sum()
    total_effort = dataframe["Effort"].sum()
    total_elevation = dataframe["Elevation (m)"].sum()
    total_time_min = dataframe["Time (min)"].sum()

    indiv_stats = dataframe.groupby("Name")["Effort"].sum().sort_values(ascending=False)
    top_performers = indiv_stats.head(5).to_dict()
    bottom_performers = indiv_stats.tail(5).to_dict()

    recent_names = (
        dataframe.sort_values("Date", ascending=False).head(5)["Name"].unique().tolist()
    )

    return f"""
Overall Stats:
- Total Distance: {total_km:.1f} km
- Total Effort: {total_effort:.1f}
- Total Elevation: {total_elevation:.0f} m
- Total Time: {total_time_min:.0f} mins

Leaderboard (Top 5): {top_performers}
Leaderboard (Bottom 5): {bottom_performers}
Recently Active: {", ".join(recent_names)}
"""


def get_manual_fun_facts(dataframe: pd.DataFrame) -> list[str]:
    """Return a list of manually calculated fun facts based on the dataframe."""
    if dataframe.empty:
        return ["No data to calculate fun facts."]

    total_km = dataframe["Distance (km)"].sum()
    total_elevation = dataframe["Elevation (m)"].sum()
    total_time_min = dataframe["Time (min)"].sum()

    return [
        f"**{total_km / 42.195:.1f}** marathons worth of distance 🏃",
        f"**{total_km / 1600:.2f}x** the length of New Zealand 🇳🇿",
        f"**{(total_km / 69420) * 100:.4f}%** of the way around your mom 🤰",
        f"**{total_elevation / 328:.1f}** times the Auckland Sky Tower 🗼",
        f"**{total_elevation / 8848:.2f}** Mount Everests climbed 🏔️",
        f"**{total_time_min / 480:.1f}** full 8-hour work days 💼",
        f"**{total_time_min / 22:.1f}** episodes of Friends ☕️",
    ]
