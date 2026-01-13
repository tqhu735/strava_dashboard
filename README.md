# Sleep Comp Strava Dashboard

This repository contains the code for the 2026 Sleep Comp fitness competition. It consists of two main parts: a Google Apps Script backend for data fetching and a Streamlit dashboard for visualisation.

## Architecture

1. Google Apps Script (apps_script/)
   - Fetches activity data from the Strava Club API.
   - Stores data in a Google Sheet.
   - Sends notifications to fitness-comp via a Discord webhook.
   - Runs on a time-based trigger (hourly, as defined in Apps Script).

2. Streamlit Dashboard (streamlit_dashboard/)
   - Reads the data from the Google Sheet (published as CSV).
   - Displays leaderboards, team standings, and progress charts.
   - Calculates win probabilities using Monte Carlo simulations.
   - Generates commentary using the Google Gemini API.

## Setup

### Google Apps Script

1. Create a new Google Apps Script project attached to a Google Sheet.
2. Copy the files from `apps_script/` into the project.
3. Add the OAuth2 library for Google Apps Script.
4. Set the following Script Properties (Project Settings > Script Properties):
   - CLIENT_ID: Strava Client ID.
   - CLIENT_SECRET: Strava Client Secret.
   - CLUB_ID: Strava Club ID.
   - DISCORD_WEBHOOK_URL: Discord webhook URL.
5. Set up a time-driven trigger to run 'fetchClubActivities' hourly.

### Streamlit Dashboard

1. Navigate to the `streamlit_dashboard` directory.
2. Install dependencies:
   `pip install -r requirements.txt`

3. Set up environment variables or a '.streamlit/secrets.toml' file with your API keys:
   `GEMINI_API_KEY = "your_google_ai_studio_key"`

4. Run the dashboard locally:
   `streamlit run dashboard.py`

## Configuration

### Multipliers

Effort points are calculated based on the activity type using these multipliers:
- Swim: 3.5x
- Run: 1.0x
- Rowing: 0.75x
- Hike: 0.5x
- Ride: 0.35x
- Walk: 0.1x
- Workout: 0.3x

### Teams

The current teams are configured in `apps_script/config.js`:
- Team Srikar: Srikar, Wilco, Trisan, Jared, Raymond, Minnie, Grace
- Team Ravi: Ravi, Andy, Scott, Ben, Tommy, Kevin, Jinchien, Chris

## Deployment

Dashboard is currently deployed on Streamlit Community Cloud. Ensure the repository is connected and the secrets are configured in the site settings.