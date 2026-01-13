# Sleep Comp Strava Dashboard

This repository contains the code for tracking the Sleep Comp fitness competition. It consists of two main parts: a Google Apps Script backend for data fetching and a Streamlit dashboard for visualization.

## Architecture

1. Google Apps Script (apps_script/)
   - Fetches activity data from the Strava Club API.
   - Stores data in a Google Sheet.
   - Sends notifications to Discord webhooks.
   - Runs on a time-based trigger or via a web app endpoint.

2. Streamlit Dashboard (streamlit_dashboard/)
   - Reads the data from the Google Sheet (published as CSV).
   - Displays leaderboards, team standings, and progress charts.
   - Calculates win probabilities using Monte Carlo simulations.
   - Generates commentary using the Google Gemini API.

## Setup

### Google Apps Script

1. Create a new Google Apps Script project attached to a Google Sheet.
2. Copy the files from 'apps_script/' into the project.
3. Add the OAuth2 library for Google Apps Script (ID: 1B7FSrk5Zi6L1rSxxTDgDEUsPzlukDsi4KGuTMirs4KDa85TR22HI3).
4. Set the following Script Properties (Project Settings > Script Properties):
   - CLIENT_ID: Your Strava Application Client ID.
   - CLIENT_SECRET: Your Strava Application Client Secret.
   - CLUB_ID: The Strava Club ID to track.
   - DISCORD_WEBHOOK_URL: The webhook URL for the Discord channel.
5. Deploy as a Web App to enable the update trigger endpoint.
6. Set up a time-driven trigger to run 'fetchClubActivities' hourly.

### Streamlit Dashboard

1. Navigate to the 'streamlit_dashboard' directory.
2. Install dependencies:
   pip install -r requirements.txt

3. Set up environment variables or a '.streamlit/secrets.toml' file with your API keys:
   GEMINI_API_KEY = "your_google_ai_studio_key"

4. Run the dashboard locally:
   streamlit run dashboard.py

## Configuration

### Multipliers

Effort points are calculated based on the activity type using these multipliers:
- Swim: 3.5x
- Run: 1.0x
- Rowing: 0.75x
- Hike: 0.5x
- Ride: 0.35x
- Walk: 0.1x

### Teams

The current teams are configured in 'apps_script/config.js':
- Team Srikar: Srikar, Wilco, Trisan, Jared, Raymond
- Team Ravi: Ravi, Andy, Scott, Ben, Tommy

## Deployment

The dashboard is designed to be deployed on Streamlit Community Cloud. Ensure the repository is connected and the secrets are configured in the platform's settings.