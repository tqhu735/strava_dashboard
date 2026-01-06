

import streamlit as st
import pandas as pd
import altair as alt
import datetime

# --- CONFIG ---
# Replace with your actual "Publish to Web" CSV link
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRePCvC9b_RY80n7ulOgVQQwKEWi5GZm8gDeyl7UTaTBONtAOqOsNgGGRm5R9vQtoospZ7RaPbIupBp/pub?gid=0&single=true&output=csv"

st.set_page_config(
    page_title="Sleep Comp Strava Tracker",
    page_icon="🏃",
    layout="wide"
)

# --- DATA LOADING ---
@st.cache_data(ttl=600)  # Refresh every 10 mins
def load_data():
    try:
        df = pd.read_csv(SHEET_URL)
        
        # Ensure column names match your sheet exactly
        # Based on your screenshot: ID, Name, Team, Date, Distance (km), Effort, Time (min), Pace (min/km), Elevation (m), Type
        
        # Parse Dates
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle Numeric Columns (coerce errors to 0 just in case)
        cols_to_numeric = ['Distance (km)', 'Effort', 'Time (min)', 'Elevation (m)']
        for col in cols_to_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

# Year/Month Filter (Optional, defaults to all time)
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Team Filter
all_teams = sorted(df['Team'].unique().tolist())
selected_teams = st.sidebar.multiselect("Select Teams", all_teams, default=all_teams)

# Activity Type Filter
all_types = sorted(df['Type'].unique().tolist())
selected_types = st.sidebar.multiselect("Activity Type", all_types, default=all_types)

# Apply Filters
mask = (
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Team'].isin(selected_teams)) &
    (df['Type'].isin(selected_types))
)
filtered_df = df[mask]

# --- MAIN DASHBOARD ---
st.title("🏃 Sleep Comp 100k Challenge")
st.markdown(f"*Tracking activities from **{date_range[0]}** to **{date_range[1]}***")

# Top Level Metrics
total_km = filtered_df['Distance (km)'].sum()
total_effort = filtered_df['Effort'].sum()
total_runs = len(filtered_df)

c1, c2, c3 = st.columns(3)
c1.metric("Total Distance", f"{total_km:,.1f} km")
c2.metric("Total Effort Points", f"{total_effort:,.1f}")
c3.metric("Total Activities", f"{total_runs}")

st.divider()

# --- LEADERBOARDS ---
col_team, col_indiv = st.columns([1, 2])

with col_team:
    st.subheader("🏆 Team Standings")
    team_stats = filtered_df.groupby('Team')[['Effort', 'Distance (km)']].sum().sort_values('Effort', ascending=False).reset_index()
    
    # Display as a clean dataframe or styled table
    st.dataframe(
        team_stats.style.format({"Effort": "{:.1f}", "Distance (km)": "{:.1f}"})
        .background_gradient(cmap="Oranges", subset=["Effort"]),
        use_container_width=True,
        hide_index=True
    )

with col_indiv:
    st.subheader("👤 Individual Leaderboard")
    indiv_stats = filtered_df.groupby(['Name', 'Team'])[['Effort', 'Distance (km)', 'Time (min)']].sum().reset_index()
    indiv_stats = indiv_stats.sort_values('Effort', ascending=False).reset_index(drop=True)
    indiv_stats.index += 1  # Start ranking at 1
    
    st.dataframe(
        indiv_stats.style.format({
            "Effort": "{:.1f}", 
            "Distance (km)": "{:.1f}",
            "Time (min)": "{:.0f}"
        }),
        use_container_width=True
    )

st.divider()

# --- CHARTS ---
st.subheader("📈 Progress Over Time (Effort Points)")

# Create Cumulative Sum for the Chart
# Sort by date first
chart_df = filtered_df.sort_values('Date')
chart_df['Cumulative Effort'] = chart_df.groupby('Name')['Effort'].cumsum()

# Altair Line Chart
line_chart = alt.Chart(chart_df).mark_line(point=True).encode(
    x='Date:T',
    y='Cumulative Effort:Q',
    color='Name:N',
    tooltip=['Date', 'Name', 'Type', 'Distance (km)', 'Effort']
).interactive()

st.altair_chart(line_chart, use_container_width=True)

# --- RECENT ACTIVITY FEED ---
st.subheader("📅 Recent Activities")
recent_df = filtered_df.sort_values('Date', ascending=False).head(15)

# Select and rename columns for display
display_cols = ['Date', 'Name', 'Team', 'Type', 'Distance (km)', 'Effort', 'Pace (min/km)']
st.dataframe(
    recent_df[display_cols].style.format({
        "Date": lambda t: t.strftime("%Y-%m-%d"),
        "Distance (km)": "{:.2f}",
        "Effort": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)