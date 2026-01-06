

import streamlit as st
import pandas as pd
import altair as alt
import datetime

# --- CONFIG ---
# Replace with your actual "Publish to Web" CSV link
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRePCvC9b_RY80n7ulOgVQQwKEWi5GZm8gDeyl7UTaTBONtAOqOsNgGGRm5R9vQtoospZ7RaPbIupBp/pub?gid=0&single=true&output=csv"

st.set_page_config(
    page_title="Sleep Comp Fitness Challenge",
    page_icon="🏃",
    layout="wide"
)

# --- DATA LOADING ---
@st.cache_data(ttl=10)  # Refresh every 30 minutes
def load_data():
    try:
        # Read and drop empty rows
        df = pd.read_csv(SHEET_URL)
        df = df.dropna(subset=['Team', 'Date', 'Name'])
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])

        # Convert columns to strings
        df['Team'] = df['Team'].astype(str)
        df['Type'] = df['Type'].astype(str)
        df['Name'] = df['Name'].astype(str)

        # Handle numeric columns
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

if st.sidebar.button("Update Data", use_container_width=True):
    load_data.clear()
    st.rerun()

st.sidebar.divider()

# Date range filter
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Team filter
all_teams = sorted(df['Team'].unique().tolist())
selected_teams = st.sidebar.pills("Select Teams", all_teams, default=all_teams, selection_mode="multi")

# Activity type filter
all_types = sorted(df['Type'].unique().tolist())
selected_types = st.sidebar.pills("Activity Type", all_types, default=all_types, selection_mode="multi")

# Apply filters
mask = (
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Team'].isin(selected_teams)) &
    (df['Type'].isin(selected_types))
)
filtered_df = df[mask]


# --- MAIN DASHBOARD ---
st.title("Sleep Comp Fitness Challenge")
st.markdown(f"*Tracking activities from **{date_range[0]}** to **{date_range[1]}***")

# Top level metrics
total_km = filtered_df['Distance (km)'].sum()
total_effort = filtered_df['Effort'].sum()
total_runs = len(filtered_df)
total_elevation = filtered_df['Elevation (m)'].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Distance", f"{total_km:,.1f} km")
c2.metric("Total Effort", f"{total_effort:,.1f}")
c3.metric("Total Activities", f"{total_runs}")
c4.metric("Total Elevation", f"{total_elevation:,.1f} m")

st.divider()

# --- LEADERBOARDS ---
col_team, col_indiv = st.columns([1, 2])

with col_team:
    st.subheader("Team Standings")
    team_stats = filtered_df.groupby('Team')[['Effort', 'Distance (km)']].sum().sort_values('Effort', ascending=False).reset_index()
    
    st.dataframe(team_stats, use_container_width=True, hide_index=True)

with col_indiv:
    st.subheader("Individual Leaderboard")
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
st.subheader("Effort Over Time")

# Create cumulative sum for Effort
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
st.subheader("Recent Activities")
recent_df = filtered_df.sort_values('Date', ascending=False).head(15)

# Select and rename columns for display
display_cols = ['Date', 'Name', 'Team', 'Type', 'Distance (km)', 'Effort', 'Pace (min/km)']
st.dataframe(
    recent_df[display_cols].style.format({
        "Date": lambda t: t.strftime("%Y-%m-%d"),
        "Distance (km)": "{:.2f}",
        "Effort": "{:.2f}",
        "Pace (min/km)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)