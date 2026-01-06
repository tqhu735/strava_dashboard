import streamlit as st
import pandas as pd
import altair as alt
import datetime
import numpy as np

# --- CONFIG ---
SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRePCvC9b_RY80n7ulOgVQQwKEWi5GZm8gDeyl7UTaTBONtAOqOsNgGGRm5R9vQtoospZ7RaPbIupBp/pub?gid=0&single=true&output=csv"

st.set_page_config(
    page_title="Sleep Comp Fitness Challenge",
    page_icon="🏃",
    layout="wide"
)


# --- DATA LOADING ---
@st.cache_data(ttl=10)  # Refresh every 10 seconds
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

if st.sidebar.button("Update Data", width="stretch"):
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

# Competitor filter
all_names = sorted(df['Name'].unique().tolist())
selected_names = st.sidebar.pills("Competitors", all_names, default=all_names, selection_mode="multi")

# Apply filters
mask = (
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Team'].isin(selected_teams)) &
    (df['Type'].isin(selected_types)) &
    (df['Name'].isin(selected_names))
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
total_time_min = filtered_df['Time (min)'].sum()

# Convert time to hours/minutes for display
hours = int(total_time_min // 60)
minutes = int(total_time_min % 60)
days = int(hours // 24)
hours = int(hours % 24)
time_display = f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Distance", f"{total_km:,.1f} km")
c2.metric("Total Effort", f"{total_effort:,.1f}")
c3.metric("Total Time", time_display)
c4.metric("Total Activities", f"{total_runs}")
c5.metric("Total Elevation", f"{total_elevation:,.1f} m")

st.divider()


# --- LEADERBOARDS ---
col_team, col_indiv = st.columns([1, 2])

with col_team:
    st.subheader("Team Standings")
    team_stats = filtered_df.groupby('Team')[['Effort', 'Distance (km)']].sum().sort_values('Effort', ascending=False).reset_index()
    
    st.dataframe(team_stats, width="stretch", hide_index=True)

    # --- WIN PROBABILITY ---
    st.subheader("Predicted Win Chance")
    
    # Configuration - Hardcoded End Date
    COMPETITION_END_DATE = datetime.date(2026, 12, 31)
    
    # 1. Calculate remaining days
    today = datetime.date.today()
    days_remaining = (COMPETITION_END_DATE - today).days
    
    if days_remaining > 0:
        # 2. Daily stats per team
        # We need to make sure we include 0-effort days in the stats if they are valid days, 
        # but for simplicity based on the prompt "mean and SD of effort per day", 
        # we will aggregate by existing data points. 
        # Ideally we should reindex by all dates, but we'll stick to the user's "mean/sd of effort" intuition on available data.
        
        # Group by Team and Date first to get daily totals
        daily_effort = df.groupby(['Team', 'Date'])['Effort'].sum().reset_index()
        
        # Calculate Mean and SD per team
        team_stats_daily = daily_effort.groupby('Team')['Effort'].agg(['mean', 'std']).fillna(0)
        
        # Current totals
        current_totals = df.groupby('Team')['Effort'].sum()
        
        # 3. Monte Carlo Simulation
        N_SIMULATIONS = 10000
        teams = current_totals.index.tolist()
        sim_results = {team: 0 for team in teams}
        
        # Create a matrix of future efforts: (N_SIMULATIONS, n_teams)
        # We simulate the *total* future effort for the remaining days
        # Total Future ~ N(mean * days, std * sqrt(days))
        
        future_efforts = {}
        for team in teams:
            mu = team_stats_daily.loc[team, 'mean']
            sigma = team_stats_daily.loc[team, 'std']
            
            # Simulated total remaining effort
            # The sum of N independent normal variables N(mu, sigma) is N(N*mu, sqrt(N)*sigma)
            sim_mu = mu * days_remaining
            sim_sigma = sigma * np.sqrt(days_remaining)
            
            future_efforts[team] = np.random.normal(sim_mu, sim_sigma, N_SIMULATIONS)
        
        # Add current totals to get final simulated totals
        final_scores = pd.DataFrame(index=range(N_SIMULATIONS))
        for team in teams:
            final_scores[team] = current_totals[team] + future_efforts[team]
            
        # Determine winner for each simulation
        winners = final_scores.idxmax(axis=1)
        win_counts = winners.value_counts()
        
        # Calculate probabilities
        win_probs = (win_counts / N_SIMULATIONS).reset_index()
        win_probs.columns = ['Team', 'Probability']
        
        # 4. Visualization (Donut Chart)
        prob_chart = alt.Chart(win_probs).mark_arc(innerRadius=60).encode(
            theta=alt.Theta(field="Probability", type="quantitative"),
            color=alt.Color(field="Team", type="nominal"),
            tooltip=['Team', alt.Tooltip('Probability', format='.1%')],
            order=alt.Order("Probability", sort="descending")
        ).properties(height=200)
        
        st.altair_chart(prob_chart, use_container_width=True)
        st.caption(f"Based on daily effort history. Simulating {days_remaining} remaining days.")
        
    else:
        st.info("Competition has ended.")

with col_indiv:
    st.subheader("Individual Standings")
    indiv_stats = filtered_df.groupby(['Name', 'Team'])[['Effort', 'Distance (km)', 'Time (min)']].sum().reset_index()
    indiv_stats = indiv_stats.sort_values('Effort', ascending=False).reset_index(drop=True)
    indiv_stats.index += 1  # Start ranking at 1
    
    st.dataframe(
        indiv_stats.style.format({
            "Effort": "{:.1f}", 
            "Distance (km)": "{:.1f}",
            "Time (min)": "{:.0f}"
        }),
        width="stretch"
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

st.altair_chart(line_chart, width="stretch")


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
    width="stretch",
    hide_index=True
)