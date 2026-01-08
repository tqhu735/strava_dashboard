"""
Sleep Comp Fitness Challenge Dashboard
Main entry point for the Streamlit application.
"""
import streamlit as st
import pandas as pd
import altair as alt
import datetime
import numpy as np
import random

from utils.data_manager import load_data, get_data_summary, get_manual_fun_facts
from utils.ai_manager import generate_ai_content, get_client

# --- Configuration ---
COMPETITION_START_DATE = datetime.date(2026, 1, 1)
COMPETITION_END_DATE = datetime.date(2026, 12, 31)
N_SIMULATIONS = 10000

st.set_page_config(
    page_title="Sleep Comp Fitness Challenge",
    page_icon="🏃",
    layout="wide"
)

# --- Data Loading ---
df = load_data()
if df.empty:
    st.stop()

df['Month'] = df['Date'].dt.strftime('%Y-%m')
TODAY = datetime.date.today()
CURRENT_MONTH = TODAY.strftime('%Y-%m')


# --- Helper Functions ---
def display_metrics(data):
    """Displays key summary metrics in a row of columns."""
    total_km = data['Distance (km)'].sum()
    total_effort = data['Effort'].sum()
    total_activities = len(data)
    total_elevation = data['Elevation (m)'].sum()
    total_time_min = data['Time (min)'].sum()

    hours, minutes = divmod(int(total_time_min), 60)
    days, hours = divmod(hours, 24)
    time_display = f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"

    cols = st.columns(5)
    cols[0].metric("Total Distance", f"{total_km:,.1f} km")
    cols[1].metric("Total Effort", f"{total_effort:,.1f}")
    cols[2].metric("Total Time", time_display)
    cols[3].metric("Total Activities", f"{total_activities}")
    cols[4].metric("Total Elevation", f"{int(total_elevation)} m")


def get_winner_history(source_df, group_cols, value_col='Effort'):
    """Calculates the winner for each month based on the provided grouping."""
    if source_df.empty:
        return pd.DataFrame()

    monthly_sums = source_df.groupby(['Month'] + group_cols)[value_col].sum().reset_index()
    
    results = []
    for month in sorted(monthly_sums['Month'].unique(), reverse=True):
        m_data = monthly_sums[monthly_sums['Month'] == month]
        if not m_data.empty:
            winner = m_data.loc[m_data[value_col].idxmax()]
            entry = {'Month': month, 'Winner': winner[group_cols[0]], 'Effort': winner[value_col]}
            if 'Name' in group_cols:
                entry['Team'] = winner['Team']
            results.append(entry)
    return pd.DataFrame(results)


def run_monte_carlo_simulation(daily_effort_df, current_totals, days_remaining):
    """Runs Monte Carlo simulation to estimate win probabilities."""
    all_dates = pd.date_range(start=COMPETITION_START_DATE, end=TODAY)
    teams = current_totals.index.tolist()
    
    # Calculate daily stats per team
    team_stats = {}
    for team in daily_effort_df['Team'].unique():
        team_data = daily_effort_df[daily_effort_df['Team'] == team].set_index('Date').reindex(all_dates, fill_value=0)
        team_stats[team] = {'mean': team_data['Effort'].mean(), 'std': team_data['Effort'].std()}
    
    # Simulate future efforts
    future_efforts = {}
    for team in teams:
        mu, sigma = team_stats[team]['mean'], team_stats[team]['std']
        sim_mu = mu * days_remaining
        sim_sigma = np.sqrt(days_remaining * (sigma**2) + (days_remaining**2) * ((sigma**2) / len(all_dates)))
        future_efforts[team] = np.random.normal(sim_mu, sim_sigma, N_SIMULATIONS)
    
    final_scores = pd.DataFrame({team: current_totals[team] + future_efforts[team] for team in teams})
    win_probs = (final_scores.idxmax(axis=1).value_counts() / N_SIMULATIONS).reset_index()
    win_probs.columns = ['Team', 'Probability']
    return win_probs


@st.cache_data(ttl=60*60)
def get_ai_content_cached(summary):
    """Cached wrapper for AI content generation (1 hour TTL)."""
    return generate_ai_content(summary)


# --- Sidebar Filters ---
st.sidebar.header("Filters")

if st.sidebar.button("Update Data", use_container_width=True):
    load_data.clear()
    st.rerun()

st.sidebar.divider()

min_date, max_date = df['Date'].min(), df['Date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

all_teams = sorted(df['Team'].unique().tolist())
selected_teams = st.sidebar.pills("Select Teams", all_teams, default=all_teams, selection_mode="multi")

all_types = sorted(df['Type'].unique().tolist())
selected_types = st.sidebar.pills("Activity Type", all_types, default=all_types, selection_mode="multi")

all_names = sorted(df['Name'].unique().tolist())
selected_names = st.sidebar.pills("Competitors", all_names, default=all_names, selection_mode="multi")

# --- Apply Filters ---
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (
        (df['Date'].dt.date >= start_date) &
        (df['Date'].dt.date <= end_date) &
        (df['Team'].isin(selected_teams)) &
        (df['Type'].isin(selected_types)) &
        (df['Name'].isin(selected_names))
    )
    filtered_df = df[mask]
else:
    filtered_df = df

# History uses categorical filters only (ignores date range)
history_mask = (
    df['Team'].isin(selected_teams) &
    df['Type'].isin(selected_types) &
    df['Name'].isin(selected_names)
)
history_df = df[history_mask].copy()

# --- Main Dashboard ---
st.title("Sleep Comp Fitness Challenge")
if len(date_range) == 2:
    st.markdown(f"*Tracking activities from **{date_range[0]}** to **{date_range[1]}***")

display_metrics(filtered_df)

# Reserve space for AI section (rendered last to avoid blocking)
ai_placeholder = st.empty()
st.divider()

# --- Leaderboards ---
col_team, col_indiv = st.columns([1, 2])

with col_team:
    st.subheader("Team Standings")
    tab_month, tab_year, tab_history = st.tabs(["Month", "Year", "Monthly History"])

    with tab_month:
        st.caption(f"Standings for {TODAY.strftime('%B %Y')}")
        month_data = filtered_df[filtered_df['Month'] == CURRENT_MONTH]
        if not month_data.empty:
            stats = month_data.groupby('Team')[['Effort', 'Distance (km)']].sum().sort_values('Effort', ascending=False).reset_index()
            st.dataframe(stats, use_container_width=True, hide_index=True)
        else:
            st.info("No activities for the current month.")

    with tab_year:
        stats = filtered_df.groupby('Team')[['Effort', 'Distance (km)']].sum().sort_values('Effort', ascending=False).reset_index()
        st.dataframe(stats, use_container_width=True, hide_index=True)

    with tab_history:
        st.caption("Winners of each month")
        history_table = get_winner_history(history_df, ['Team'])
        if not history_table.empty:
            st.dataframe(history_table, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for history.")

    # Win Probability (Monte Carlo)
    st.subheader("Win Probability")
    days_remaining = (COMPETITION_END_DATE - TODAY).days

    if days_remaining > 0:
        daily_effort = df.groupby(['Team', 'Date'])['Effort'].sum().reset_index()
        current_totals = df.groupby('Team')['Effort'].sum()
        win_probs = run_monte_carlo_simulation(daily_effort, current_totals, days_remaining)
        
        prob_chart = alt.Chart(win_probs).mark_arc(innerRadius=60).encode(
            theta=alt.Theta(field="Probability", type="quantitative"),
            color=alt.Color(field="Team", type="nominal"),
            tooltip=['Team', alt.Tooltip('Probability', format='.1%')],
            order=alt.Order("Probability", sort="descending")
        ).properties(height=200)
        
        st.altair_chart(prob_chart, use_container_width=True)
        st.caption(f"10k simulations over {days_remaining} remaining days.")
    else:
        st.info("Competition has ended.")

with col_indiv:
    st.subheader("Individual Standings")
    indiv_tab_month, indiv_tab_year, indiv_tab_history = st.tabs(["Month", "Year", "Monthly History"])
    
    with indiv_tab_month:
        st.caption(f"Standings for {TODAY.strftime('%B %Y')}")
        month_data_indiv = filtered_df[filtered_df['Month'] == CURRENT_MONTH]
        if not month_data_indiv.empty:
            stats = month_data_indiv.groupby(['Name', 'Team'])[['Effort', 'Distance (km)', 'Time (min)']].sum().reset_index()
            stats = stats.sort_values('Effort', ascending=False).reset_index(drop=True)
            stats.index += 1
            st.dataframe(stats.style.format({"Effort": "{:.1f}", "Distance (km)": "{:.1f}", "Time (min)": "{:.0f}"}), use_container_width=True)
        else:
            st.info("No activities for the current month.")

    with indiv_tab_year:
        stats = filtered_df.groupby(['Name', 'Team'])[['Effort', 'Distance (km)', 'Time (min)']].sum().reset_index()
        stats = stats.sort_values('Effort', ascending=False).reset_index(drop=True)
        stats.index += 1
        st.dataframe(stats.style.format({"Effort": "{:.1f}", "Distance (km)": "{:.1f}", "Time (min)": "{:.0f}"}), use_container_width=True)
    
    with indiv_tab_history:
        st.caption("Winners of each month")
        history_table_indiv = get_winner_history(history_df, ['Name', 'Team'])
        if not history_table_indiv.empty:
            st.dataframe(history_table_indiv, use_container_width=True, hide_index=True)
        else:
            st.info("No data available for history.")

st.divider()

# --- Activity Heatmap ---
st.subheader("Activity Heatmap")
daily_summary = filtered_df.groupby('Date').agg({
    'Effort': 'sum',
    'Distance (km)': 'sum',
    'Name': lambda x: ', '.join(sorted(x.unique())),
    'Type': lambda x: ', '.join(sorted(x.unique()))
}).reset_index()
daily_summary['Week'] = daily_summary['Date'].dt.isocalendar().week
daily_summary['Day'] = daily_summary['Date'].dt.day_name()

heatmap = alt.Chart(daily_summary).mark_rect().encode(
    x=alt.X('Week:O', title='Week of Year'),
    y=alt.Y('Day:N', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], title='Day of Week'),
    color=alt.Color('Effort:Q', title='Daily Effort', scale=alt.Scale(scheme='greens')),
    tooltip=[
        alt.Tooltip('Date:T', format='%Y-%m-%d'),
        alt.Tooltip('Effort:Q', format='.1f', title='Total Effort'),
        alt.Tooltip('Distance (km):Q', format='.1f', title='Total Km'),
        alt.Tooltip('Name:N', title='People'),
        alt.Tooltip('Type:N', title='Activities')
    ]
).properties(height=250).configure_axis(labelFontSize=10, titleFontSize=12)

st.altair_chart(heatmap, use_container_width=True)

# --- Effort Trends ---
st.subheader("Effort Over Time")
chart_df = filtered_df.sort_values('Date').copy()
chart_df['Cumulative Effort'] = chart_df.groupby('Name')['Effort'].cumsum()

line_chart = alt.Chart(chart_df).mark_line(point=True).encode(
    x='Date:T',
    y='Cumulative Effort:Q',
    color='Name:N',
    tooltip=['Date', 'Name', 'Type', 'Distance (km)', 'Effort']
)

st.altair_chart(line_chart, use_container_width=True)

# --- Activity Feed ---
st.subheader("Activities")
display_cols = ['Date', 'Name', 'Team', 'Type', 'Distance (km)', 'Effort', 'Pace (min/km)']
st.dataframe(
    filtered_df.sort_values('Date', ascending=False)[display_cols].style.format({
        "Date": lambda t: t.strftime("%Y-%m-%d"),
        "Distance (km)": "{:.2f}",
        "Effort": "{:.2f}",
        "Pace (min/km)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

# --- AI Section (Rendered Last) ---
with ai_placeholder.container():
    # Initialize AI data on first load
    if 'ai_data' not in st.session_state:
        if get_client():
            data_summary = get_data_summary(filtered_df)
            st.session_state['ai_data'] = get_ai_content_cached(data_summary)
        else:
            st.session_state['ai_data'] = {}

    ai_data = st.session_state.get('ai_data', {})
    manual_facts = get_manual_fun_facts(filtered_df)

    if get_client():
        # Header with inline refresh link
        col_title, col_btn = st.columns([10, 1])
        with col_title:
            st.subheader("Summary")
        with col_btn:
            if st.button("🔄", help="Refresh insights for current filters"):
                with st.spinner("Refreshing..."):
                    data_summary = get_data_summary(filtered_df)
                    st.session_state['ai_data'] = get_ai_content_cached(data_summary)
                st.rerun()
        
        st.info(ai_data.get('insight', "The coach is currently judging you in silence."))
        if 'model' in ai_data:
            st.caption(f"Generated with: {ai_data['model']}")
        
        st.subheader("Fun Facts")
        ai_facts = ai_data.get('facts', [])
        if ai_facts and not any("Error" in f or "AI is being shy" in f for f in ai_facts):
            for fact in ai_facts:
                st.write(f"• {fact}")
        elif not ai_facts:
            for fact in random.sample(manual_facts, min(3, len(manual_facts))):
                st.write(f"• {fact}")
    else:
        st.subheader("Fun Facts")
        st.info("\n\n".join([f"• {fact}" for fact in random.sample(manual_facts, min(3, len(manual_facts)))]))
