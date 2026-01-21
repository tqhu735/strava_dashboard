"""
Sleep Comp Fitness Challenge Dashboard
Main entry point for the Streamlit application.
"""

import datetime
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.ai_manager import generate_ai_content, get_client
from utils.data_manager import get_data_summary, get_manual_fun_facts, load_data

# --- Configuration ---
COMPETITION_START_DATE = datetime.date(2026, 1, 1)
COMPETITION_END_DATE = datetime.date(2026, 12, 31)
N_SIMULATIONS = 10000
DAYS_OF_WEEK = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
GROUP_DISTANCE_GOAL = 10000

st.set_page_config(
    page_title="Sleep Comp Fitness Challenge", page_icon="🏃", layout="wide"
)


# --- Helper Functions ---
def display_metrics(data: pd.DataFrame) -> None:
    """Display key summary metrics in a row of columns."""
    total_km = data["Distance (km)"].sum()
    total_effort = data["Effort"].sum()
    total_elevation = data["Elevation (m)"].sum()
    total_time_min = data["Time (min)"].sum()

    hours, minutes = divmod(int(total_time_min), 60)
    days, hours = divmod(hours, 24)
    time_str = f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"

    cols = st.columns(5)
    cols[0].metric("Total Distance", f"{total_km:,.1f} km")
    cols[1].metric("Total Effort", f"{total_effort:,.1f}")
    cols[2].metric("Total Time", time_str)
    cols[3].metric("Total Activities", f"{len(data)}")
    cols[4].metric("Total Elevation", f"{int(total_elevation)} m")


def render_goal_progress(data: pd.DataFrame) -> None:
    """Render a progress bar for the group distance goal with predictions."""
    total_distance = data["Distance (km)"].sum()
    progress = min(1.0, total_distance / GROUP_DISTANCE_GOAL)

    # Calculate prediction based on daily average
    days_elapsed = (TODAY - COMPETITION_START_DATE).days + 1
    if days_elapsed > 0:
        predicted_total = (total_distance / days_elapsed) * 365
    else:
        predicted_total = 0

    st.subheader("Group Goal Progress")

    # Custom progress bar with metric labels
    st.progress(progress)

    c1, c2 = st.columns(2)
    c1.metric("Goal", f"{total_distance:,.1f}/{GROUP_DISTANCE_GOAL:,.0f} km")
    c2.metric(
        "Projected Total",
        f"{predicted_total:,.1f} km",
        delta=f"{predicted_total - GROUP_DISTANCE_GOAL:,.1f} km",
        help="Prediction based on current daily average extrapolated to 365 days.",
    )

    # TODO: Progress vs linear path to goal


def get_winner_history(
    source_df: pd.DataFrame, group_cols: list, value_col: str = "Effort"
) -> pd.DataFrame:
    """Calculate the winner for each month based on the provided grouping."""
    if source_df.empty:
        return pd.DataFrame()

    monthly_sums = (
        source_df.groupby(["Month"] + group_cols)[value_col].sum().reset_index()
    )
    results = []
    for month in sorted(monthly_sums["Month"].unique(), reverse=True):
        month_data = monthly_sums[monthly_sums["Month"] == month]
        if month_data.empty:
            continue
        winner = month_data.loc[month_data[value_col].idxmax()]
        entry = {
            "Month": month,
            "Winner": winner[group_cols[0]],
            "Effort": winner[value_col],
        }
        if "Name" in group_cols:
            entry["Team"] = winner["Team"]
        results.append(entry)
    return pd.DataFrame(results)


def run_monte_carlo_simulation(
    daily_effort_df: pd.DataFrame, current_totals: pd.Series, days_remaining: int
) -> pd.DataFrame:
    """Run Monte Carlo simulation to estimate win probabilities."""
    all_dates = pd.date_range(start=COMPETITION_START_DATE, end=TODAY)
    teams = current_totals.index.tolist()
    n_days = len(all_dates)

    # Calculate daily stats per team
    team_stats = {}
    for team in daily_effort_df["Team"].unique():
        team_data = (
            daily_effort_df[daily_effort_df["Team"] == team]
            .set_index("Date")
            .reindex(all_dates, fill_value=0)
        )
        team_stats[team] = {
            "mean": team_data["Effort"].mean(),
            "std": team_data["Effort"].std(),
        }

    # Simulate future efforts
    future_efforts = {}
    for team in teams:
        mu, sigma = team_stats[team]["mean"], team_stats[team]["std"]
        sim_mu = mu * days_remaining
        sim_sigma = np.sqrt(
            days_remaining * (sigma**2) + (days_remaining**2) * ((sigma**2) / n_days)
        )
        future_efforts[team] = np.random.normal(sim_mu, sim_sigma, N_SIMULATIONS)

    final_scores = pd.DataFrame(
        {team: current_totals[team] + future_efforts[team] for team in teams}
    )
    win_probs = (
        final_scores.idxmax(axis=1).value_counts() / N_SIMULATIONS
    ).reset_index()
    win_probs.columns = ["Team", "Probability"]
    return win_probs


@st.cache_data(ttl=3600)
def get_ai_content_cached(summary: str) -> dict:
    """Cached wrapper for AI content generation (1 hour TTL)."""
    return generate_ai_content(summary)


def render_standings_table(
    data: pd.DataFrame, columns: list, format_dict: dict = None
) -> None:
    """Render a standings dataframe with optional formatting."""
    if format_dict:
        st.dataframe(data.style.format(format_dict), width="stretch")
    else:
        st.dataframe(data, width="stretch", hide_index=True)


def render_h2h_comparison(data: pd.DataFrame) -> None:
    """Render a head-to-head comparison between two competitors."""
    st.subheader("Head-to-Head Comparison")

    names = sorted(data["Name"].unique())
    if len(names) < 2:
        st.info("Select at least two competitors to see a comparison.")
        return

    # User Selection
    c1, c2 = st.columns(2)
    with c1:
        player1 = st.selectbox("Compare competitor...", names, index=0, key="h2h_p1")
    with c2:
        player2 = st.selectbox(
            "...with competitor", names, index=1 if len(names) > 1 else 0, key="h2h_p2"
        )

    if player1 == player2:
        st.warning("Please select two different people to compare.")
        return

    p1_data = data[data["Name"] == player1]
    p2_data = data[data["Name"] == player2]

    # Helper for time formatting
    def format_time(mins):
        h, m = divmod(int(mins), 60)
        return f"{h}h {m}m"

    stats_to_compare = [
        ("Effort", "Effort", "{:,.1f}"),
        ("Distance", "Distance (km)", "{:,.1f} km"),
        ("Activities", "Name", "{}"),
        ("Elevation", "Elevation (m)", "{:,.0f} m"),
        ("Total Time", "Time (min)", "time"),
    ]

    # Visual Layout
    st.write("")

    # Avatar and Name Headers
    h1, h_vs, h2 = st.columns([2, 1, 2])

    # with h1:
    #     # Placeholder for profile picture - keeping space for future JPGs
    #     st.markdown(
    #         f"""
    #         <div style="text-align: center;">
    #             <div style="font-size: 60px; margin-bottom: 10px; background-color: #f0f2f6; border-radius: 50%; width: 100px; height: 100px; line-height: 100px; margin-left: auto; margin-right: auto;">👤</div>
    #             <h2 style="margin-top: 10px;">{player1}</h2>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    # with h_vs:
    #     st.markdown(
    #         """
    #         <div style="text-align: center; height: 100px; display: flex; align-items: center; justify-content: center;">
    #             <h1 style="color: #ff4b4b; opacity: 0.8; margin-top: 30px;">VS</h1>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    # with h2:
    #     # Placeholder for profile picture - keeping space for future JPGs
    #     st.markdown(
    #         f"""
    #         <div style="text-align: center;">
    #             <div style="font-size: 60px; margin-bottom: 10px; background-color: #f0f2f6; border-radius: 50%; width: 100px; height: 100px; line-height: 100px; margin-left: auto; margin-right: auto;">👤</div>
    #             <h2 style="margin-top: 10px;">{player2}</h2>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )

    # Comparative Metrics
    for label, col, fmt in stats_to_compare:
        if label == "Activities":
            v1, v2 = len(p1_data), len(p2_data)
        else:
            v1, v2 = p1_data[col].sum(), p2_data[col].sum()

        r1, r_lbl, r2 = st.columns([2, 1, 2])

        with r_lbl:
            st.markdown(
                f"<div style='text-align: center; font-weight: bold; color: #888; margin-top: 12px; font-size: 0.8rem;'>{label.upper()}</div>",
                unsafe_allow_html=True,
            )

        def get_stat_html(val, delta, fmt, is_time=False):
            color = "#00c853" if delta > 0 else ("#ff3d00" if delta < 0 else "#747474")
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "")

            if is_time:
                val_str = format_time(val)
                delta_str = f"{arrow} {format_time(abs(delta))}" if delta != 0 else ""
            else:
                val_str = fmt.format(val)
                # For non-time, we use the same formula to show delta
                delta_str = f"{arrow} {abs(delta):,.1f}" if delta != 0 else ""

            return f"""
                <div style="text-align: center; padding: 0;">
                    <div style="font-size: 22px; font-weight: 700; color: #1f1f1f; line-height: 1.2;">{val_str}</div>
                    <div style="font-size: 13px; font-weight: 500; color: {color}; margin-top: -2px;">{delta_str}</div>
                </div>
            """

        with r1:
            st.markdown(
                get_stat_html(v1, v1 - v2, fmt, is_time=(fmt == "time")),
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                get_stat_html(v2, v2 - v1, fmt, is_time=(fmt == "time")),
                unsafe_allow_html=True,
            )


# --- Data Loading ---
df = load_data()
if df.empty:
    st.stop()

df["Month"] = df["Date"].dt.strftime("%Y-%m")
TODAY = datetime.date.today()
CURRENT_MONTH = TODAY.strftime("%Y-%m")


# --- Sidebar Filters ---
st.sidebar.header("Filters")

if st.sidebar.button("Update Data", width="stretch"):
    load_data.clear()
    st.rerun()

st.sidebar.divider()

min_date, max_date = df["Date"].min(), df["Date"].max()
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

all_teams = sorted(df["Team"].unique())
selected_teams = st.sidebar.pills(
    "Select Teams", all_teams, default=all_teams, selection_mode="multi"
)

all_types = sorted(df["Type"].unique())
selected_types = st.sidebar.pills(
    "Activity Type", all_types, default=all_types, selection_mode="multi"
)

all_names = sorted(df["Name"].unique())
selected_names = st.sidebar.pills(
    "Competitors", all_names, default=all_names, selection_mode="multi"
)


# --- Apply Filters ---
categorical_mask = (
    df["Team"].isin(selected_teams)
    & df["Type"].isin(selected_types)
    & df["Name"].isin(selected_names)
)

if len(date_range) == 2:
    start_date, end_date = date_range
    date_mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    filtered_df = df[date_mask & categorical_mask]
else:
    filtered_df = df[categorical_mask]

# History uses categorical filters only (ignores date range)
history_df = df[categorical_mask].copy()


# --- Main Dashboard ---
st.title("Sleep Comp Fitness Challenge")
if len(date_range) == 2:
    st.markdown(
        f"*Tracking activities from **{date_range[0]}** to **{date_range[1]}***"
    )

display_metrics(filtered_df)
st.divider()
render_goal_progress(filtered_df)
st.divider()


# Reserve space for AI section (rendered last to avoid blocking)
ai_placeholder = st.empty()
st.divider()


# --- Leaderboards ---
col_team, col_indiv = st.columns([1, 2])

with col_team:
    # TODO: Add plot of team scores over time
    st.subheader("Team Standings")
    tab_month, tab_year, tab_history = st.tabs(["Month", "Year", "Monthly History"])

    with tab_month:
        st.caption(f"Standings for {TODAY.strftime('%B %Y')}")
        month_data = filtered_df[filtered_df["Month"] == CURRENT_MONTH]
        if not month_data.empty:
            stats = (
                month_data.groupby("Team")[["Effort", "Distance (km)"]]
                .sum()
                .sort_values("Effort", ascending=False)
                .reset_index()
            )
            st.dataframe(stats, width="stretch", hide_index=True)
        else:
            st.info("No activities for the current month.")

    with tab_year:
        stats = (
            filtered_df.groupby("Team")[["Effort", "Distance (km)"]]
            .sum()
            .sort_values("Effort", ascending=False)
            .reset_index()
        )
        st.dataframe(stats, width="stretch", hide_index=True)

    with tab_history:
        st.caption("Winners of each month")
        history_table = get_winner_history(history_df, ["Team"])
        if not history_table.empty:
            st.dataframe(history_table, width="stretch", hide_index=True)
        else:
            st.info("No data available for history.")

    # Win Probability (Monte Carlo)
    st.subheader("Win Probability")
    days_remaining = (COMPETITION_END_DATE - TODAY).days

    if days_remaining > 0:
        daily_effort = df.groupby(["Team", "Date"])["Effort"].sum().reset_index()
        current_totals = df.groupby("Team")["Effort"].sum()
        win_probs = run_monte_carlo_simulation(
            daily_effort, current_totals, days_remaining
        )

        prob_chart = (
            alt.Chart(win_probs)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta(field="Probability", type="quantitative"),
                color=alt.Color(field="Team", type="nominal"),
                tooltip=["Team", alt.Tooltip("Probability", format=".1%")],
                order=alt.Order("Probability", sort="descending"),
            )
            .properties(height=200)
        )

        st.altair_chart(prob_chart, width="stretch")
        st.caption(f"10k simulations over {days_remaining} remaining days.")
    else:
        st.info("Competition has ended.")

with col_indiv:
    st.subheader("Individual Standings")
    indiv_tab_month, indiv_tab_year, indiv_tab_history = st.tabs(
        ["Month", "Year", "Monthly History"]
    )

    indiv_format = {
        "Effort": "{:.1f}",
        "Distance (km)": "{:.1f}",
        "Time (min)": "{:.0f}",
    }

    with indiv_tab_month:
        st.caption(f"Standings for {TODAY.strftime('%B %Y')}")
        month_data_indiv = filtered_df[filtered_df["Month"] == CURRENT_MONTH]
        if not month_data_indiv.empty:
            stats = (
                month_data_indiv.groupby(["Name", "Team"])[
                    ["Effort", "Distance (km)", "Time (min)"]
                ]
                .sum()
                .reset_index()
                .sort_values("Effort", ascending=False)
                .reset_index(drop=True)
            )
            stats.index += 1
            st.dataframe(stats.style.format(indiv_format), width="stretch")
        else:
            st.info("No activities for the current month.")

    with indiv_tab_year:
        stats = (
            filtered_df.groupby(["Name", "Team"])[
                ["Effort", "Distance (km)", "Time (min)"]
            ]
            .sum()
            .reset_index()
            .sort_values("Effort", ascending=False)
            .reset_index(drop=True)
        )
        stats.index += 1
        st.dataframe(stats.style.format(indiv_format), width="stretch")

    with indiv_tab_history:
        st.caption("Winners of each month")
        history_table_indiv = get_winner_history(history_df, ["Name", "Team"])
        if not history_table_indiv.empty:
            st.dataframe(history_table_indiv, width="stretch", hide_index=True)
        else:
            st.info("No data available for history.")

st.divider()


# --- Activity Heatmap ---
st.subheader("Activity Heatmap")
daily_summary = (
    filtered_df.groupby("Date")
    .agg(
        {
            "Effort": "sum",
            "Distance (km)": "sum",
            "Name": lambda x: ", ".join(sorted(x.unique())),
            "Type": lambda x: ", ".join(sorted(x.unique())),
        }
    )
    .reset_index()
)
daily_summary["Week"] = daily_summary["Date"].dt.isocalendar().week
daily_summary["Day"] = daily_summary["Date"].dt.day_name()

heatmap = (
    alt.Chart(daily_summary)
    .mark_rect()
    .encode(
        x=alt.X("Week:O", title="Week of Year"),
        y=alt.Y("Day:N", sort=DAYS_OF_WEEK, title="Day of Week"),
        color=alt.Color(
            "Effort:Q", title="Daily Effort", scale=alt.Scale(scheme="greens")
        ),
        tooltip=[
            alt.Tooltip("Date:T", format="%Y-%m-%d"),
            alt.Tooltip("Effort:Q", format=".1f", title="Total Effort"),
            alt.Tooltip("Distance (km):Q", format=".1f", title="Total Km"),
            alt.Tooltip("Name:N", title="People"),
            alt.Tooltip("Type:N", title="Activities"),
        ],
    )
    .properties(height=250)
    .configure_axis(labelFontSize=10, titleFontSize=12)
)

st.altair_chart(heatmap, width="stretch")


# --- Effort Trends ---
st.subheader("Effort Over Time")
chart_df = filtered_df.sort_values("Date").copy()
chart_df["Cumulative Effort"] = chart_df.groupby("Name")["Effort"].cumsum()

line_chart = (
    alt.Chart(chart_df)
    .mark_line(point=True)
    .encode(
        x="Date:T",
        y="Cumulative Effort:Q",
        color="Name:N",
        tooltip=["Date", "Name", "Type", "Distance (km)", "Effort"],
    )
)

st.altair_chart(line_chart, width="stretch")
st.divider()


# --- Head-to-Head ---
render_h2h_comparison(filtered_df)
st.divider()


# --- Activity Feed ---
st.subheader("Activities")
DISPLAY_COLS = [
    "Date",
    "Name",
    "Team",
    "Type",
    "Distance (km)",
    "Effort",
    "Pace (min/km)",
]
ACTIVITY_FORMAT = {
    "Date": lambda t: t.strftime("%Y-%m-%d"),
    "Distance (km)": "{:.2f}",
    "Effort": "{:.2f}",
    "Pace (min/km)": "{:.2f}",
}
st.dataframe(
    filtered_df.sort_values("Date", ascending=False)[DISPLAY_COLS].style.format(
        ACTIVITY_FORMAT
    ),
    width="stretch",
    hide_index=True,
)


# --- AI Section (Rendered Last) ---
with ai_placeholder.container():
    # Initialize AI data on first load
    if "ai_data" not in st.session_state:
        if get_client():
            data_summary = get_data_summary(filtered_df)
            st.session_state["ai_data"] = get_ai_content_cached(data_summary)
        else:
            st.session_state["ai_data"] = {}

    ai_data = st.session_state.get("ai_data", {})
    manual_facts = get_manual_fun_facts(filtered_df)

    if get_client():
        col_title, col_btn = st.columns([10, 1])
        with col_title:
            st.subheader("Summary")
        with col_btn:
            if st.button("🔄", help="Refresh insights for current filters"):
                with st.spinner("Refreshing..."):
                    data_summary = get_data_summary(filtered_df)
                    st.session_state["ai_data"] = get_ai_content_cached(data_summary)
                st.rerun()

        st.info(
            ai_data.get("insight", "The coach is currently judging you in silence.")
        )
        if "model" in ai_data:
            st.caption(f"Generated with: {ai_data['model']}")

        st.subheader("Fun Facts")
        ai_facts = ai_data.get("facts", [])
        if ai_facts and not any(
            "Error" in f or "AI is being shy" in f for f in ai_facts
        ):
            for fact in ai_facts:
                st.write(f"• {fact}")
        elif not ai_facts:
            for fact in random.sample(manual_facts, min(3, len(manual_facts))):
                st.write(f"• {fact}")
    else:
        st.subheader("Fun Facts")
        st.info(
            "\n\n".join(
                [
                    f"• {fact}"
                    for fact in random.sample(manual_facts, min(3, len(manual_facts)))
                ]
            )
        )
