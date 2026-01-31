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
    time_str = f"{days}d {hours}h" if days > 0 else f"{hours}h {minutes}m"

    cols = st.columns(5)
    cols[0].metric("Total Distance", f"{total_km:,.1f} km")
    cols[1].metric("Total Effort", f"{total_effort:,.1f}")
    cols[2].metric("Total Time", time_str)
    cols[3].metric("Total Activities", f"{len(data)}")
    cols[4].metric("Total Elevation", f"{int(total_elevation)} m")


def render_goal_progress(data: pd.DataFrame, today: datetime.date) -> None:
    """Render a progress bar for the group distance goal with predictions."""
    # # --- January Challenge ---
    # st.markdown("---")
    # st.subheader("January 1000 km Goal")

    # jan_goal = 1000.0
    # jan_start = datetime.date(COMPETITION_START_DATE.year, 1, 1)
    # jan_end = datetime.date(COMPETITION_START_DATE.year, 1, 31)

    # # Filter for January data
    # jan_mask = (data["Date"].dt.date >= jan_start) & (data["Date"].dt.date <= jan_end)
    # jan_distance = data.loc[jan_mask, "Distance (km)"].sum()
    # jan_progress = min(1.0, jan_distance / jan_goal)

    # # Calculate Jan prediction
    # days_passed_jan = (min(today, jan_end) - jan_start).days + 1
    # days_passed_jan = max(1, days_passed_jan)
    # jan_total_days = 31

    # if days_passed_jan > 0:
    #     jan_predicted = (jan_distance / days_passed_jan) * jan_total_days
    # else:
    #     jan_predicted = 0

    # st.progress(jan_progress)

    # j1, j2 = st.columns(2)
    # j1.metric("Jan Distance", f"{jan_distance:,.1f}/{jan_goal:,.0f} km")
    # j2.metric(
    #     "Jan Projected",
    #     f"{jan_predicted:,.1f} km",
    #     delta=f"{jan_predicted - jan_goal:,.1f} km",
    # )

    total_distance = data["Distance (km)"].sum()
    progress = min(1.0, total_distance / GROUP_DISTANCE_GOAL)

    # Calculate prediction based on daily average
    days_elapsed = (today - COMPETITION_START_DATE).days + 1
    if days_elapsed > 0:
        predicted_total = (total_distance / days_elapsed) * 365
    else:
        predicted_total = 0

    st.subheader("Goal Progress")
    st.progress(progress)

    c1, c2 = st.columns(2)
    c1.metric("Goal", f"{total_distance:,.1f}/{GROUP_DISTANCE_GOAL:,.0f} km")
    c2.metric(
        "Projected Total",
        f"{predicted_total:,.1f} km",
        delta=f"{predicted_total - GROUP_DISTANCE_GOAL:,.1f} km",
        help="Prediction based on current daily average extrapolated to 365 days.",
    )


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
    daily_effort_df: pd.DataFrame,
    current_totals: pd.Series,
    days_remaining: int,
    today: datetime.date,
) -> pd.DataFrame:
    """Run Monte Carlo simulation to estimate win probabilities."""
    all_dates = pd.date_range(start=COMPETITION_START_DATE, end=today)
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


# --- Render Functions ---
def render_activity_heatmap(data: pd.DataFrame) -> None:
    """Render the activity heatmap showing daily effort."""
    st.subheader("Activity Heatmap")

    daily_summary = (
        data.groupby("Date")
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


def render_team_standings(
    filtered_df: pd.DataFrame,
    history_df: pd.DataFrame,
    today: datetime.date,
    current_month: str,
) -> None:
    """Render the team standings tabs (Month, Year, History)."""
    st.subheader("Standings")
    tab_month, tab_year, tab_history = st.tabs(["Month", "Year", "History"])

    with tab_month:
        st.caption(f"Standings for {today.strftime('%B %Y')}")
        month_data = filtered_df[filtered_df["Month"] == current_month]
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


def render_win_probability(df: pd.DataFrame, today: datetime.date) -> None:
    """Render the win probability donut chart."""
    st.subheader("Win Probability")
    days_remaining = (COMPETITION_END_DATE - today).days

    if days_remaining > 0:
        daily_effort = df.groupby(["Team", "Date"])["Effort"].sum().reset_index()
        current_totals = df.groupby("Team")["Effort"].sum()
        win_probs = run_monte_carlo_simulation(
            daily_effort, current_totals, days_remaining, today
        )

        prob_chart = (
            alt.Chart(win_probs)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta(field="Probability", type="quantitative"),
                color=alt.Color(field="Team"),
                tooltip=["Team", alt.Tooltip("Probability", format=".1%")],
                order=alt.Order("Probability", sort="descending"),
            )
            .properties(height=230)
        )

        st.altair_chart(prob_chart, width="stretch")
        st.caption(f"10k simulations over {days_remaining} remaining days.")
    else:
        st.info("Competition has ended.")


def render_team_effort_chart(data: pd.DataFrame) -> None:
    """Render the cumulative team effort line chart."""
    st.subheader("Effort")

    team_daily = data.groupby(["Team", "Date"])["Effort"].sum().reset_index()
    team_daily = team_daily.sort_values("Date")
    team_daily["Cumulative Effort"] = team_daily.groupby("Team")["Effort"].cumsum()

    team_line_chart = (
        alt.Chart(team_daily)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Cumulative Effort:Q", title="Effort"),
            color=alt.Color("Team:N"),
            tooltip=["Date", "Team", "Cumulative Effort"],
        )
        .properties(height=230)
    )
    st.altair_chart(team_line_chart, width="stretch")


def render_individual_standings(
    filtered_df: pd.DataFrame,
    history_df: pd.DataFrame,
    today: datetime.date,
    current_month: str,
) -> None:
    """Render the individual standings tabs (Month, Year, History)."""
    st.subheader("Standings")
    indiv_tab_month, indiv_tab_year, indiv_tab_history = st.tabs(
        ["Month", "Year", "History"]
    )

    indiv_format = {
        "Effort": "{:.1f}",
        "Distance (km)": "{:.1f}",
        "Time (min)": "{:.0f}",
    }

    with indiv_tab_month:
        st.caption(f"Standings for {today.strftime('%B %Y')}")
        month_data_indiv = filtered_df[filtered_df["Month"] == current_month]
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


def render_individual_effort_chart(data: pd.DataFrame) -> None:
    """Render the cumulative individual effort line chart."""
    st.subheader("Effort")

    chart_df = data.sort_values("Date").copy()
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


def render_activity_feed(data: pd.DataFrame) -> None:
    """Render the activity feed table."""
    st.header("Activities")

    display_cols = [
        "Date",
        "Name",
        "Team",
        "Type",
        "Distance (km)",
        "Time (min)",
        "Pace (min/km)",
        "Effort",
    ]
    activity_format = {
        "Date": lambda t: t.strftime("%Y-%m-%d"),
        "Distance (km)": "{:.2f}",
        "Time (min)": "{:.2f}",
        "Pace (min/km)": "{:.2f}",
        "Effort": "{:.2f}",
    }

    st.dataframe(
        data.sort_values("Date", ascending=False)[display_cols].style.format(
            activity_format
        ),
        width="stretch",
        hide_index=True,
    )


def render_ai_section(data: pd.DataFrame) -> None:
    """Render the AI-powered insights and fun facts section."""
    # Initialize AI data on first load
    if "ai_data" not in st.session_state:
        if get_client():
            data_summary = get_data_summary(data)
            st.session_state["ai_data"] = get_ai_content_cached(data_summary)
        else:
            st.session_state["ai_data"] = {}

    ai_data = st.session_state.get("ai_data", {})
    manual_facts = get_manual_fun_facts(data)

    if get_client():
        col_title, col_btn = st.columns([10, 1])
        with col_title:
            st.subheader("Summary")
        with col_btn:
            if st.button("🔄", help="Refresh insights for current filters"):
                with st.spinner("Refreshing..."):
                    data_summary = get_data_summary(data)
                    st.session_state["ai_data"] = get_ai_content_cached(data_summary)
                st.rerun()

        st.info(
            ai_data.get("insight", "The coach is currently judging you in silence.")
        )
        if "model" in ai_data:
            st.caption(f"Generated with: {ai_data['model']}")

        st.subheader("Key Insights")
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
        st.subheader("Insights")
        st.info(
            "\n\n".join(
                [
                    f"• {fact}"
                    for fact in random.sample(manual_facts, min(3, len(manual_facts)))
                ]
            )
        )


def render_sidebar(df: pd.DataFrame) -> tuple:
    """Render sidebar filters and return filter values."""
    st.sidebar.header("Filters")

    # --- Data Refresh & Reset Controls ---
    col_update, col_reset = st.sidebar.columns(2)
    with col_update:
        if st.button("Refresh", use_container_width=True, help="Fetch latest data"):
            load_data.clear()
            st.rerun()
    with col_reset:
        if st.button("Reset", use_container_width=True, help="Reset all filters"):
            for key in list(st.session_state.keys()):
                if key.startswith("filter_"):
                    del st.session_state[key]
            st.rerun()

    st.sidebar.divider()

    # --- Date Range Filter ---
    min_date, max_date = df["Date"].min(), df["Date"].max()
    today = datetime.date.today()

    # Initialize date filter state
    if "filter_date_preset" not in st.session_state:
        st.session_state["filter_date_preset"] = "All Time"

    with st.sidebar.expander("Date Range", expanded=True):
        # Quick date presets
        date_presets = {
            "Last 7 Days": (today - datetime.timedelta(days=7), today),
            "Last 30 Days": (today - datetime.timedelta(days=30), today),
            "This Month": (today.replace(day=1), today),
            "Last Month": (
                (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1),
                today.replace(day=1) - datetime.timedelta(days=1),
            ),
            "Year to Date": (datetime.date(today.year, 1, 1), today),
            "All Time": (min_date.date(), max_date.date()),
        }

        selected_preset = st.radio(
            "Quick Select",
            options=list(date_presets.keys()),
            horizontal=True,
            key="filter_date_preset",
            label_visibility="collapsed",
        )

        # Get preset dates, clamped to available data range
        preset_start, preset_end = date_presets[selected_preset]
        preset_start = max(preset_start, min_date.date())
        preset_end = min(preset_end, max_date.date())

        # Custom date input (updates based on preset)
        date_range = st.date_input(
            "Custom Range",
            value=(preset_start, preset_end),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )

    # --- Team Filter ---
    all_teams = sorted(df["Team"].unique())

    if "filter_teams" not in st.session_state:
        st.session_state["filter_teams"] = all_teams

    teams_label = "Teams"

    with st.sidebar.expander(teams_label, expanded=True):
        col_all, col_clear = st.columns(2)
        with col_all:
            if st.button("Select All", key="teams_all", use_container_width=True):
                st.session_state["filter_teams"] = all_teams
                st.rerun()
        with col_clear:
            if st.button("Clear", key="teams_clear", use_container_width=True):
                st.session_state["filter_teams"] = []
                st.rerun()

        selected_teams = st.pills(
            "Teams",
            all_teams,
            selection_mode="multi",
            key="filter_teams",
            label_visibility="collapsed",
        )

    # --- Activity Type Filter ---
    all_types = sorted(df["Type"].unique())

    if "filter_types" not in st.session_state:
        st.session_state["filter_types"] = all_types

    types_label = "Activity Type"

    with st.sidebar.expander(types_label, expanded=True):
        col_all, col_clear = st.columns(2)
        with col_all:
            if st.button("Select All", key="types_all", use_container_width=True):
                st.session_state["filter_types"] = all_types
                st.rerun()
        with col_clear:
            if st.button("Clear", key="types_clear", use_container_width=True):
                st.session_state["filter_types"] = []
                st.rerun()

        selected_types = st.pills(
            "Types",
            all_types,
            selection_mode="multi",
            key="filter_types",
            label_visibility="collapsed",
        )

    # --- Competitors Filter ---
    all_names = sorted(df["Name"].unique())

    if "filter_names" not in st.session_state:
        st.session_state["filter_names"] = all_names

    names_label = "Competitors"

    with st.sidebar.expander(names_label, expanded=True):
        col_all, col_clear = st.columns(2)
        with col_all:
            if st.button("Select All", key="names_all", use_container_width=True):
                st.session_state["filter_names"] = all_names
                st.rerun()
        with col_clear:
            if st.button("Clear", key="names_clear", use_container_width=True):
                st.session_state["filter_names"] = []
                st.rerun()

        selected_names = st.pills(
            "Competitors",
            all_names,
            selection_mode="multi",
            key="filter_names",
            label_visibility="collapsed",
        )

    return date_range, selected_teams, selected_types, selected_names


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple,
    selected_teams: list,
    selected_types: list,
    selected_names: list,
) -> tuple:
    """Apply filters to the dataframe and return filtered + history dataframes."""
    categorical_mask = (
        df["Team"].isin(selected_teams)
        & df["Type"].isin(selected_types)
        & df["Name"].isin(selected_names)
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        date_mask = (df["Date"].dt.date >= start_date) & (
            df["Date"].dt.date <= end_date
        )
        filtered_df = df[date_mask & categorical_mask]
    else:
        filtered_df = df[categorical_mask]

    # History uses categorical filters only (ignores date range)
    history_df = df[categorical_mask].copy()

    return filtered_df, history_df


# --- Main Application ---
def main():
    """Main application entry point."""
    # Data Loading
    df = load_data()
    if df.empty:
        st.stop()

    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    today = datetime.date.today()
    current_month = today.strftime("%Y-%m")

    # Sidebar Filters
    date_range, selected_teams, selected_types, selected_names = render_sidebar(df)
    filtered_df, history_df = apply_filters(
        df, date_range, selected_teams, selected_types, selected_names
    )

    # Main Dashboard Header
    st.title("Sleep Comp Fitness Challenge")
    if len(date_range) == 2:
        st.markdown(
            f"*Tracking activities from **{date_range[0]}** to **{date_range[1]}***"
        )

    display_metrics(filtered_df)
    st.divider()

    # Reserve space for AI section (rendered last to avoid blocking)
    ai_placeholder = st.empty()
    st.divider()

    # Group Section
    st.header("Group")
    render_activity_heatmap(filtered_df)
    render_goal_progress(filtered_df, today)
    st.divider()

    # Team Section
    st.header("Team")
    col_team_stats, col_win_prob = st.columns(2)

    with col_team_stats:
        render_team_standings(filtered_df, history_df, today, current_month)

    with col_win_prob:
        render_win_probability(df, today)

    render_team_effort_chart(filtered_df)
    st.divider()

    # Individual Section
    st.header("Individual")
    render_individual_standings(filtered_df, history_df, today, current_month)
    render_individual_effort_chart(filtered_df)
    st.divider()

    # Activity Feed
    render_activity_feed(filtered_df)

    # AI Section (Rendered Last)
    with ai_placeholder.container():
        render_ai_section(filtered_df)


if __name__ == "__main__":
    main()
