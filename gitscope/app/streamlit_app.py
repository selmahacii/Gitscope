"""
GitScope - Streamlit Dashboard.

See the full scope of any GitHub developer.

This module provides an interactive web dashboard:
    - User input for GitHub username and token
    - Real-time data collection with progress indicators
    - Multiple tabs for different analysis views
    - AI-generated insights display

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates Streamlit proficiency for rapid prototyping
    - Shows UI/UX considerations for data dashboards
    - Implements proper session state management
    - Handles errors gracefully with user feedback
    - Creates professional, interactive data visualizations

Usage:
    streamlit run app/streamlit_app.py

Or with custom port:
    streamlit run app/streamlit_app.py --server.port 8501
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Add project root to path for imports
# WHY: Streamlit runs from a different working directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.config import settings
from src.collector import GitHubCollector, GitHubAPIError, UserNotFoundError
from src.storage import save_all_data, load_commits_df, load_repos_df, load_languages_df, get_db_stats
from src.transformer import clean_commits, aggregate_by_time, compute_advanced_metrics
from src.analytics import (
    analyze_languages,
    analyze_repositories,
    analyze_commit_messages,
    generate_developer_profile,
)
from src.visualizations import (
    commit_heatmap,
    monthly_activity,
    language_breakdown,
    productivity_timeline,
    repo_stars_chart,
    score_gauge,
)
from src.insights import generate_insights, generate_fun_facts, generate_basic_insights


# =============================================================================
# Page Configuration
# =============================================================================

# WHY: Page config must be the first Streamlit command
st.set_page_config(
    page_title="GitScope - GitHub Profile Analyzer",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": """
        **GitScope - GitHub Profile Analyzer**

        See the full scope of any GitHub developer.

        Built with Python, Streamlit, and ZhipuAI GLM.
        """,
        "Get help": None,
        "Report a bug": None,
    },
)


# =============================================================================
# Session State Initialization
# =============================================================================

# WHY: Session state persists data between reruns
# Without this, data would be lost on every widget interaction

def init_session_state() -> None:
    """Initialize all session state variables."""
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False

    if "current_username" not in st.session_state:
        st.session_state["current_username"] = ""

    if "raw_data" not in st.session_state:
        st.session_state["raw_data"] = None

    if "commits_df" not in st.session_state:
        st.session_state["commits_df"] = pd.DataFrame()

    if "repos_df" not in st.session_state:
        st.session_state["repos_df"] = pd.DataFrame()

    if "languages_df" not in st.session_state:
        st.session_state["languages_df"] = pd.DataFrame()

    if "metrics" not in st.session_state:
        st.session_state["metrics"] = {}

    if "aggregations" not in st.session_state:
        st.session_state["aggregations"] = {}

    if "lang_analysis" not in st.session_state:
        st.session_state["lang_analysis"] = {}

    if "repo_analysis" not in st.session_state:
        st.session_state["repo_analysis"] = {}

    if "commit_analysis" not in st.session_state:
        st.session_state["commit_analysis"] = {}

    if "profile" not in st.session_state:
        st.session_state["profile"] = {}

    if "insights" not in st.session_state:
        st.session_state["insights"] = {}

    if "fun_facts" not in st.session_state:
        st.session_state["fun_facts"] = {}


init_session_state()


# =============================================================================
# Helper Functions
# =============================================================================

def run_analysis(username: str, token: Optional[str], force_refresh: bool) -> bool:
    """
    Run the complete analysis pipeline.

    WHY: Centralized analysis function keeps code DRY and ensures
    consistent execution order.

    Args:
        username: GitHub username to analyze
        token: Optional GitHub token
        force_refresh: Whether to skip cache

    Returns:
        True if successful, False otherwise
    """
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    try:
        # Step 1: Collect data (0-40%)
        status_text.text("📡 Fetching data from GitHub API...")
        with GitHubCollector(username=username, token=token) as collector:
            if force_refresh:
                collector.cache.clear()

            raw_data = collector.collect_all()

        st.session_state["raw_data"] = raw_data
        progress_bar.progress(40, text="Data collection complete")

        # Step 2: Save to database (40-50%)
        status_text.text("💾 Saving to database...")
        save_all_data(raw_data, settings.db_path)
        progress_bar.progress(50, text="Database updated")

        # Step 3: Load DataFrames (50-60%)
        status_text.text("📊 Loading data for analysis...")
        commits_df = load_commits_df(settings.db_path, username)
        repos_df = load_repos_df(settings.db_path, username)
        languages_df = load_languages_df(settings.db_path, username)

        st.session_state["commits_df"] = commits_df
        st.session_state["repos_df"] = repos_df
        st.session_state["languages_df"] = languages_df
        progress_bar.progress(60, text="Data loaded")

        # Step 4: Transform data (60-70%)
        status_text.text("🔄 Transforming data...")
        clean_df = clean_commits(commits_df)
        aggregations = aggregate_by_time(clean_df)
        metrics = compute_advanced_metrics(clean_df)

        st.session_state["clean_df"] = clean_df
        st.session_state["aggregations"] = aggregations
        st.session_state["metrics"] = metrics
        progress_bar.progress(70, text="Transformation complete")

        # Step 5: Run analytics (70-80%)
        status_text.text("📈 Computing analytics...")
        lang_analysis = analyze_languages(languages_df)
        repo_analysis = analyze_repositories(repos_df)
        commit_analysis = analyze_commit_messages(clean_df)

        st.session_state["lang_analysis"] = lang_analysis
        st.session_state["repo_analysis"] = repo_analysis
        st.session_state["commit_analysis"] = commit_analysis
        progress_bar.progress(80, text="Analytics computed")

        # Step 6: Generate profile (80-85%)
        status_text.text("👤 Generating developer profile...")
        profile = generate_developer_profile(
            metrics=metrics,
            languages=lang_analysis,
            repos=repo_analysis,
            commits=commit_analysis,
        )
        st.session_state["profile"] = profile
        progress_bar.progress(85, text="Profile generated")

        # Step 7: Generate AI insights (85-100%)
        status_text.text("🤖 Generating AI insights...")
        insights = generate_insights(profile)
        fun_facts = generate_fun_facts(metrics)

        st.session_state["insights"] = insights
        st.session_state["fun_facts"] = fun_facts
        progress_bar.progress(100, text="Analysis complete!")

        # Mark as complete
        st.session_state["analysis_complete"] = True
        st.session_state["current_username"] = username

        return True

    except UserNotFoundError:
        st.error(f"❌ User '{username}' not found on GitHub.")
        return False

    except GitHubAPIError as e:
        st.error(f"❌ GitHub API error: {e.message}")
        if "rate limit" in str(e.message).lower():
            st.info("💡 Tip: Add a GitHub token to increase your rate limit.")
        return False

    except Exception as e:
        logger.exception("Analysis failed")
        st.error(f"❌ Analysis failed: {str(e)}")
        return False


def render_metrics_cards(profile: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """
    Render the key metrics cards at the top of the dashboard.

    WHY: Metrics cards provide quick overview of key statistics
    without requiring scrolling or tab switching.

    Args:
        profile: Developer profile dictionary
        metrics: Advanced metrics dictionary
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Commits",
            value=f"{metrics.get('total_commits', 0):,}",
            help="Total number of commits analyzed",
        )

    with col2:
        st.metric(
            label="Active Days",
            value=f"{metrics.get('active_days', 0):,}",
            help="Number of days with at least one commit",
        )

    with col3:
        longest_streak = metrics.get("longest_streak", 0)
        current_streak = metrics.get("current_streak", 0)
        st.metric(
            label="Longest Streak",
            value=f"{longest_streak} days",
            delta=f"Current: {current_streak} days",
            help="Longest consecutive days with commits",
        )

    with col4:
        scores = profile.get("scores", {})
        st.metric(
            label="Overall Score",
            value=f"{scores.get('overall', 0):.0f}/100",
            help="Composite score based on consistency, quality, impact, and diversity",
        )

    with col5:
        languages = profile.get("languages", {})
        st.metric(
            label="Primary Language",
            value=languages.get("primary", "Unknown"),
            help="Most-used programming language by code volume",
        )


def render_activity_tab() -> None:
    """Render the Activity Analysis tab."""
    st.subheader("📈 Activity Analysis")

    metrics = st.session_state.get("metrics", {})
    aggregations = st.session_state.get("aggregations", {})
    profile = st.session_state.get("profile", {})

    # Score gauges row
    st.write("#### Profile Scores")
    col1, col2, col3, col4 = st.columns(4)

    scores = profile.get("scores", {})

    with col1:
        fig = score_gauge(scores.get("consistency", 0), "Consistency")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = score_gauge(scores.get("quality", 0), "Quality")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        fig = score_gauge(scores.get("impact", 0), "Impact")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = score_gauge(scores.get("diversity", 0), "Diversity")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Activity charts
    col_left, col_right = st.columns(2)

    with col_left:
        # Monthly activity chart
        monthly_df = aggregations.get("monthly_commits", pd.DataFrame())
        if not monthly_df.empty:
            fig = monthly_activity(monthly_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monthly activity data available")

    with col_right:
        # Productivity timeline
        daily_df = aggregations.get("daily_commits", pd.DataFrame())
        if not daily_df.empty:
            fig = productivity_timeline(daily_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available")

    # Commit heatmap
    st.write("#### Commit Activity Heatmap")
    heatmap_df = aggregations.get("hourly_heatmap", pd.DataFrame())
    if not heatmap_df.empty:
        fig = commit_heatmap(heatmap_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No heatmap data available")

    # Work pattern metrics
    st.write("#### Work Pattern Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        peak_hour = metrics.get("peak_hour")
        st.metric("Peak Hour", f"{peak_hour}:00" if peak_hour else "N/A")

    with col2:
        peak_day = metrics.get("peak_day")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        st.metric("Peak Day", days[peak_day] if peak_day is not None else "N/A")

    with col3:
        st.metric("Preferred Time", metrics.get("preferred_time", "Unknown").title())

    with col4:
        st.metric("Work Style", metrics.get("work_style", "Unknown").replace("_", " ").title())

    # Burnout indicator
    burnout_periods = metrics.get("burnout_periods", [])
    if burnout_periods:
        st.warning(f"⚠️ {len(burnout_periods)} potential burnout period(s) detected")
        with st.expander("View details"):
            for period in burnout_periods:
                st.write(f"- {period['start_date']} to {period['end_date']} ({period['duration_days']} days)")


def render_languages_tab() -> None:
    """Render the Languages Analysis tab."""
    st.subheader("🔤 Language Analysis")

    lang_analysis = st.session_state.get("lang_analysis", {})

    if not lang_analysis:
        st.info("No language data available")
        return

    # Top row: key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Languages Used", lang_analysis.get("language_count", 0))

    with col2:
        st.metric("Primary Language", lang_analysis.get("primary_language", "Unknown"))

    with col3:
        st.metric("Diversity Score", f"{lang_analysis.get('diversity_score', 0):.2f}")

    with col4:
        st.metric("Diversity Level", lang_analysis.get("diversity_level", "Unknown").title())

    st.divider()

    # Language breakdown chart
    top_languages = lang_analysis.get("top_languages", [])
    if top_languages:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            fig = language_breakdown(top_languages)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.write("#### Top Languages")
            for lang in top_languages[:5]:
                st.write(f"- **{lang['language']}**: {lang['percentage']:.1f}%")

    # Category breakdown
    category_breakdown = lang_analysis.get("category_breakdown", {})
    if category_breakdown:
        st.write("#### Technology Categories")
        cols = st.columns(len(category_breakdown))
        for i, (category, percentage) in enumerate(category_breakdown.items()):
            if i < len(cols):
                with cols[i]:
                    st.metric(category.title(), f"{percentage}%")


def render_repositories_tab() -> None:
    """Render the Repositories Analysis tab."""
    st.subheader("📁 Repository Analysis")

    repo_analysis = st.session_state.get("repo_analysis", {})

    if not repo_analysis:
        st.info("No repository data available")
        return

    # Top row: key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Repos", repo_analysis.get("total_repos", 0))

    with col2:
        st.metric("Total Stars", f"{repo_analysis.get('total_stars', 0):,}")

    with col3:
        st.metric("Total Forks", f"{repo_analysis.get('total_forks', 0):,}")

    with col4:
        st.metric("Active Repos", repo_analysis.get("active_repos", 0))

    with col5:
        st.metric("Avg Stars/Repo", f"{repo_analysis.get('avg_stars', 0):.1f}")

    st.divider()

    # Top repositories chart
    top_repos = repo_analysis.get("top_repositories", [])
    if top_repos:
        st.write("#### Top Repositories by Stars")
        fig = repo_stars_chart(top_repos)
        st.plotly_chart(fig, use_container_width=True)

    # Repository details table
    st.write("#### Repository Details")
    repos_df = st.session_state.get("repos_df", pd.DataFrame())
    if not repos_df.empty:
        # Select relevant columns
        display_cols = ["name", "primary_language", "stargazers_count", "forks_count", "is_archived"]
        available_cols = [c for c in display_cols if c in repos_df.columns]

        if available_cols:
            df_display = repos_df[available_cols].head(20)
            df_display = df_display.rename(columns={
                "name": "Repository",
                "primary_language": "Language",
                "stargazers_count": "Stars",
                "forks_count": "Forks",
                "is_archived": "Archived",
            })
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    # Topics analysis
    topics = repo_analysis.get("topic_frequency", [])
    if topics:
        st.write("#### Common Topics")
        topic_cols = st.columns(min(len(topics), 5))
        for i, topic in enumerate(topics[:5]):
            if i < len(topic_cols):
                with topic_cols[i]:
                    st.metric(topic["topic"], f"{topic['count']} repos")


def render_insights_tab() -> None:
    """Render the AI Insights tab."""
    st.subheader("🤖 AI Insights")

    insights = st.session_state.get("insights", {})
    fun_facts = st.session_state.get("fun_facts", {})
    profile = st.session_state.get("profile", {})

    # Check for errors
    if insights.get("error"):
        st.warning(f"AI insights unavailable: {insights['error']}")
        st.info("💡 Tip: Set ZHIPUAI_API_KEY environment variable to enable AI insights.")

        # Show basic insights as fallback
        st.write("#### Basic Analysis")
        basic_insights = generate_basic_insights(profile)
        st.markdown(basic_insights)
        return

    # Display AI-generated report
    if insights.get("report"):
        st.markdown(insights["report"])

        # Generation info
        st.caption(
            f"Generated using {insights.get('model', 'unknown')} "
            f"in {insights.get('generation_time_ms', 0)}ms "
            f"({insights.get('tokens_used', 0)} tokens)"
        )

    # Fun facts
    facts = fun_facts.get("facts", [])
    if facts:
        st.divider()
        st.write("#### 🎉 Fun Facts")
        for fact in facts:
            st.info(f"✨ {fact}")


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar() -> None:
    """Render the sidebar with inputs and controls."""
    with st.sidebar:
        st.title("🔭 GitScope")
        st.markdown("---")

        # Input section
        st.write("### Configuration")

        username = st.text_input(
            "GitHub Username",
            value=st.session_state.get("current_username", ""),
            placeholder="e.g., octocat",
            help="Enter the GitHub username to analyze",
        )

        token = st.text_input(
            "GitHub Token (Optional)",
            type="password",
            help="Increases rate limit from 60 to 5000 requests/hour",
        )

        force_refresh = st.checkbox(
            "Force Refresh",
            help="Skip cache and fetch fresh data from GitHub API",
        )

        st.markdown("---")

        # Analyze button
        analyze_button = st.button(
            "🚀 Analyze Profile",
            type="primary",
            use_container_width=True,
            disabled=not username,
        )

        if analyze_button:
            # Clear previous results
            st.session_state["analysis_complete"] = False
            success = run_analysis(username, token or None, force_refresh)
            if success:
                st.rerun()

        st.markdown("---")

        # Cache status
        st.write("### Cache Status")
        db_stats = get_db_stats(settings.db_path)
        st.write(f"Users: {db_stats.get('users', 0)}")
        st.write(f"Repositories: {db_stats.get('repositories', 0)}")
        st.write(f"Commits: {db_stats.get('commits', 0)}")

        if st.button("Clear Cache", use_container_width=True):
            cache_dir = Path("./cache")
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                st.success("Cache cleared!")
                st.rerun()

        st.markdown("---")

        # Info section
        st.write("### About")
        st.markdown("""
        This tool analyzes GitHub profiles to provide insights about:

        - **Activity patterns**: When and how often you code
        - **Technology focus**: Your primary languages and skills
        - **Repository impact**: Stars, forks, and contributions
        - **Work style**: Night owl vs early bird, consistency

        Built with Python, Streamlit, and ZhipuAI GLM.
        """)


# =============================================================================
# Main Application
# =============================================================================

def main() -> None:
    """Main application entry point."""
    # Render sidebar
    render_sidebar()

    # Main content area
    if not st.session_state.get("analysis_complete"):
        # Welcome screen
        st.title("🔭 GitScope")
        st.markdown("""
        **See the full scope of any GitHub developer.**

        - **Activity Analysis**: Commit patterns, streaks, and productivity trends
        - **Language Analysis**: Technology stack and language diversity
        - **Repository Analysis**: Project impact and focus areas
        - **AI Insights**: LLM-powered analysis and recommendations

        ### Getting Started

        1. Enter a GitHub username in the sidebar
        2. (Optional) Add a GitHub token for higher rate limits
        3. Click "Analyze Profile" to begin

        ### Example Profiles to Try

        - `torvalds` - Linus Torvalds (Linux creator)
        - `gvanrossum` - Guido van Rossum (Python creator)
        - `sindresorhus` - Sindre Sorhus (Prolific open source maintainer)
        """)

        return

    # Analysis is complete, show results
    profile = st.session_state.get("profile", {})
    metrics = st.session_state.get("metrics", {})
    labels = profile.get("labels", {})

    # Header with user info
    col1, col2 = st.columns([1, 4])

    with col1:
        raw_data = st.session_state.get("raw_data", {})
        avatar_url = raw_data.get("profile", {}).get("avatar_url")
        if avatar_url:
            st.image(avatar_url, width=150)

    with col2:
        st.title(f"@{st.session_state.get('current_username', 'Unknown')}")
        st.markdown(f"""
        **{labels.get('developer_type', 'Developer')}** | \
        {labels.get('activity_level', 'Unknown')} Activity | \
        {labels.get('experience_level', 'Unknown experience')}
        """)
        st.markdown(profile.get("summary", ""))

    st.divider()

    # Metrics cards
    render_metrics_cards(profile, metrics)

    st.divider()

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Activity",
        "🔤 Languages",
        "📁 Repositories",
        "🤖 AI Insights",
    ])

    with tab1:
        render_activity_tab()

    with tab2:
        render_languages_tab()

    with tab3:
        render_repositories_tab()

    with tab4:
        render_insights_tab()


if __name__ == "__main__":
    main()
