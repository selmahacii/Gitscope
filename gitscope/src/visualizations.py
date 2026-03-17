"""
Visualization Module for GitHub Profile Analyzer.

This module creates interactive Plotly visualizations:
    - Commit activity heatmap (hour × day)
    - Language evolution over time
    - Monthly activity bar chart
    - Language breakdown donut chart
    - Productivity timeline (GitHub-style calendar)
    - Repository stars comparison

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates data visualization skills with Plotly
    - Shows ability to create interactive, web-ready charts
    - Implements proper color schemes and accessibility
    - Creates professional-quality visualizations for dashboards
    - Handles edge cases (empty data, missing values)

Usage:
    from src.visualizations import (
        commit_heatmap,
        language_breakdown,
        monthly_activity,
    )

    fig = commit_heatmap(hourly_df)
    fig.show()
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from loguru import logger


# =============================================================================
# Color Schemes
# =============================================================================

# Professional color palette
# WHY: Consistent colors create a cohesive visual identity
# Using a blue-green gradient for activity (professional, not distracting)
COLORS = {
    # Primary palette
    "primary": "#2563eb",       # Blue
    "secondary": "#10b981",     # Green
    "accent": "#f59e0b",        # Amber
    "danger": "#ef4444",        # Red

    # Activity levels (GitHub-inspired)
    "activity": {
        "none": "#161b22",
        "low": "#0e4429",
        "medium": "#006d32",
        "high": "#26a641",
        "very_high": "#39d353",
    },

    # Chart colors (categorical)
    "chart": [
        "#2563eb",  # Blue
        "#10b981",  # Green
        "#f59e0b",  # Amber
        "#ef4444",  # Red
        "#8b5cf6",  # Purple
        "#06b6d4",  # Cyan
        "#f97316",  # Orange
        "#84cc16",  # Lime
        "#ec4899",  # Pink
        "#6366f1",  # Indigo
    ],

    # Language colors (approximating GitHub's language colors)
    "languages": {
        "Python": "#3572A5",
        "JavaScript": "#f1e05a",
        "TypeScript": "#2b7489",
        "Java": "#b07219",
        "C": "#555555",
        "C++": "#f34b7d",
        "C#": "#178600",
        "Go": "#00ADD8",
        "Rust": "#dea584",
        "Ruby": "#701516",
        "PHP": "#4F5D95",
        "Swift": "#ffac45",
        "Kotlin": "#F18E33",
        "Shell": "#89e051",
        "HTML": "#e34c26",
        "CSS": "#563d7c",
        "Vue": "#41b883",
        "Dart": "#00B4AB",
        "Scala": "#c22d40",
        "R": "#198CE7",
    },
}


# =============================================================================
# Chart Layout Configuration
# =============================================================================

def get_base_layout(title: str, height: int = 400) -> Dict[str, Any]:
    """
    Get base layout configuration for all charts.

    WHY: Consistent layout creates professional appearance across all charts.
    Centralizing layout config makes updates easier.

    Args:
        title: Chart title
        height: Chart height in pixels

    Returns:
        Dictionary of layout configuration
    """
    return {
        "title": {
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16, "color": "#1f2937"},
        },
        "height": height,
        "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    }


# =============================================================================
# Commit Activity Heatmap
# =============================================================================

def commit_heatmap(
    hourly_df: pd.DataFrame,
    title: str = "Commit Activity Heatmap",
) -> go.Figure:
    """
    Create a heatmap showing commit activity by hour and day of week.

    WHY: Heatmaps reveal work patterns:
        - Preferred coding times
        - Work-life balance indicators
        - Consistency of schedule

    The x-axis shows days of the week, y-axis shows hours.
    Color intensity represents commit count.

    Args:
        hourly_df: DataFrame with hourly heatmap data (from aggregate_by_time)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    logger.debug("Creating commit activity heatmap")

    if hourly_df.empty:
        logger.warning("Empty DataFrame for heatmap, creating placeholder")
        return _create_empty_chart(title, "No commit data available")

    # Create figure
    fig = go.Figure()

    # Add heatmap trace
    fig.add_trace(go.Heatmap(
        z=hourly_df.values,
        x=hourly_df.columns.tolist(),
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale=[
            [0, COLORS["activity"]["none"]],
            [0.2, COLORS["activity"]["low"]],
            [0.4, COLORS["activity"]["medium"]],
            [0.7, COLORS["activity"]["high"]],
            [1, COLORS["activity"]["very_high"]],
        ],
        showscale=True,
        colorbar={
            "title": "Commits",
            "titleside": "right",
            "thickness": 15,
            "len": 0.8,
        },
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Hour: %{y}<br>"
            "Commits: %{z}<extra></extra>"
        ),
    ))

    # Update layout
    layout = get_base_layout(title, height=500)
    layout.update({
        "xaxis": {
            "title": "Day of Week",
            "tickangle": 0,
        },
        "yaxis": {
            "title": "Hour of Day",
            "autorange": "reversed",  # Midnight at top
        },
    })
    fig.update_layout(**layout)

    return fig


# =============================================================================
# Language Evolution Chart
# =============================================================================

def language_evolution(
    monthly_df: pd.DataFrame,
    top_n: int = 5,
    title: str = "Language Evolution Over Time",
) -> go.Figure:
    """
    Create a stacked area chart showing language usage over time.

    WHY: Shows how developer's technology focus has evolved:
        - Learning new technologies
        - Project type changes
        - Skill development trajectory

    Uses stacked area for clear visual comparison.

    Args:
        monthly_df: DataFrame with monthly language data
        top_n: Number of top languages to show
        title: Chart title

    Returns:
        Plotly Figure object
    """
    logger.debug("Creating language evolution chart")

    # Create placeholder if no data
    if monthly_df.empty:
        return _create_empty_chart(title, "No language evolution data available")

    fig = go.Figure()

    # Get columns to plot (exclude non-language columns)
    exclude_cols = ["month_year", "month_start", "commit_count", "repo_count"]
    language_cols = [c for c in monthly_df.columns if c not in exclude_cols][:top_n]

    if not language_cols:
        return _create_empty_chart(title, "No language data available")

    # Add area trace for each language
    for i, lang in enumerate(language_cols):
        color = COLORS["languages"].get(lang, COLORS["chart"][i % len(COLORS["chart"])])

        fig.add_trace(go.Scatter(
            x=monthly_df["month_start"],
            y=monthly_df[lang],
            name=lang,
            mode="lines",
            stackgroup="one",
            line={"width": 0.5, "color": color},
            fillcolor=color,
            hovertemplate=f"<b>{lang}</b>: %{{y}}<extra></extra>",
        ))

    # Update layout
    layout = get_base_layout(title, height=400)
    layout.update({
        "xaxis": {"title": "Month"},
        "yaxis": {"title": "Commits"},
        "hovermode": "x unified",
    })
    fig.update_layout(**layout)

    return fig


# =============================================================================
# Monthly Activity Chart
# =============================================================================

def monthly_activity(
    monthly_df: pd.DataFrame,
    title: str = "Monthly Commit Activity",
) -> go.Figure:
    """
    Create a bar chart with rolling average line showing monthly commits.

    WHY: Shows activity trends over time:
        - Growth or decline in activity
        - Seasonal patterns
        - Project lifecycle stages

    Combines bars (actual) with line (trend) for clarity.

    Args:
        monthly_df: DataFrame with monthly commit data (from aggregate_by_time)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    logger.debug("Creating monthly activity chart")

    if monthly_df.empty:
        return _create_empty_chart(title, "No monthly activity data available")

    fig = go.Figure()

    # Add bar trace for monthly commits
    fig.add_trace(go.Bar(
        x=monthly_df["month_start"],
        y=monthly_df["commit_count"],
        name="Monthly Commits",
        marker_color=COLORS["primary"],
        opacity=0.7,
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Commits: %{y}<extra></extra>"
        ),
    ))

    # Add rolling average line if available
    # WHY: Rolling average smooths noise and shows trend
    if "rolling_30day" in monthly_df.columns:
        fig.add_trace(go.Scatter(
            x=monthly_df["month_start"],
            y=monthly_df["rolling_30day"] * 30,  # Scale to monthly
            name="30-Day Average",
            mode="lines",
            line={"color": COLORS["secondary"], "width": 2},
            hovertemplate="Avg: %{y:.0f}/month<extra></extra>",
        ))

    # Update layout
    layout = get_base_layout(title, height=400)
    layout.update({
        "xaxis": {"title": "Month"},
        "yaxis": {"title": "Commits"},
        "bargap": 0.1,
    })
    fig.update_layout(**layout)

    return fig


# =============================================================================
# Language Breakdown Donut Chart
# =============================================================================

def language_breakdown(
    languages: List[Dict[str, Any]],
    title: str = "Language Breakdown",
) -> go.Figure:
    """
    Create a donut chart showing language distribution.

    WHY: Shows technology focus at a glance:
        - Primary expertise area
        - Breadth of skills
        - Technology investment balance

    Donut chart allows for clean percentage display in center.

    Args:
        languages: List of language dicts with 'language' and 'percentage' keys
        title: Chart title

    Returns:
        Plotly Figure object
    """
    logger.debug("Creating language breakdown chart")

    if not languages:
        return _create_empty_chart(title, "No language data available")

    fig = go.Figure()

    # Extract data
    labels = [lang["language"] for lang in languages[:10]]  # Top 10
    values = [lang["percentage"] for lang in languages[:10]]
    colors = [
        COLORS["languages"].get(lang, COLORS["chart"][i % len(COLORS["chart"])])
        for i, lang in enumerate(labels)
    ]

    # Add pie trace (donut)
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="percent",
        textposition="outside",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "%{value:.1f}%<extra></extra>"
        ),
    ))

    # Update layout
    layout = get_base_layout(title, height=400)
    layout.update({
        "showlegend": True,
        "legend": {
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.05,
        },
    })
    fig.update_layout(**layout)

    # Add center annotation
    if languages:
        primary = languages[0]["language"]
        primary_pct = languages[0]["percentage"]
        fig.add_annotation(
            text=f"<b>{primary}</b><br>{primary_pct:.0f}%",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False,
        )

    return fig


# =============================================================================
# Productivity Timeline (GitHub-style Calendar)
# =============================================================================

def productivity_timeline(
    daily_df: pd.DataFrame,
    title: str = "Contribution Activity",
) -> go.Figure:
    """
    Create a GitHub-style contribution calendar heatmap.

    WHY: Shows activity pattern across the year:
        - Consistent vs sporadic contributions
        - Busy periods and lulls
        - Overall activity level

    This mirrors GitHub's contribution graph for familiarity.

    Args:
        daily_df: DataFrame with daily commit data (from aggregate_by_time)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    logger.debug("Creating productivity timeline")

    if daily_df.empty:
        return _create_empty_chart(title, "No activity data available")

    fig = go.Figure()

    # Ensure date column is datetime
    daily_df = daily_df.copy()
    daily_df["commit_date"] = pd.to_datetime(daily_df["commit_date"])

    # Add scatter plot with varying marker sizes
    # WHY: Scatter plot with date axis handles irregular dates better
    sizes = daily_df["commit_count"] * 3 + 5  # Scale for visibility

    fig.add_trace(go.Scatter(
        x=daily_df["commit_date"],
        y=daily_df["commit_count"],
        mode="markers",
        marker={
            "size": sizes,
            "color": daily_df["commit_count"],
            "colorscale": [
                [0, COLORS["activity"]["low"]],
                [0.5, COLORS["activity"]["medium"]],
                [1, COLORS["activity"]["very_high"]],
            ],
            "showscale": True,
            "colorbar": {"title": "Commits", "thickness": 15},
            "line": {"width": 0.5, "color": "white"},
        },
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            "Commits: %{y}<extra></extra>"
        ),
        name="Daily Activity",
    ))

    # Add trend line
    # WHY: Trend line shows overall direction
    if len(daily_df) >= 7:
        # Calculate rolling average for trend
        daily_sorted = daily_df.sort_values("commit_date")
        rolling = daily_sorted["commit_count"].rolling(window=7, min_periods=1).mean()

        fig.add_trace(go.Scatter(
            x=daily_sorted["commit_date"],
            y=rolling,
            mode="lines",
            name="7-Day Average",
            line={"color": COLORS["secondary"], "width": 2, "dash": "dash"},
            hovertemplate="Avg: %{y:.1f}<extra></extra>",
        ))

    # Update layout
    layout = get_base_layout(title, height=300)
    layout.update({
        "xaxis": {"title": "Date"},
        "yaxis": {"title": "Commits"},
        "hovermode": "x unified",
    })
    fig.update_layout(**layout)

    return fig


# =============================================================================
# Repository Stars Chart
# =============================================================================

def repo_stars_chart(
    repos: List[Dict[str, Any]],
    title: str = "Top Repositories by Stars",
) -> go.Figure:
    """
    Create a horizontal bar chart showing repository stars.

    WHY: Shows project impact at a glance:
        - Most successful projects
        - Range of popularity
        - Project focus areas

    Horizontal bars allow for readable repository names.

    Args:
        repos: List of repo dicts with 'name' and 'stars' keys
        title: Chart title

    Returns:
        Plotly Figure object
    """
    logger.debug("Creating repository stars chart")

    if not repos:
        return _create_empty_chart(title, "No repository data available")

    fig = go.Figure()

    # Sort by stars and take top 10
    sorted_repos = sorted(repos, key=lambda x: x.get("stars", 0), reverse=True)[:10]
    sorted_repos.reverse()  # Reverse for horizontal bar (highest at top)

    names = [r["name"] for r in sorted_repos]
    stars = [r.get("stars", 0) for r in sorted_repos]

    # Add horizontal bar trace
    fig.add_trace(go.Bar(
        y=names,
        x=stars,
        orientation="h",
        marker_color=COLORS["primary"],
        opacity=0.8,
        text=stars,
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Stars: %{x}<extra></extra>"
        ),
    ))

    # Update layout
    layout = get_base_layout(title, height=350)
    layout.update({
        "xaxis": {"title": "Stars"},
        "yaxis": {"title": ""},
        "showlegend": False,
        "margin": {"l": 120, "r": 60, "t": 60, "b": 60},
    })
    fig.update_layout(**layout)

    return fig


# =============================================================================
# Work Pattern Chart
# =============================================================================

def work_pattern_chart(
    time_distribution: Dict[str, int],
    title: str = "Work Pattern Distribution",
) -> go.Figure:
    """
    Create a radar chart showing work pattern distribution.

    WHY: Shows work style preferences:
        - Time of day preferences
        - Balance between morning/afternoon/evening/night

    Radar chart provides intuitive visualization of distribution.

    Args:
        time_distribution: Dict mapping time categories to counts
        title: Chart title

    Returns:
        Plotly Figure object
    """
    logger.debug("Creating work pattern chart")

    if not time_distribution:
        return _create_empty_chart(title, "No work pattern data available")

    fig = go.Figure()

    # Prepare data
    categories = ["Morning", "Afternoon", "Evening", "Night"]
    values = [
        time_distribution.get("morning", 0),
        time_distribution.get("afternoon", 0),
        time_distribution.get("evening", 0),
        time_distribution.get("night", 0),
    ]

    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])

    # Add radar trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="Activity",
        fillcolor=COLORS["primary"],
        line={"color": COLORS["primary"]},
        opacity=0.6,
    ))

    # Update layout
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
        },
        polar={
            "radialaxis": {"visible": True, "range": [0, max(values) * 1.2] if values else [0, 1]},
        },
        height=350,
        showlegend=False,
    )

    return fig


# =============================================================================
# Score Gauge Chart
# =============================================================================

def score_gauge(
    score: float,
    title: str = "Overall Score",
    max_score: float = 100,
) -> go.Figure:
    """
    Create a gauge chart showing a score.

    WHY: Single metric visualization is powerful for dashboards:
        - Quick assessment
        - Color-coded status
        - Progress indication

    Args:
        score: The score value to display
        title: Gauge title
        max_score: Maximum possible score

    Returns:
        Plotly Figure object
    """
    logger.debug(f"Creating score gauge for {title}")

    # Determine color based on score
    if score >= 70:
        color = COLORS["secondary"]  # Green
    elif score >= 40:
        color = COLORS["accent"]     # Amber
    else:
        color = COLORS["danger"]     # Red

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_score], "ticksuffix": ""},
            "bar": {"color": color},
            "bgcolor": "#f3f4f6",
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, max_score], "color": "#d1fae5"},
            ],
            "threshold": {
                "line": {"color": "#374151", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"suffix": "", "font": {"size": 24}},
    ))

    fig.update_layout(
        height=250,
        margin={"l": 30, "r": 30, "t": 50, "b": 30},
    )

    return fig


# =============================================================================
# Combined Dashboard
# =============================================================================

def create_dashboard(
    aggregations: Dict[str, pd.DataFrame],
    analysis: Dict[str, Any],
    metrics: Dict[str, Any],
) -> go.Figure:
    """
    Create a combined dashboard with multiple charts.

    WHY: Dashboard view provides comprehensive overview:
        - All key metrics at once
        - Correlations between different views
        - Professional presentation

    Args:
        aggregations: Dict of aggregated DataFrames
        analysis: Dict containing language, repo, commit analyses
        metrics: Dict of advanced metrics

    Returns:
        Plotly Figure object with subplots
    """
    logger.info("Creating combined dashboard")

    # Create subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Monthly Activity",
            "Language Distribution",
            "Activity Heatmap",
            "Work Pattern",
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "heatmap"}, {"type": "polar"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Note: Due to subplot complexity, individual charts are recommended
    # This serves as a template for dashboard layout

    fig.update_layout(
        height=800,
        title={
            "text": "GitHub Profile Dashboard",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
        showlegend=True,
    )

    return fig


# =============================================================================
# Helper Functions
# =============================================================================

def _create_empty_chart(title: str, message: str) -> go.Figure:
    """
    Create an empty chart with a message when data is unavailable.

    WHY: Graceful handling of missing data improves user experience
    and clearly indicates why no visualization is shown.

    Args:
        title: Chart title
        message: Message to display

    Returns:
        Plotly Figure with text annotation
    """
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        font={"size": 14, "color": "#6b7280"},
        showarrow=False,
    )

    layout = get_base_layout(title, height=300)
    layout["xaxis"] = {"visible": False}
    layout["yaxis"] = {"visible": False}
    fig.update_layout(**layout)

    return fig


def save_chart(fig: go.Figure, filename: str, format: str = "png") -> bool:
    """
    Save a chart to a file.

    WHY: Exporting charts enables:
        - Report generation
        - Presentation creation
        - Documentation

    Args:
        fig: Plotly Figure to save
        filename: Output filename (without extension)
        format: Output format (png, svg, html)

    Returns:
        True if successful, False otherwise
    """
    try:
        if format == "html":
            fig.write_html(f"{filename}.html")
        elif format == "png":
            fig.write_image(f"{filename}.png", scale=2)
        elif format == "svg":
            fig.write_image(f"{filename}.svg")
        else:
            logger.error(f"Unknown format: {format}")
            return False

        logger.info(f"Saved chart to {filename}.{format}")
        return True

    except Exception as e:
        logger.error(f"Failed to save chart: {e}")
        return False


# =============================================================================
# Export Public Interface
# =============================================================================

__all__ = [
    "commit_heatmap",
    "language_evolution",
    "monthly_activity",
    "language_breakdown",
    "productivity_timeline",
    "repo_stars_chart",
    "work_pattern_chart",
    "score_gauge",
    "create_dashboard",
    "save_chart",
    "COLORS",
]
