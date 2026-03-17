"""
Data Transformation Module for GitHub Profile Analyzer.

This module implements a three-layer transformation pipeline:
    Layer 1: Clean and enrich raw commit data
    Layer 2: Aggregate data by time periods
    Layer 3: Compute advanced metrics and scores

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates pandas proficiency with complex transformations
    - Shows understanding of time-series data handling
    - Implements efficient vectorized operations (no loops)
    - Uses proper handling of timezone-aware datetime data
    - Creates derived metrics with business meaning

Usage:
    from src.transformer import clean_commits, aggregate_by_time, compute_advanced_metrics

    clean_df = clean_commits(raw_df)
    aggregations = aggregate_by_time(clean_df)
    metrics = compute_advanced_metrics(clean_df)
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from loguru import logger

from src.config import settings


# =============================================================================
# Layer 1: Data Cleaning and Enrichment
# =============================================================================

def clean_commits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich raw commit data.

    WHY: Raw data from the database needs several transformations
    before analysis:
        - Type conversions (strings to datetime)
        - Missing value handling (empty messages, null dates)
        - Derived columns (time categories, commit sizes)
        - Outlier handling (spam commits, bot activity)

    This function applies all necessary cleaning steps in sequence.

    Args:
        df: Raw commits DataFrame from database

    Returns:
        Cleaned and enriched DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to clean_commits")
        return df

    logger.info(f"Cleaning {len(df)} commits")

    # Create a copy to avoid SettingWithCopyWarning
    # WHY: Modifying a view of a DataFrame can cause unpredictable behavior
    df = df.copy()

    # =========================================================================
    # Step 1: Ensure datetime columns are proper datetime types
    # =========================================================================
    # WHY: Database returns strings for datetime; pandas needs datetime objects
    # for time-series operations like resampling and rolling windows

    datetime_cols = ["author_date", "committer_date"]
    for col in datetime_cols:
        if col in df.columns:
            # Convert to datetime with UTC timezone
            # WHY: UTC ensures consistent time comparisons across timezones
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # =========================================================================
    # Step 2: Handle missing values
    # =========================================================================
    # WHY: Missing values can cause errors in calculations and skew statistics

    # Fill missing messages with empty string
    # WHY: Enables string operations without null checks
    if "message" in df.columns:
        df["message"] = df["message"].fillna("")
        # Recalculate message length for missing values
        df["message_length"] = df["message"].str.len()

    # Fill missing numeric values with defaults
    # WHY: Prevents NaN propagation in aggregations
    numeric_cols = ["hour_of_day", "day_of_week", "week_number", "year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)

    # =========================================================================
    # Step 3: Create derived date columns
    # =========================================================================
    # WHY: Pre-computing these columns enables efficient grouping and filtering

    if "author_date" in df.columns and df["author_date"].notna().any():
        # Extract date components from author_date
        # WHY: Enables grouping by month, week, quarter for trend analysis
        df["commit_date"] = df["author_date"].dt.date
        df["month_year"] = df["author_date"].dt.to_period("M")
        df["quarter"] = df["author_date"].dt.to_period("Q")
        df["year_month"] = df["author_date"].dt.strftime("%Y-%m")

        # Recompute hour and day if they were missing
        # WHY: Database might have gaps; we can recover from the datetime
        mask = df["hour_of_day"] == -1
        df.loc[mask, "hour_of_day"] = df.loc[mask, "author_date"].dt.hour

        mask = df["day_of_week"] == -1
        df.loc[mask, "day_of_week"] = df.loc[mask, "author_date"].dt.weekday

        mask = df["year"] == -1
        df.loc[mask, "year"] = df.loc[mask, "author_date"].dt.year

    # =========================================================================
    # Step 4: Categorize time of day
    # =========================================================================
    # WHY: Time-of-day patterns reveal work habits (night owl vs early bird)

    if "hour_of_day" in df.columns:
        # Define time categories based on hour
        # WHY: Human-readable categories are more meaningful than raw hours
        conditions = [
            (df["hour_of_day"] >= 5) & (df["hour_of_day"] < 12),    # Morning: 5-11
            (df["hour_of_day"] >= 12) & (df["hour_of_day"] < 17),   # Afternoon: 12-16
            (df["hour_of_day"] >= 17) & (df["hour_of_day"] < 21),   # Evening: 17-20
            (df["hour_of_day"] >= 21) | (df["hour_of_day"] < 5),    # Night: 21-4
        ]
        choices = ["morning", "afternoon", "evening", "night"]

        # WHY: np.select is vectorized and much faster than apply with if/else
        df["time_of_day"] = np.select(conditions, choices, default="unknown")

    # =========================================================================
    # Step 5: Categorize commit size by message length
    # =========================================================================
    # WHY: Message length correlates with commit quality and type
    # (small fixes vs large features vs merges)

    if "message_length" in df.columns:
        # Define commit size categories
        # WHY: Industry research shows optimal commit message lengths
        conditions = [
            df["message_length"] < 10,                              # Tiny: < 10 chars (likely automated)
            (df["message_length"] >= 10) & (df["message_length"] < 30),   # Short: 10-29 chars
            (df["message_length"] >= 30) & (df["message_length"] < 72),   # Medium: 30-71 chars (ideal)
            df["message_length"] >= 72,                             # Long: 72+ chars (detailed)
        ]
        choices = ["tiny", "short", "medium", "long"]

        df["commit_size"] = np.select(conditions, choices, default="unknown")

    # =========================================================================
    # Step 6: Detect conventional commits
    # =========================================================================
    # WHY: Conventional commits (feat:, fix:, etc.) indicate mature development practices
    # https://www.conventionalcommits.org/

    if "message" in df.columns:
        # Check for conventional commit prefixes
        # WHY: Regex matching is faster than string methods for patterns
        conventional_pattern = r"^(feat|fix|docs|style|refactor|test|chore|build|ci|perf|revert)(\(.+\))?:"
        df["is_conventional"] = df["message"].str.match(conventional_pattern, case=False, na=False)

    # =========================================================================
    # Step 7: Sort by date and reset index
    # =========================================================================
    # WHY: Sorted data enables efficient time-series operations
    # and ensures correct streak calculations

    if "author_date" in df.columns:
        df = df.sort_values("author_date", ascending=True).reset_index(drop=True)

    logger.info(f"Cleaned {len(df)} commits")

    return df


# =============================================================================
# Layer 2: Time-Based Aggregation
# =============================================================================

def aggregate_by_time(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Aggregate commit data by various time periods.

    WHY: Time-based aggregations reveal patterns that aren't visible
    in raw data:
        - Daily patterns show work rhythm
        - Hourly patterns show preferred coding times
        - Monthly patterns show project lifecycle
        - Yearly patterns show career trajectory

    Args:
        df: Cleaned commits DataFrame

    Returns:
        Dictionary containing:
            - daily_commits: Commits per day
            - hourly_heatmap: Hour × Day of week matrix
            - monthly_commits: Commits per month
            - yearly_summary: Commits and stats per year
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to aggregate_by_time")
        return {
            "daily_commits": pd.DataFrame(),
            "hourly_heatmap": pd.DataFrame(),
            "monthly_commits": pd.DataFrame(),
            "yearly_summary": pd.DataFrame(),
        }

    logger.info("Aggregating data by time periods")

    aggregations = {}

    # =========================================================================
    # Daily Commits Aggregation
    # =========================================================================
    # WHY: Daily aggregations show the rhythm of development work

    if "commit_date" in df.columns:
        # Group by date and count commits
        # WHY: Counting commits per day reveals activity spikes and lulls
        daily = df.groupby("commit_date").agg(
            commit_count=("sha", "count"),
            repo_count=("repo_name", "nunique"),
            avg_message_length=("message_length", "mean"),
        ).reset_index()

        # Convert date back to datetime for time-series operations
        daily["commit_date"] = pd.to_datetime(daily["commit_date"])

        # Add rolling averages
        # WHY: 7-day rolling average smooths out daily noise
        daily["rolling_7day"] = daily["commit_count"].rolling(window=7, min_periods=1).mean()

        # Add 30-day rolling average for trend
        # WHY: 30-day window captures monthly patterns
        daily["rolling_30day"] = daily["commit_count"].rolling(window=30, min_periods=1).mean()

        # Calculate day-over-day change
        # WHY: Shows momentum in activity
        daily["change_from_prev"] = daily["commit_count"].diff()

        aggregations["daily_commits"] = daily
        logger.debug(f"Daily aggregation: {len(daily)} days with activity")

    # =========================================================================
    # Hourly Heatmap (Hour × Day of Week)
    # =========================================================================
    # WHY: Heatmap shows work patterns across time dimensions
    # Identifies preferred coding times and potential overwork

    if "hour_of_day" in df.columns and "day_of_week" in df.columns:
        # Filter out invalid hours and days
        # WHY: -1 indicates missing data; would create invalid matrix cells
        valid_activity = df[(df["hour_of_day"] >= 0) & (df["day_of_week"] >= 0)].copy()

        if not valid_activity.empty:
            # Create pivot table: rows=hour, columns=day_of_week
            # WHY: Pivot table creates the matrix format needed for heatmaps
            heatmap = valid_activity.pivot_table(
                index="hour_of_day",
                columns="day_of_week",
                values="sha",
                aggfunc="count",
                fill_value=0,
            )

            # Rename columns to day names
            # WHY: Human-readable names improve chart readability
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            # Filter colors to existing index range
            actual_days = [int(c) for c in heatmap.columns if 0 <= int(c) < 7]
            heatmap.columns = [day_names[c] for c in actual_days]

            # Ensure all 24 hours are present
            # WHY: Missing hours would cause gaps in the visualization
            for hour in range(24):
                if hour not in heatmap.index:
                    heatmap.loc[hour] = 0

            heatmap = heatmap.sort_index()

            aggregations["hourly_heatmap"] = heatmap
            logger.debug(f"Hourly heatmap: {heatmap.shape}")

    # =========================================================================
    # Monthly Commits Aggregation
    # =========================================================================
    # WHY: Monthly aggregations show project phases and seasonality

    if "month_year" in df.columns:
        # Group by month
        # WHY: Monthly resolution balances detail and overview
        monthly = df.groupby("month_year").agg(
            commit_count=("sha", "count"),
            repo_count=("repo_name", "nunique"),
            unique_languages=("repo_language", "nunique"),
            avg_message_length=("message_length", "mean"),
            conventional_ratio=("is_conventional", "mean"),
        ).reset_index()

        # Convert period to timestamp for plotting
        # WHY: Plotly handles timestamps better than periods
        monthly["month_start"] = monthly["month_year"].dt.to_timestamp()

        # Calculate month-over-month growth
        # WHY: Shows acceleration or deceleration in activity
        monthly["mom_growth"] = monthly["commit_count"].pct_change() * 100

        # Calculate cumulative commits
        # WHY: Shows total progress over time
        monthly["cumulative_commits"] = monthly["commit_count"].cumsum()

        aggregations["monthly_commits"] = monthly
        logger.debug(f"Monthly aggregation: {len(monthly)} months")

    # =========================================================================
    # Yearly Summary
    # =========================================================================
    # WHY: Yearly stats provide high-level career progress overview

    if "year" in df.columns:
        # Filter out invalid years
        valid_years = df[df["year"] > 0].copy()

        if not valid_years.empty:
            yearly = valid_years.groupby("year").agg(
                commit_count=("sha", "count"),
                repo_count=("repo_name", "nunique"),
                unique_languages=("repo_language", "nunique"),
                active_days=("commit_date", "nunique"),
                weekend_commits=("is_weekend", "sum"),
                avg_message_length=("message_length", "mean"),
            ).reset_index()

            # Calculate year-over-year growth
            # WHY: Shows career growth trajectory
            yearly["yoy_growth"] = yearly["commit_count"].pct_change() * 100

            # Calculate weekend ratio
            # WHY: High weekend ratio might indicate overwork or passion project
            yearly["weekend_ratio"] = (yearly["weekend_commits"] / yearly["commit_count"] * 100).round(1)

            # Calculate commits per active day
            # WHY: Shows intensity of work sessions
            yearly["commits_per_day"] = (yearly["commit_count"] / yearly["active_days"]).round(1)

            aggregations["yearly_summary"] = yearly
            logger.debug(f"Yearly summary: {len(yearly)} years")

    return aggregations


# =============================================================================
# Layer 3: Advanced Metrics Computation
# =============================================================================

def compute_advanced_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute advanced metrics from commit data.

    WHY: Advanced metrics provide insights that go beyond simple counts:
        - Consistency score shows reliability
        - Streaks show dedication
        - Peak times show preferences
        - Trends show trajectory
        - Burnout periods show sustainability

    These metrics feed directly into the developer profile generation.

    Args:
        df: Cleaned commits DataFrame

    Returns:
        Dictionary containing:
            - total_commits: Total number of commits
            - total_repos: Number of unique repositories
            - active_days: Number of days with commits
            - consistency_score: 0-100 score for commit regularity
            - longest_streak: Longest consecutive days with commits
            - current_streak: Current consecutive days with commits
            - peak_hour: Hour with most commits
            - peak_day: Day of week with most commits
            - preferred_time: Time category with most commits
            - burnout_periods: List of extended inactive periods
            - productivity_trend: increasing/stable/decreasing
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to compute_advanced_metrics")
        return _empty_metrics()

    logger.info("Computing advanced metrics")

    metrics: Dict[str, Any] = {}

    # =========================================================================
    # Basic Counts
    # =========================================================================
    # WHY: Foundation metrics for all other calculations

    metrics["total_commits"] = len(df)
    metrics["total_repos"] = df["repo_name"].nunique() if "repo_name" in df.columns else 0
    metrics["active_days"] = df["commit_date"].nunique() if "commit_date" in df.columns else 0

    # =========================================================================
    # Consistency Score
    # =========================================================================
    # WHY: Consistency indicates professional habits and reliability
    # Score is based on regularity of commits, not just volume

    if "commit_date" in df.columns and metrics["active_days"] > 1:
        metrics["consistency_score"] = _calculate_consistency_score(df)
    else:
        metrics["consistency_score"] = 0.0

    # =========================================================================
    # Streaks
    # =========================================================================
    # WHY: Streaks show dedication and habit formation
    # GitHub-style streaks are a recognized engagement metric

    if "commit_date" in df.columns and metrics["active_days"] > 0:
        longest, current = _calculate_streaks(df)
        metrics["longest_streak"] = longest
        metrics["current_streak"] = current
    else:
        metrics["longest_streak"] = 0
        metrics["current_streak"] = 0

    # =========================================================================
    # Peak Activity Times
    # =========================================================================
    # WHY: Peak times reveal work style preferences

    if "hour_of_day" in df.columns:
        valid_hours = df[df["hour_of_day"] >= 0]
        if not valid_hours.empty:
            metrics["peak_hour"] = int(valid_hours["hour_of_day"].mode().iloc[0])
        else:
            metrics["peak_hour"] = None
    else:
        metrics["peak_hour"] = None

    if "day_of_week" in df.columns:
        valid_days = df[df["day_of_week"] >= 0]
        if not valid_days.empty:
            metrics["peak_day"] = int(valid_days["day_of_week"].mode().iloc[0])
        else:
            metrics["peak_day"] = None
    else:
        metrics["peak_day"] = None

    if "time_of_day" in df.columns:
        valid_times = df[df["time_of_day"] != "unknown"]
        if not valid_times.empty:
            metrics["preferred_time"] = valid_times["time_of_day"].mode().iloc[0]
        else:
            metrics["preferred_time"] = "unknown"
    else:
        metrics["preferred_time"] = "unknown"

    # =========================================================================
    # Burnout Periods
    # =========================================================================
    # WHY: Extended inactivity may indicate burnout, job change, or life events
    # Helps identify sustainability issues in work patterns

    if "commit_date" in df.columns and metrics["active_days"] > 0:
        metrics["burnout_periods"] = _detect_burnout_periods(df)
    else:
        metrics["burnout_periods"] = []

    # =========================================================================
    # Productivity Trend
    # =========================================================================
    # WHY: Trend shows trajectory (improving, stable, declining)
    # Important for understanding career development

    if "commit_date" in df.columns and metrics["active_days"] >= 30:
        metrics["productivity_trend"] = _calculate_productivity_trend(df)
    else:
        metrics["productivity_trend"] = "insufficient_data"

    # =========================================================================
    # Work Style Classification
    # =========================================================================
    # WHY: Work style helps match developers with team cultures

    metrics["work_style"] = _classify_work_style(df, metrics)

    # =========================================================================
    # Commit Quality Metrics
    # =========================================================================
    # WHY: Quality metrics show professionalism beyond volume

    if "message_length" in df.columns:
        metrics["avg_message_length"] = round(df["message_length"].mean(), 1)
        metrics["median_message_length"] = df["message_length"].median()

    if "is_conventional" in df.columns:
        metrics["conventional_commit_ratio"] = round(
            df["is_conventional"].mean() * 100, 1
        )

    # =========================================================================
    # Weekend/Work Balance
    # =========================================================================
    # WHY: Work-life balance indicator

    if "is_weekend" in df.columns:
        weekend_commits = df["is_weekend"].sum()
        metrics["weekend_commit_count"] = int(weekend_commits)
        metrics["weekend_commit_ratio"] = round(
            (weekend_commits / len(df)) * 100, 1
        ) if len(df) > 0 else 0.0

    logger.info(f"Computed {len(metrics)} metrics")

    return metrics


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics structure for empty DataFrames."""
    return {
        "total_commits": 0,
        "total_repos": 0,
        "active_days": 0,
        "consistency_score": 0.0,
        "longest_streak": 0,
        "current_streak": 0,
        "peak_hour": None,
        "peak_day": None,
        "preferred_time": "unknown",
        "burnout_periods": [],
        "productivity_trend": "insufficient_data",
        "work_style": "unknown",
    }


def _calculate_consistency_score(df: pd.DataFrame) -> float:
    """
    Calculate consistency score based on commit regularity.

    WHY: A developer who commits consistently is more reliable than one
    who commits sporadically, even if total counts are similar.

    Formula:
        1. Calculate the coefficient of variation (CV) of daily commits
        2. Convert to score: lower CV = higher consistency
        3. Adjust for coverage (days with commits / total days in range)

    Args:
        df: Commits DataFrame

    Returns:
        Consistency score from 0 to 100
    """
    # Group by date
    daily_counts = df.groupby("commit_date").size()

    if len(daily_counts) < 7:
        # Not enough data for meaningful consistency measurement
        return 50.0  # Neutral score

    # Calculate coefficient of variation
    # WHY: CV = std/mean normalizes variation for comparison
    mean_commits = daily_counts.mean()
    std_commits = daily_counts.std()

    if mean_commits == 0:
        return 0.0

    cv = std_commits / mean_commits

    # Convert CV to score (lower CV = higher score)
    # WHY: CV of 0 would mean perfect consistency (all days equal)
    # CV of 1+ indicates high variability
    # We use sigmoid-like transformation for smooth scoring
    raw_score = 100 / (1 + cv)

    # Calculate coverage
    # WHY: Penalize large gaps in activity
    date_range = (df["commit_date"].max() - df["commit_date"].min()).days + 1
    coverage = len(daily_counts) / date_range if date_range > 0 else 0

    # Combine scores
    # WHY: Weight both regularity and coverage equally
    final_score = raw_score * 0.6 + coverage * 100 * 0.4

    return round(min(100, max(0, final_score)), 1)


def _calculate_streaks(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Calculate longest and current commit streaks.

    WHY: Streaks are a motivating metric popularized by GitHub's
    contribution graph. They show dedication and habit formation.

    A streak is consecutive days with at least one commit.

    Args:
        df: Commits DataFrame

    Returns:
        Tuple of (longest_streak, current_streak)
    """
    # Get unique dates with commits
    commit_dates = df["commit_date"].unique()
    commit_dates = sorted(pd.to_datetime(commit_dates).date)

    if not commit_dates:
        return 0, 0

    # Calculate streaks
    streaks = []
    current_streak = 1
    longest_streak = 1

    for i in range(1, len(commit_dates)):
        # Check if dates are consecutive
        # WHY: Consecutive days form a streak; gaps break it
        prev_date = commit_dates[i - 1]
        curr_date = commit_dates[i]
        diff = (curr_date - prev_date).days

        if diff == 1:
            # Consecutive day - extend streak
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        elif diff > 1:
            # Gap - end streak and start new one
            streaks.append(current_streak)
            current_streak = 1

    streaks.append(current_streak)

    # Check if current streak is active (includes today or yesterday)
    # WHY: A streak should only be "current" if it's still ongoing
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    last_commit = commit_dates[-1]

    if last_commit >= yesterday:
        current = current_streak
    else:
        current = 0

    return longest_streak, current


def _detect_burnout_periods(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect periods of extended inactivity that may indicate burnout.

    WHY: Extended breaks from coding can indicate:
        - Burnout (negative)
        - Job change (neutral)
        - Vacation (positive - work-life balance)

    This helps identify sustainability patterns.

    Args:
        df: Commits DataFrame

    Returns:
        List of burnout period dictionaries with start, end, duration
    """
    # Get unique dates with commits
    commit_dates = set(pd.to_datetime(df["commit_date"].unique()).date)

    if len(commit_dates) < 2:
        return []

    # Find date range
    min_date = min(commit_dates)
    max_date = max(commit_dates)

    # Find all dates in range
    # WHY: Need to identify gaps in the full date range
    all_dates = set()
    current = min_date
    while current <= max_date:
        all_dates.add(current)
        current += timedelta(days=1)

    # Find missing dates (gaps)
    # WHY: Gaps represent periods of inactivity
    missing_dates = sorted(all_dates - commit_dates)

    if not missing_dates:
        return []

    # Group consecutive missing dates into periods
    # WHY: A single gap of 14 days is more significant than two gaps of 7 days
    burnout_periods = []
    period_start = missing_dates[0]
    period_dates = [missing_dates[0]]

    for i in range(1, len(missing_dates)):
        if (missing_dates[i] - missing_dates[i - 1]).days == 1:
            # Consecutive missing day
            period_dates.append(missing_dates[i])
        else:
            # Gap in missing dates - end current period
            if len(period_dates) >= settings.burnout_threshold_days:
                burnout_periods.append({
                    "start_date": period_start.isoformat(),
                    "end_date": period_dates[-1].isoformat(),
                    "duration_days": len(period_dates),
                })

            # Start new period
            period_start = missing_dates[i]
            period_dates = [missing_dates[i]]

    # Don't forget the last period
    if len(period_dates) >= settings.burnout_threshold_days:
        burnout_periods.append({
            "start_date": period_start.isoformat(),
            "end_date": period_dates[-1].isoformat(),
            "duration_days": len(period_dates),
        })

    return burnout_periods


def _calculate_productivity_trend(df: pd.DataFrame) -> str:
    """
    Calculate productivity trend using linear regression.

    WHY: Trend analysis shows trajectory:
        - Increasing: growing activity (learning, new projects)
        - Stable: consistent contribution level
        - Decreasing: declining activity (project completion, burnout)

    Uses linear regression on daily commit counts over time.

    Args:
        df: Commits DataFrame

    Returns:
        Trend classification: "increasing", "stable", or "decreasing"
    """
    # Get daily commit counts
    daily = df.groupby("commit_date").size().reset_index(name="count")
    daily["commit_date"] = pd.to_datetime(daily["commit_date"])
    daily = daily.sort_values("commit_date")

    if len(daily) < 14:
        return "insufficient_data"

    # Convert dates to numeric for regression
    # WHY: Linear regression needs numeric input
    daily["days_since_start"] = (daily["commit_date"] - daily["commit_date"].min()).dt.days

    # Perform linear regression
    # WHY: Slope indicates trend direction
    x = daily["days_since_start"].values
    y = daily["count"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Classify trend based on slope and R-squared
    # WHY: Slope alone doesn't account for noise; R² indicates trend strength
    r_squared = r_value ** 2

    # Threshold for significant trend
    # WHY: Arbitrary but reasonable thresholds for classification
    if slope > 0.01 and r_squared > 0.1:
        return "increasing"
    elif slope < -0.01 and r_squared > 0.1:
        return "decreasing"
    else:
        return "stable"


def _classify_work_style(df: pd.DataFrame, metrics: Dict[str, Any]) -> str:
    """
    Classify work style based on time patterns.

    WHY: Work style helps understand when a developer is most productive
    and can inform team scheduling and collaboration patterns.

    Classification:
        - early_bird: Most commits in morning (5-11)
        - afternoon_warrior: Most commits in afternoon (12-16)
        - evening_person: Most commits in evening (17-20)
        - night_owl: Most commits at night (21-4)
        - balanced: Relatively even distribution

    Args:
        df: Commits DataFrame
        metrics: Computed metrics dictionary

    Returns:
        Work style classification string
    """
    if "time_of_day" not in df.columns:
        return "unknown"

    # Count commits by time of day
    time_counts = df["time_of_day"].value_counts()

    if time_counts.empty:
        return "unknown"

    # Get the dominant time category
    dominant = time_counts.index[0]
    dominant_ratio = time_counts.iloc[0] / time_counts.sum()

    # Check if distribution is balanced
    # WHY: Balanced distribution indicates flexible working style
    if dominant_ratio < 0.35:
        return "balanced"

    # Map time categories to work styles
    style_mapping = {
        "morning": "early_bird",
        "afternoon": "afternoon_warrior",
        "evening": "evening_person",
        "night": "night_owl",
    }

    return style_mapping.get(dominant, "balanced")


# =============================================================================
# Utility Functions
# =============================================================================

def get_commit_velocity(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """
    Calculate commit velocity (commits per day) over a rolling window.

    WHY: Velocity shows the rate of contribution, which is different
    from total count. A developer with fewer commits but higher velocity
    might be working intensively on a focused project.

    Args:
        df: Commits DataFrame
        window_days: Rolling window size in days

    Returns:
        DataFrame with date and velocity columns
    """
    if df.empty or "commit_date" not in df.columns:
        return pd.DataFrame(columns=["date", "velocity"])

    # Group by date
    daily = df.groupby("commit_date").size().reset_index(name="count")
    daily["commit_date"] = pd.to_datetime(daily["commit_date"])
    daily = daily.sort_values("commit_date")

    # Calculate rolling velocity
    # WHY: Rolling window smooths out daily fluctuations
    daily["velocity"] = daily["count"].rolling(
        window=window_days,
        min_periods=1
    ).mean()

    return daily[["commit_date", "velocity"]].rename(columns={"commit_date": "date"})


# =============================================================================
# Export Public Interface
# =============================================================================

__all__ = [
    "clean_commits",
    "aggregate_by_time",
    "compute_advanced_metrics",
    "get_commit_velocity",
]
