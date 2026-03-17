"""
Analytics Module for GitHub Profile Analyzer.

This module implements the core analytical functions:
    - Language analysis (diversity, primary, evolution)
    - Repository analysis (stars, forks, topics, activity)
    - Commit message analysis (quality, patterns, conventions)
    - Developer profile generation (labels, summaries)

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates data analysis skills with pandas
    - Shows understanding of statistical measures (entropy, z-scores)
    - Implements classification algorithms for developer profiles
    - Creates meaningful metrics from raw data
    - Shows domain knowledge of software development patterns

Usage:
    from src.analytics import analyze_languages, generate_developer_profile

    lang_analysis = analyze_languages(languages_df)
    profile = generate_developer_profile(metrics, languages, repos, commits)
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy

from loguru import logger


# =============================================================================
# Language Analysis
# =============================================================================

def analyze_languages(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze language distribution and diversity.

    WHY: Language analysis reveals:
        - Technology stack and expertise areas
        - Breadth vs depth of skills
        - Evolution of technology focus over time

    Uses Shannon entropy for diversity measurement, which provides
    a mathematically sound way to quantify how "spread out" the
    language usage is.

    Args:
        df: Languages DataFrame from load_languages_df()

    Returns:
        Dictionary containing:
            - language_count: Total unique languages used
            - primary_language: Most-used language by bytes
            - diversity_score: Shannon entropy (0 = single language, higher = more diverse)
            - top_languages: List of top languages with percentages
            - language_evolution: Language usage by year (if available)
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to analyze_languages")
        return _empty_language_analysis()

    logger.info(f"Analyzing {len(df)} language entries")

    result: Dict[str, Any] = {}

    # =========================================================================
    # Aggregate languages across all repositories
    # =========================================================================
    # WHY: We want the overall language profile, not per-repo breakdown

    # Sum bytes by language across all repos
    # BUSINESS LOGIC: Bytes of code indicate actual usage, not just presence
    lang_totals = df.groupby("language")["bytes_count"].sum().sort_values(ascending=False)

    total_bytes = lang_totals.sum()
    result["total_bytes"] = total_bytes

    # =========================================================================
    # Calculate language count and primary
    # =========================================================================
    # WHY: Simple but important metrics for quick assessment

    result["language_count"] = len(lang_totals)
    result["primary_language"] = lang_totals.index[0] if len(lang_totals) > 0 else None

    # =========================================================================
    # Calculate diversity score using Shannon entropy
    # =========================================================================
    # WHY: Shannon entropy measures information content and diversity
    # A developer using only Python: entropy = 0 (no surprise/diversity)
    # A developer using 10 languages equally: entropy = log2(10) ≈ 3.32

    # Calculate proportions (probabilities)
    proportions = lang_totals / total_bytes

    # Shannon entropy formula: H = -Σ(p_i * log2(p_i))
    # BUSINESS LOGIC: Higher entropy = more diverse technology stack
    # Scale: 0-1 (single language), 1-2 (moderate), 2+ (highly diverse)
    diversity = entropy(proportions, base=2)
    result["diversity_score"] = round(diversity, 3)

    # Classify diversity level
    # WHY: Raw entropy is hard to interpret; labels help
    if diversity < 0.5:
        result["diversity_level"] = "specialized"
    elif diversity < 1.5:
        result["diversity_level"] = "moderate"
    else:
        result["diversity_level"] = "diverse"

    # =========================================================================
    # Calculate top languages with percentages
    # =========================================================================
    # WHY: Shows the relative weight of each technology

    top_n = 10  # Show top 10 languages
    top_languages = []

    for i, (lang, bytes_count) in enumerate(lang_totals.head(top_n).items()):
        percentage = (bytes_count / total_bytes * 100) if total_bytes > 0 else 0
        top_languages.append({
            "rank": i + 1,
            "language": lang,
            "bytes": int(bytes_count),
            "percentage": round(percentage, 2),
        })

    result["top_languages"] = top_languages

    # =========================================================================
    # Calculate language categories
    # =========================================================================
    # WHY: Grouping languages helps understand developer focus areas

    # Define language categories
    # BUSINESS LOGIC: Categories help match developers to roles
    categories = {
        "frontend": ["JavaScript", "TypeScript", "HTML", "CSS", "Vue", "Svelte", "Angular"],
        "backend": ["Python", "Java", "Go", "Rust", "C", "C++", "C#", "Ruby", "PHP"],
        "data": ["Python", "R", "Julia", "MATLAB", "SAS"],
        "mobile": ["Swift", "Kotlin", "Java", "Dart", "Objective-C"],
        "systems": ["C", "C++", "Rust", "Assembly", "Go"],
        "scripting": ["Python", "JavaScript", "Ruby", "PHP", "Perl", "Shell"],
    }

    # Calculate percentage per category
    # BUSINESS LOGIC: Identifies primary and secondary skill areas
    category_totals = {}
    for category, langs in categories.items():
        category_bytes = lang_totals[lang_totals.index.isin(langs)].sum()
        category_totals[category] = round((category_bytes / total_bytes * 100), 1) if total_bytes > 0 else 0

    result["category_breakdown"] = category_totals

    # Determine primary category
    # WHY: Helps match developers to roles (frontend vs backend vs full-stack)
    if category_totals:
        primary_category = max(category_totals, key=category_totals.get)
        result["primary_category"] = primary_category
        result["primary_category_percentage"] = category_totals[primary_category]

    # =========================================================================
    # Language evolution over time (if we have repo timestamps)
    # =========================================================================
    # WHY: Shows how developer's technology focus has evolved

    if "repo_name" in df.columns and len(df) > 0:
        # This would require joining with repo dates
        # For now, we'll leave this as a placeholder
        result["language_evolution"] = None

    logger.info(
        f"Language analysis complete: {result['language_count']} languages, "
        f"diversity={result['diversity_score']:.2f}, primary={result['primary_language']}"
    )

    return result


def _empty_language_analysis() -> Dict[str, Any]:
    """Return empty language analysis structure."""
    return {
        "language_count": 0,
        "primary_language": None,
        "diversity_score": 0.0,
        "diversity_level": "unknown",
        "top_languages": [],
        "category_breakdown": {},
        "total_bytes": 0,
    }


# =============================================================================
# Repository Analysis
# =============================================================================

def analyze_repositories(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze repository statistics and patterns.

    WHY: Repository analysis reveals:
        - Impact (stars, forks)
        - Focus areas (topics, languages)
        - Activity levels (maintenance, growth)
        - Open source engagement

    Args:
        df: Repositories DataFrame from load_repos_df()

    Returns:
        Dictionary containing:
            - total_repos: Total number of repositories
            - total_stars: Sum of stars across all repos
            - total_forks: Sum of forks across all repos
            - avg_stars: Average stars per repo
            - most_starred: Details of most starred repo
            - active_repos: Count of non-archived repos
            - fork_ratio: Percentage of forks vs originals
            - topic_frequency: Most common topics
            - license_distribution: License usage counts
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to analyze_repositories")
        return _empty_repo_analysis()

    logger.info(f"Analyzing {len(df)} repositories")

    result: Dict[str, Any] = {}

    # =========================================================================
    # Basic counts and aggregates
    # =========================================================================
    # WHY: Foundation metrics for understanding portfolio size

    result["total_repos"] = len(df)
    result["total_stars"] = int(df["stargazers_count"].sum()) if "stargazers_count" in df.columns else 0
    result["total_forks"] = int(df["forks_count"].sum()) if "forks_count" in df.columns else 0
    result["total_issues"] = int(df["open_issues_count"].sum()) if "open_issues_count" in df.columns else 0

    # Average stars per repo
    # WHY: Shows average impact; high average with few repos = focused quality
    result["avg_stars"] = round(df["stargazers_count"].mean(), 1) if "stargazers_count" in df.columns else 0
    result["median_stars"] = df["stargazers_count"].median() if "stargazers_count" in df.columns else 0

    # =========================================================================
    # Activity analysis
    # =========================================================================
    # WHY: Distinguishes active maintainers from project starters

    # Count active (non-archived) repos
    if "is_archived" in df.columns:
        result["active_repos"] = len(df[~df["is_archived"]])
        result["archived_repos"] = len(df[df["is_archived"]])
        result["archive_ratio"] = round(
            (result["archived_repos"] / result["total_repos"] * 100), 1
        ) if result["total_repos"] > 0 else 0
    else:
        result["active_repos"] = result["total_repos"]
        result["archived_repos"] = 0
        result["archive_ratio"] = 0

    # Fork ratio
    # WHY: High fork ratio might indicate learning/tinkering vs creation
    if "is_fork" in df.columns:
        fork_count = df["is_fork"].sum()
        result["fork_count"] = int(fork_count)
        result["fork_ratio"] = round((fork_count / len(df) * 100), 1) if len(df) > 0 else 0
    else:
        result["fork_count"] = 0
        result["fork_ratio"] = 0

    # =========================================================================
    # Top repositories by stars
    # =========================================================================
    # WHY: Identifies flagship projects that define developer's impact

    if "stargazers_count" in df.columns:
        # Sort by stars and get top repos
        top_repos = df.nlargest(5, "stargazers_count")

        result["most_starred"] = {
            "name": top_repos.iloc[0]["name"] if len(top_repos) > 0 else None,
            "stars": int(top_repos.iloc[0]["stargazers_count"]) if len(top_repos) > 0 else 0,
            "description": top_repos.iloc[0].get("description"),
            "language": top_repos.iloc[0].get("primary_language"),
        }

        result["top_repositories"] = [
            {
                "name": row["name"],
                "stars": int(row["stargazers_count"]),
                "forks": int(row["forks_count"]) if "forks_count" in row else 0,
                "language": row.get("primary_language"),
                "description": row.get("description"),
            }
            for _, row in top_repos.iterrows()
        ]

    # =========================================================================
    # Topic analysis
    # =========================================================================
    # WHY: Topics show interests and expertise areas beyond language

    if "topics" in df.columns:
        # Flatten topics list
        # BUSINESS LOGIC: Topics are stored as lists; need to flatten for counting
        all_topics = []
        for topics in df["topics"].dropna():
            if isinstance(topics, list):
                all_topics.extend(topics)
            elif isinstance(topics, str):
                # Handle JSON string case
                import json
                try:
                    all_topics.extend(json.loads(topics))
                except (json.JSONDecodeError, TypeError):
                    pass

        # Count topic frequencies
        topic_counts = Counter(all_topics)
        result["topic_frequency"] = [
            {"topic": topic, "count": count}
            for topic, count in topic_counts.most_common(10)
        ]
        result["unique_topics"] = len(topic_counts)
    else:
        result["topic_frequency"] = []
        result["unique_topics"] = 0

    # =========================================================================
    # License distribution
    # =========================================================================
    # WHY: Shows open source licensing preferences and compliance awareness

    if "license" in df.columns:
        license_counts = df["license"].value_counts().to_dict()
        result["license_distribution"] = license_counts

        # Calculate open source ratio
        # BUSINESS LOGIC: Common open source licenses
        open_source_licenses = [
            "MIT", "Apache-2.0", "GPL-3.0", "GPL-2.0", "BSD-3-Clause",
            "BSD-2-Clause", "LGPL-3.0", "MPL-2.0", "ISC", "Unlicense",
        ]
        open_source_count = sum(
            license_counts.get(lic, 0) for lic in open_source_licenses
        )
        result["open_source_ratio"] = round(
            (open_source_count / len(df) * 100), 1
        ) if len(df) > 0 else 0
    else:
        result["license_distribution"] = {}
        result["open_source_ratio"] = 0

    # =========================================================================
    # Language distribution across repos
    # =========================================================================
    # WHY: Shows technology breadth across different projects

    if "primary_language" in df.columns:
        lang_counts = df["primary_language"].value_counts().to_dict()
        result["language_distribution"] = lang_counts
        result["most_common_language"] = df["primary_language"].mode().iloc[0] if len(df) > 0 else None

    # =========================================================================
    # Repository size analysis
    # =========================================================================
    # WHY: Size indicates project complexity and effort investment

    if "size" in df.columns:
        result["total_size_kb"] = int(df["size"].sum())
        result["avg_size_kb"] = round(df["size"].mean(), 1)
        result["largest_repo"] = {
            "name": df.loc[df["size"].idxmax(), "name"] if len(df) > 0 else None,
            "size_kb": int(df["size"].max()),
        }

    logger.info(
        f"Repository analysis complete: {result['total_repos']} repos, "
        f"{result['total_stars']} stars, {result['active_repos']} active"
    )

    return result


def _empty_repo_analysis() -> Dict[str, Any]:
    """Return empty repository analysis structure."""
    return {
        "total_repos": 0,
        "total_stars": 0,
        "total_forks": 0,
        "avg_stars": 0,
        "active_repos": 0,
        "fork_ratio": 0,
        "top_repositories": [],
        "topic_frequency": [],
    }


# =============================================================================
# Commit Message Analysis
# =============================================================================

def analyze_commit_messages(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze commit message patterns and quality.

    WHY: Commit messages reveal:
        - Communication skills
        - Development process maturity
        - Attention to detail
        - Use of conventions (conventional commits)

    Well-written commit messages are a hallmark of professional development.

    Args:
        df: Commits DataFrame (cleaned) with message and derived columns

    Returns:
        Dictionary containing:
            - total_commits: Total number of commits
            - avg_message_length: Average characters per message
            - conventional_commit_ratio: Percentage using conventional format
            - single_word_ratio: Percentage with single-word messages
            - common_first_words: Most common starting words
            - message_size_distribution: Breakdown by size category
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to analyze_commit_messages")
        return _empty_commit_analysis()

    logger.info(f"Analyzing {len(df)} commit messages")

    result: Dict[str, Any] = {}

    # =========================================================================
    # Basic message statistics
    # =========================================================================
    # WHY: Basic stats give overview of message habits

    result["total_commits"] = len(df)

    if "message_length" in df.columns:
        result["avg_message_length"] = round(df["message_length"].mean(), 1)
        result["median_message_length"] = float(df["message_length"].median())
        result["min_message_length"] = int(df["message_length"].min())
        result["max_message_length"] = int(df["message_length"].max())

        # Message length distribution
        # WHY: Shows if messages tend to be too short or too long
        result["message_length_std"] = round(df["message_length"].std(), 1)

    # =========================================================================
    # Conventional commits analysis
    # =========================================================================
    # WHY: Conventional commits indicate process maturity
    # Format: type(scope): description (e.g., "feat(auth): add login")

    if "is_conventional" in df.columns:
        conventional_count = df["is_conventional"].sum()
        result["conventional_commits"] = int(conventional_count)
        result["conventional_commit_ratio"] = round(
            (conventional_count / len(df) * 100), 1
        ) if len(df) > 0 else 0

        # Break down by type if possible
        # BUSINESS LOGIC: Different commit types indicate different work patterns
        if "message" in df.columns:
            conventional_df = df[df["is_conventional"]].copy()
            if not conventional_df.empty:
                # Extract commit type
                conventional_df["commit_type"] = conventional_df["message"].str.extract(
                    r"^(feat|fix|docs|style|refactor|test|chore|build|ci|perf|revert)",
                    flags=re.IGNORECASE if hasattr(re, 'IGNORECASE') else 0
                )
                type_counts = conventional_df["commit_type"].value_counts().to_dict()
                result["conventional_types"] = type_counts
    else:
        result["conventional_commits"] = 0
        result["conventional_commit_ratio"] = 0

    # =========================================================================
    # Message quality indicators
    # =========================================================================
    # WHY: Quality indicators help assess professionalism

    if "message" in df.columns:
        # Single-word commits (low quality indicator)
        # BUSINESS LOGIC: Single-word messages usually lack context
        single_word_mask = df["message"].str.split().str.len() == 1
        result["single_word_commits"] = int(single_word_mask.sum())
        result["single_word_ratio"] = round(
            (single_word_mask.sum() / len(df) * 100), 1
        ) if len(df) > 0 else 0

        # Empty or whitespace-only messages
        # BUSINESS LOGIC: Empty messages are a red flag for code quality process
        empty_mask = df["message"].str.strip() == ""
        result["empty_commits"] = int(empty_mask.sum())
        result["empty_commit_ratio"] = round(
            (empty_mask.sum() / len(df) * 100), 1
        ) if len(df) > 0 else 0

    # =========================================================================
    # Common first words analysis
    # =========================================================================
    # WHY: First words reveal commit message patterns and style

    if "message" in df.columns:
        # Extract first word from each message
        # BUSINESS LOGIC: First word sets context for the entire message
        first_words = (
            df["message"]
            .str.strip()
            .str.split()
            .str[0]
            .str.lower()
            .dropna()
        )

        # Filter out very short words and common fillers
        # WHY: These add noise to the analysis
        meaningful_words = first_words[
            (first_words.str.len() >= 3) &
            (~first_words.isin(["the", "and", "for", "with", "this", "that"]))
        ]

        if not meaningful_words.empty:
            word_counts = meaningful_words.value_counts()
            result["common_first_words"] = [
                {"word": word, "count": int(count)}
                for word, count in word_counts.head(10).items()
            ]

    # =========================================================================
    # Message size distribution
    # =========================================================================
    # WHY: Shows if developer tends toward brief or detailed messages

    if "commit_size" in df.columns:
        size_dist = df["commit_size"].value_counts().to_dict()
        result["message_size_distribution"] = size_dist

    # =========================================================================
    # Commit timing quality
    # =========================================================================
    # WHY: Late-night commits might indicate crunch or passion

    if "time_of_day" in df.columns:
        time_dist = df["time_of_day"].value_counts().to_dict()
        result["time_distribution"] = time_dist

    # Calculate commit quality score
    # WHY: Single metric combining multiple quality indicators
    quality_score = _calculate_message_quality_score(result)
    result["quality_score"] = quality_score

    logger.info(
        f"Commit message analysis complete: {result['total_commits']} commits, "
        f"{result.get('conventional_commit_ratio', 0)}% conventional, "
        f"quality={quality_score}"
    )

    return result


def _empty_commit_analysis() -> Dict[str, Any]:
    """Return empty commit analysis structure."""
    return {
        "total_commits": 0,
        "avg_message_length": 0,
        "conventional_commit_ratio": 0,
        "single_word_ratio": 0,
        "quality_score": 0,
    }


def _calculate_message_quality_score(analysis: Dict[str, Any]) -> float:
    """
    Calculate overall commit message quality score.

    WHY: Provides a single metric for comparing developers' commit hygiene.

    Factors:
        - Conventional commit usage (positive)
        - Low single-word ratio (positive)
        - Appropriate message length (positive)
        - Low empty commit ratio (positive)

    Args:
        analysis: Commit message analysis results

    Returns:
        Quality score from 0 to 100
    """
    score = 50.0  # Start at neutral

    # Conventional commits bonus (up to +20)
    # WHY: Conventional commits are a best practice
    conventional_ratio = analysis.get("conventional_commit_ratio", 0)
    score += min(20, conventional_ratio * 0.4)

    # Single-word penalty (up to -20)
    # WHY: Single-word messages lack context
    single_word_ratio = analysis.get("single_word_ratio", 0)
    score -= min(20, single_word_ratio * 0.4)

    # Empty message penalty (up to -30)
    # WHY: Empty messages are unacceptable
    empty_ratio = analysis.get("empty_commit_ratio", 0)
    score -= min(30, empty_ratio * 0.6)

    # Message length bonus (up to +10)
    # WHY: Neither too short nor too long is ideal
    avg_length = analysis.get("avg_message_length", 0)
    if 30 <= avg_length <= 72:
        score += 10  # Ideal range
    elif 20 <= avg_length <= 100:
        score += 5   # Acceptable range

    return round(max(0, min(100, score)), 1)


# =============================================================================
# Developer Profile Generation
# =============================================================================

def generate_developer_profile(
    metrics: Dict[str, Any],
    languages: Dict[str, Any],
    repos: Dict[str, Any],
    commits: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a comprehensive developer profile with labels.

    WHY: The profile synthesizes all analyses into actionable insights:
        - Labels for quick categorization
        - Scores for quantitative comparison
        - Strengths and areas for improvement
        - Recommended roles and technologies

    This is the main output used by LLM for generating insights.

    Args:
        metrics: Advanced metrics from compute_advanced_metrics()
        languages: Language analysis from analyze_languages()
        repos: Repository analysis from analyze_repositories()
        commits: Commit message analysis from analyze_commit_messages()

    Returns:
        Dictionary containing:
            - labels: Classification labels (type, activity, experience, work_style)
            - scores: Numerical scores (consistency, quality, impact, diversity)
            - strengths: Identified strengths
            - improvements: Areas for improvement
            - summary: Human-readable summary
    """
    logger.info("Generating developer profile")

    profile: Dict[str, Any] = {
        "labels": {},
        "scores": {},
        "strengths": [],
        "improvements": [],
        "languages": {},
        "repositories": {},
        "commits": {},
    }

    # =========================================================================
    # Generate Classification Labels
    # =========================================================================
    # WHY: Labels provide quick categorization for comparison

    # Developer Type
    # BUSINESS LOGIC: Developer type helps match to roles
    profile["labels"]["developer_type"] = _classify_developer_type(
        metrics, languages, repos
    )

    # Activity Level
    # BUSINESS LOGIC: Activity level shows engagement
    profile["labels"]["activity_level"] = _classify_activity_level(metrics)

    # Experience Level
    # BUSINESS LOGIC: Experience level guides appropriate opportunities
    profile["labels"]["experience_level"] = _classify_experience_level(
        metrics, repos
    )

    # Work Style
    # BUSINESS LOGIC: Work style affects team fit
    profile["labels"]["work_style"] = metrics.get("work_style", "balanced")

    # =========================================================================
    # Calculate Composite Scores
    # =========================================================================
    # WHY: Scores enable quantitative comparison between developers

    # Consistency Score (from metrics)
    profile["scores"]["consistency"] = metrics.get("consistency_score", 0)

    # Quality Score (from commit analysis)
    profile["scores"]["quality"] = commits.get("quality_score", 0)

    # Impact Score (from stars/forks)
    profile["scores"]["impact"] = _calculate_impact_score(repos)

    # Diversity Score (from language analysis)
    profile["scores"]["diversity"] = _normalize_diversity_score(
        languages.get("diversity_score", 0)
    )

    # Overall Score
    # WHY: Single metric for overall assessment
    scores = profile["scores"]
    profile["scores"]["overall"] = round(
        (scores["consistency"] + scores["quality"] + scores["impact"] + scores["diversity"]) / 4,
        1
    )

    # =========================================================================
    # Extract Key Information for LLM
    # =========================================================================
    # WHY: Structured data helps LLM generate relevant insights

    # Language info
    profile["languages"] = {
        "primary": languages.get("primary_language"),
        "count": languages.get("language_count", 0),
        "top_languages": languages.get("top_languages", [])[:5],
        "diversity_score": languages.get("diversity_score", 0),
        "primary_category": languages.get("primary_category"),
    }

    # Repository info
    profile["repositories"] = {
        "total": repos.get("total_repos", 0),
        "total_stars": repos.get("total_stars", 0),
        "active": repos.get("active_repos", 0),
        "top_repos": repos.get("top_repositories", [])[:3],
    }

    # Commit info
    profile["commits"] = {
        "total": metrics.get("total_commits", 0),
        "active_days": metrics.get("active_days", 0),
        "longest_streak": metrics.get("longest_streak", 0),
        "current_streak": metrics.get("current_streak", 0),
        "peak_hour": metrics.get("peak_hour"),
        "peak_day": metrics.get("peak_day"),
        "preferred_time": metrics.get("preferred_time"),
        "conventional_ratio": commits.get("conventional_commit_ratio", 0),
    }

    # Burnout info
    profile["wellbeing"] = {
        "burnout_periods": len(metrics.get("burnout_periods", [])),
        "weekend_ratio": metrics.get("weekend_commit_ratio", 0),
        "productivity_trend": metrics.get("productivity_trend", "unknown"),
    }

    # =========================================================================
    # Identify Strengths and Improvements
    # =========================================================================
    # WHY: Actionable feedback for professional development

    # Strengths
    # BUSINESS LOGIC: Identify what developer does well
    if profile["scores"]["consistency"] >= 70:
        profile["strengths"].append("Consistent contribution pattern")

    if profile["scores"]["quality"] >= 70:
        profile["strengths"].append("High-quality commit messages")

    if profile["scores"]["impact"] >= 70:
        profile["strengths"].append("High-impact open source contributions")

    if commits.get("conventional_commit_ratio", 0) >= 50:
        profile["strengths"].append("Uses conventional commit format")

    if metrics.get("longest_streak", 0) >= 30:
        profile["strengths"].append("Strong dedication (30+ day streak)")

    if languages.get("diversity_score", 0) >= 1.5:
        profile["strengths"].append("Broad technology expertise")

    # Improvements
    # BUSINESS LOGIC: Identify areas for growth
    if profile["scores"]["consistency"] < 50:
        profile["improvements"].append("Could improve contribution consistency")

    if profile["scores"]["quality"] < 50:
        profile["improvements"].append("Commit messages could be more descriptive")

    if commits.get("single_word_ratio", 0) >= 20:
        profile["improvements"].append("Avoid single-word commit messages")

    if len(metrics.get("burnout_periods", [])) >= 3:
        profile["improvements"].append("Consider maintaining sustainable pace")

    if repos.get("total_stars", 0) == 0 and repos.get("total_repos", 0) > 5:
        profile["improvements"].append("Focus on quality over quantity in projects")

    # =========================================================================
    # Generate Summary
    # =========================================================================
    # WHY: Quick overview for human readers

    labels = profile["labels"]
    scores = profile["scores"]

    summary = (
        f"A {labels.get('activity_level', 'unknown').lower()} "
        f"{labels.get('developer_type', 'developer').lower()} "
        f"with {labels.get('experience_level', 'unknown').lower()} experience. "
        f"Primary language: {languages.get('primary_language', 'Unknown')}. "
        f"Overall score: {scores.get('overall', 0)}/100."
    )
    profile["summary"] = summary

    logger.info(
        f"Profile generated: {labels.get('developer_type')} "
        f"({labels.get('activity_level')}, {labels.get('experience_level')})"
    )

    return profile


def _classify_developer_type(
    metrics: Dict[str, Any],
    languages: Dict[str, Any],
    repos: Dict[str, Any]
) -> str:
    """
    Classify developer type based on patterns.

    WHY: Developer types help understand work focus and match to roles.

    Types:
        - Full-Stack: Frontend + backend languages
        - Backend: Server-side focused
        - Frontend: Client-side focused
        - Data Scientist: Data-focused languages
        - Systems: Low-level languages
        - Open Source Maintainer: Many repos, high stars
        - Polyglot: High language diversity
    """
    diversity = languages.get("diversity_score", 0)
    primary_category = languages.get("primary_category", "")
    total_repos = repos.get("total_repos", 0)
    total_stars = repos.get("total_stars", 0)

    # Check for maintainer pattern
    # BUSINESS LOGIC: Maintainers have many repos with significant stars
    if total_repos >= 20 and total_stars >= 100:
        return "Open Source Maintainer"

    # Check for polyglot
    # BUSINESS LOGIC: High diversity indicates broad skill set
    if diversity >= 2.0:
        return "Polyglot"

    # Check by category
    category_breakdown = languages.get("category_breakdown", {})

    frontend_pct = category_breakdown.get("frontend", 0)
    backend_pct = category_breakdown.get("backend", 0)
    data_pct = category_breakdown.get("data", 0)
    systems_pct = category_breakdown.get("systems", 0)

    # Full-stack: significant frontend and backend
    # BUSINESS LOGIC: Full-stack developers have balance
    if frontend_pct >= 20 and backend_pct >= 20:
        return "Full-Stack Developer"

    # Data scientist
    if data_pct >= 40:
        return "Data Scientist"

    # Systems developer
    if systems_pct >= 40:
        return "Systems Developer"

    # Frontend
    if frontend_pct >= 50:
        return "Frontend Developer"

    # Backend
    if backend_pct >= 50:
        return "Backend Developer"

    return "Software Developer"


def _classify_activity_level(metrics: Dict[str, Any]) -> str:
    """
    Classify activity level based on commit patterns.

    WHY: Activity level shows engagement and availability.
    """
    total_commits = metrics.get("total_commits", 0)
    active_days = metrics.get("active_days", 0)
    consistency = metrics.get("consistency_score", 0)

    # Calculate commits per active day
    commits_per_day = total_commits / active_days if active_days > 0 else 0

    # Classify based on combined metrics
    # BUSINESS LOGIC: High commits + high consistency = very active
    if consistency >= 70 and commits_per_day >= 3:
        return "Extremely High"
    elif consistency >= 60 and commits_per_day >= 2:
        return "Very High"
    elif consistency >= 50 or commits_per_day >= 1.5:
        return "High"
    elif consistency >= 30 or commits_per_day >= 1:
        return "Moderate"
    else:
        return "Low"


def _classify_experience_level(
    metrics: Dict[str, Any],
    repos: Dict[str, Any]
) -> str:
    """
    Classify experience level based on portfolio indicators.

    WHY: Experience level helps match to appropriate opportunities.
    """
    total_commits = metrics.get("total_commits", 0)
    total_repos = repos.get("total_repos", 0)
    total_stars = repos.get("total_stars", 0)
    active_days = metrics.get("active_days", 0)

    # Calculate a composite experience score
    # BUSINESS LOGIC: Combine multiple signals for better classification
    exp_score = (
        min(total_commits / 100, 10) +  # Up to 10 points for commits
        min(total_repos / 5, 10) +       # Up to 10 points for repos
        min(np.log10(total_stars + 1) * 2, 10) +  # Up to 10 points for stars
        min(active_days / 100, 10)       # Up to 10 points for activity span
    )

    if exp_score >= 30:
        return "Veteran (10+ years)"
    elif exp_score >= 20:
        return "Senior (5-10 years)"
    elif exp_score >= 10:
        return "Mid-Level (2-5 years)"
    else:
        return "Junior (0-2 years)"


def _calculate_impact_score(repos: Dict[str, Any]) -> float:
    """
    Calculate impact score based on stars and forks.

    WHY: Impact score shows real-world influence of contributions.
    Uses logarithmic scale to avoid domination by outliers.
    """
    total_stars = repos.get("total_stars", 0)
    total_forks = repos.get("total_forks", 0)

    # Logarithmic scale for stars and forks
    # BUSINESS LOGIC: 100 stars is meaningful, 10000 is amazing
    # Log scale compresses this range appropriately
    star_score = np.log10(total_stars + 1) * 25  # Max ~100 at 10,000 stars
    fork_score = np.log10(total_forks + 1) * 30  # Max ~120 at 10,000 forks

    # Combine and cap at 100
    impact = (star_score + fork_score) / 2

    return round(min(100, impact), 1)


def _normalize_diversity_score(raw_score: float) -> float:
    """
    Normalize Shannon entropy diversity score to 0-100 scale.

    WHY: Raw entropy is hard to interpret; normalized score is intuitive.

    Raw score range:
        - 0: Single language (no diversity)
        - ~3.3: 10 languages equally used (maximum typical diversity)
    """
    # Normalize assuming max practical entropy of ~3.5
    normalized = (raw_score / 3.5) * 100
    return round(min(100, normalized), 1)


# =============================================================================
# Export Public Interface
# =============================================================================

__all__ = [
    "analyze_languages",
    "analyze_repositories",
    "analyze_commit_messages",
    "generate_developer_profile",
]


# Import re for regex operations (needed in analyze_commit_messages)
import re
