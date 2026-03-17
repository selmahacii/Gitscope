#!/usr/bin/env python3
"""
GitScope - GitHub Profile Analyzer.

See the full scope of any GitHub developer.

This module provides the command-line interface and main pipeline:
    1. Parse CLI arguments
    2. Collect data from GitHub API
    3. Store in SQLite database
    4. Transform and analyze data
    5. Generate developer profile
    6. Generate AI insights (if configured)
    7. Display results in console

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates CLI design with argparse
    - Shows end-to-end pipeline implementation
    - Implements proper error handling and user feedback
    - Uses Rich for professional terminal output
    - Provides multiple output formats (console, JSON)

Usage:
    python main.py <username>
    python main.py octocat --force-refresh
    python main.py octocat --no-ai --output results.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

from src.config import settings, configure_logging
from src.collector import GitHubCollector, GitHubAPIError, UserNotFoundError
from src.storage import save_all_data, load_commits_df, load_repos_df, load_languages_df
from src.transformer import clean_commits, aggregate_by_time, compute_advanced_metrics
from src.analytics import (
    analyze_languages,
    analyze_repositories,
    analyze_commit_messages,
    generate_developer_profile,
)
from src.insights import generate_insights, generate_fun_facts, generate_basic_insights


# Initialize Rich console
console = Console()


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_analysis(
    username: str,
    token: Optional[str] = None,
    force_refresh: bool = False,
    generate_ai_insights: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete GitHub profile analysis pipeline.

    WHY: Provides a single entry point for programmatic analysis.
    This function is also used by tests and other modules.

    Pipeline steps:
        1. Initialize GitHubCollector
        2. Collect all data (with cache check)
        3. Save to SQLite database
        4. Load DataFrames from database
        5. Run all transformations
        6. Compute all analytics
        7. Generate developer profile
        8. Generate GLM insights (if enabled)

    Args:
        username: GitHub username to analyze
        token: Optional GitHub token for higher rate limits
        force_refresh: Skip cache and fetch fresh data
        generate_ai_insights: Whether to generate GLM insights
        verbose: Enable verbose logging

    Returns:
        Dictionary containing all analysis results:
            - raw_data: Raw data from GitHub API
            - profile: Developer profile dictionary
            - metrics: Computed metrics
            - aggregations: Time-series aggregations
            - lang_analysis: Language analysis results
            - repo_analysis: Repository analysis results
            - commit_analysis: Commit message analysis
            - insights: GLM-generated insights
            - fun_facts: Fun facts from metrics
            - execution_time: Total execution time in seconds
            - timestamp: ISO timestamp of analysis
    """
    start_time = datetime.now(timezone.utc)

    # Validate input
    if not username or not username.strip():
        raise ValueError("Username cannot be empty")

    username = username.strip()

    # Initialize result dictionary
    result: Dict[str, Any] = {
        "username": username,
        "raw_data": None,
        "profile": None,
        "metrics": None,
        "aggregations": None,
        "lang_analysis": None,
        "repo_analysis": None,
        "commit_analysis": None,
        "insights": None,
        "fun_facts": None,
        "execution_time": 0,
        "timestamp": start_time.isoformat(),
        "errors": [],
    }

    # Print header
    console.print(Panel(
        f"[bold blue]GitScope[/bold blue]\n"
        f"[dim]See the full scope of any GitHub developer[/dim]\n"
        f"Analyzing: [cyan]{username}[/cyan]",
        expand=False,
    ))

    # =========================================================================
    # Step 1: Initialize GitHubCollector
    # =========================================================================
    console.print("\n[bold]Step 1: Initialize Collector[/bold]")

    try:
        collector = GitHubCollector(
            username=username,
            token=token or settings.github_token,
        )
        console.print(f"[dim]  ✓ Collector initialized for {username}[/dim]")
    except Exception as e:
        console.print(f"[red]  ✗ Failed to initialize collector: {e}[/red]")
        result["errors"].append(str(e))
        return result

    # =========================================================================
    # Step 2: Collect all data from GitHub API
    # =========================================================================
    console.print("\n[bold]Step 2: Collect Data from GitHub API[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching data...", total=None)

        try:
            if force_refresh:
                # Clear cache to force fresh data
                cleared = collector.cache.clear()
                console.print(f"[dim]  Cleared {cleared} cache entries[/dim]")

            raw_data = collector.collect_all()
            result["raw_data"] = raw_data

        except UserNotFoundError:
            console.print(f"[red]  ✗ User '{username}' not found on GitHub[/red]")
            result["errors"].append(f"User not found: {username}")
            return result

        except GitHubAPIError as e:
            console.print(f"[red]  ✗ GitHub API error: {e.message}[/red]")
            result["errors"].append(e.message)
            return result

        except Exception as e:
            console.print(f"[red]  ✗ Unexpected error: {e}[/red]")
            result["errors"].append(str(e))
            logger.exception("Data collection failed")
            return result

    # Report collection results
    repos_count = len(raw_data.get("repositories", []))
    commits_count = sum(len(c) for c in raw_data.get("commits", {}).values())
    console.print(f"[green]  ✓ Collected data for {username}[/green]")
    console.print(f"[dim]    - {repos_count} repositories[/dim]")
    console.print(f"[dim]    - {commits_count} commits[/dim]")

    # =========================================================================
    # Step 3: Save to SQLite database
    # =========================================================================
    console.print("\n[bold]Step 3: Save to Database[/bold]")

    try:
        counts = save_all_data(raw_data, settings.db_path)
        console.print(f"[green]  ✓ Saved to {settings.db_path}[/green]")
        console.print(f"[dim]    - {counts['commits']} commits[/dim]")
        console.print(f"[dim]    - {counts['repositories']} repositories[/dim]")
    except Exception as e:
        console.print(f"[red]  ✗ Failed to save to database: {e}[/red]")
        result["errors"].append(f"Database error: {e}")
        logger.exception("Database save failed")
        return result

    # =========================================================================
    # Step 4: Load DataFrames from database
    # =========================================================================
    console.print("\n[bold]Step 4: Load DataFrames[/bold]")

    try:
        commits_df = load_commits_df(settings.db_path, username)
        repos_df = load_repos_df(settings.db_path, username)
        languages_df = load_languages_df(settings.db_path, username)

        console.print(f"[green]  ✓ Loaded DataFrames[/green]")
        console.print(f"[dim]    - {len(commits_df)} commits[/dim]")
        console.print(f"[dim]    - {len(repos_df)} repositories[/dim]")
        console.print(f"[dim]    - {len(languages_df)} language entries[/dim]")

    except Exception as e:
        console.print(f"[red]  ✗ Failed to load DataFrames: {e}[/red]")
        result["errors"].append(f"DataFrame load error: {e}")
        logger.exception("DataFrame loading failed")
        return result

    # =========================================================================
    # Step 5: Run all transformations
    # =========================================================================
    console.print("\n[bold]Step 5: Transform Data[/bold]")

    try:
        # Clean commits
        clean_df = clean_commits(commits_df)
        console.print(f"[dim]    - Cleaned {len(clean_df)} commits[/dim]")

        # Aggregate by time
        aggregations = aggregate_by_time(clean_df)
        console.print(f"[dim]    - Created {len(aggregations)} aggregation types[/dim]")

        # Compute metrics
        metrics = compute_advanced_metrics(clean_df)
        console.print(f"[dim]    - Computed {len(metrics)} metrics[/dim]")

        result["aggregations"] = {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in aggregations.items()
        }
        result["metrics"] = metrics

    except Exception as e:
        console.print(f"[red]  ✗ Transformation failed: {e}[/red]")
        result["errors"].append(f"Transformation error: {e}")
        logger.exception("Transformation failed")
        return result

    console.print(f"[green]  ✓ Transformations complete[/green]")

    # =========================================================================
    # Step 6: Compute all analytics
    # =========================================================================
    console.print("\n[bold]Step 6: Compute Analytics[/bold]")

    try:
        # Language analysis
        lang_analysis = analyze_languages(languages_df)
        result["lang_analysis"] = lang_analysis
        console.print(f"[dim]    - Language analysis: {lang_analysis.get('language_count', 0)} languages[/dim]")

        # Repository analysis
        repo_analysis = analyze_repositories(repos_df)
        result["repo_analysis"] = repo_analysis
        console.print(f"[dim]    - Repository analysis: {repo_analysis.get('total_repos', 0)} repos[/dim]")

        # Commit message analysis
        commit_analysis = analyze_commit_messages(clean_df)
        result["commit_analysis"] = commit_analysis
        console.print(f"[dim]    - Commit analysis: avg {commit_analysis.get('avg_message_length', 0):.0f} chars[/dim]")

    except Exception as e:
        console.print(f"[red]  ✗ Analytics computation failed: {e}[/red]")
        result["errors"].append(f"Analytics error: {e}")
        logger.exception("Analytics failed")
        return result

    console.print(f"[green]  ✓ Analytics complete[/green]")

    # =========================================================================
    # Step 7: Generate developer profile
    # =========================================================================
    console.print("\n[bold]Step 7: Generate Developer Profile[/bold]")

    try:
        profile = generate_developer_profile(
            metrics=metrics,
            languages=lang_analysis,
            repos=repo_analysis,
            commits=commit_analysis,
        )
        profile["username"] = username
        result["profile"] = profile

        labels = profile.get("labels", {})
        console.print(f"[green]  ✓ Profile generated[/green]")
        console.print(f"[dim]    - Type: {labels.get('developer_type', 'Unknown')}[/dim]")
        console.print(f"[dim]    - Experience: {labels.get('experience_level', 'Unknown')}[/dim]")

    except Exception as e:
        console.print(f"[red]  ✗ Profile generation failed: {e}[/red]")
        result["errors"].append(f"Profile error: {e}")
        logger.exception("Profile generation failed")
        return result

    # =========================================================================
    # Step 8: Generate AI insights
    # =========================================================================
    if generate_ai_insights:
        console.print("\n[bold]Step 8: Generate AI Insights[/bold]")

        try:
            insights = generate_insights(profile)
            result["insights"] = insights

            # Generate fun facts
            fun_facts = generate_fun_facts(metrics)
            result["fun_facts"] = fun_facts

            if insights.get("error"):
                console.print(f"[yellow]  ⚠ AI insights unavailable: {insights['error']}[/yellow]")
            else:
                console.print(f"[green]  ✓ AI insights generated[/green]")
                console.print(f"[dim]    - Model: {insights.get('model', 'unknown')}[/dim]")
                console.print(f"[dim]    - Tokens: {insights.get('tokens_used', 0)}[/dim]")

        except Exception as e:
            console.print(f"[yellow]  ⚠ AI insights generation failed: {e}[/yellow]")
            result["errors"].append(f"AI insights error: {e}")
            logger.exception("AI insights failed")
    else:
        console.print("\n[dim]Step 8: Skipping AI insights (disabled)[/dim]")

    # =========================================================================
    # Calculate execution time
    # =========================================================================
    end_time = datetime.now(timezone.utc)
    execution_time = (end_time - start_time).total_seconds()
    result["execution_time"] = round(execution_time, 2)

    # =========================================================================
    # Print summary
    # =========================================================================
    print_summary(result)

    return result


# =============================================================================
# Summary Printer
# =============================================================================

def print_summary(result: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the analysis to console.

    WHY: Provides a quick overview of results in the terminal,
    suitable for CLI usage and quick checks.

    Args:
        result: Analysis result dictionary
    """
    console.print("\n")
    console.print("=" * 70)
    console.print("[bold]ANALYSIS SUMMARY[/bold]")
    console.print("=" * 70)

    profile = result.get("profile", {})
    labels = profile.get("labels", {})
    languages = profile.get("languages", {})
    metrics = result.get("metrics", {})
    insights = result.get("insights", {})
    scores = profile.get("scores", {})

    # -------------------------------------------------------------------------
    # Basic info line
    # -------------------------------------------------------------------------
    username = result.get("username", "Unknown")
    dev_type = labels.get("developer_type", "Unknown")
    experience = labels.get("experience_level", "Unknown")
    primary_lang = languages.get("primary", "Unknown")

    console.print(f"\n[bold cyan]@{username}[/bold cyan]")
    console.print(
        f"  Type: {dev_type}  |  Experience: {experience}  |  Top Language: {primary_lang}"
    )

    # -------------------------------------------------------------------------
    # Score visualization
    # -------------------------------------------------------------------------
    console.print(f"\n[bold]Overall Score: {scores.get('overall', 0):.0f}/100[/bold]")

    # Create ASCII progress bar
    bar_width = 30
    overall = scores.get("overall", 0)
    filled = int(overall / 100 * bar_width)
    empty = bar_width - filled

    # Color based on score
    if overall >= 70:
        bar_color = "green"
    elif overall >= 40:
        bar_color = "yellow"
    else:
        bar_color = "red"

    bar = "█" * filled + "░" * empty
    console.print(f"  [{bar_color}]{bar}[/{bar_color}] {overall:.0f}%")

    # -------------------------------------------------------------------------
    # Scores table
    # -------------------------------------------------------------------------
    console.print("\n[bold]Score Breakdown:[/bold]")

    score_table = Table(show_header=False, box=None, padding=(0, 2))
    score_table.add_column("Metric", style="cyan")
    score_table.add_column("Value", style="white")

    score_table.add_row("Consistency", f"{scores.get('consistency', 0):.1f}/100")
    score_table.add_row("Quality", f"{scores.get('quality', 0):.1f}/100")
    score_table.add_row("Impact", f"{scores.get('impact', 0):.1f}/100")
    score_table.add_row("Diversity", f"{scores.get('diversity', 0):.1f}/100")

    console.print(score_table)

    # -------------------------------------------------------------------------
    # Key metrics table
    # -------------------------------------------------------------------------
    console.print("\n[bold]Key Metrics:[/bold]")

    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="white")

    metrics_table.add_row("Total Commits", f"{metrics.get('total_commits', 0):,}")
    metrics_table.add_row("Total Repositories", str(metrics.get("total_repos", 0)))
    metrics_table.add_row("Active Days", f"{metrics.get('active_days', 0):,}")
    metrics_table.add_row("Longest Streak", f"{metrics.get('longest_streak', 0)} days")
    metrics_table.add_row("Current Streak", f"{metrics.get('current_streak', 0)} days")
    metrics_table.add_row("Peak Hour", f"{metrics.get('peak_hour', 'N/A')}:00")
    metrics_table.add_row("Peak Day", _day_name(metrics.get("peak_day")))
    metrics_table.add_row("Preferred Time", str(metrics.get("preferred_time", "Unknown")).title())
    metrics_table.add_row("Productivity Trend", str(metrics.get("productivity_trend", "Unknown")).title())
    metrics_table.add_row("Burnout Periods", str(len(metrics.get("burnout_periods", []))))

    console.print(metrics_table)

    # -------------------------------------------------------------------------
    # Language breakdown
    # -------------------------------------------------------------------------
    console.print("\n[bold]Top Languages:[/bold]")

    lang_table = Table(show_header=False, box=None, padding=(0, 2))
    lang_table.add_column("Language", style="cyan")
    lang_table.add_column("Percentage", style="white")

    for lang_data in languages.get("top_languages", [])[:5]:
        lang_table.add_row(
            lang_data.get("language", "Unknown"),
            f"{lang_data.get('percentage', 0):.1f}%"
        )

    console.print(lang_table)

    # -------------------------------------------------------------------------
    # Strengths and improvements
    # -------------------------------------------------------------------------
    strengths = profile.get("strengths", [])
    if strengths:
        console.print("\n[bold green]Strengths:[/bold green]")
        for strength in strengths[:3]:
            console.print(f"  ✓ {strength}")

    improvements = profile.get("improvements", [])
    if improvements:
        console.print("\n[bold yellow]Areas for Improvement:[/bold yellow]")
        for improvement in improvements[:3]:
            console.print(f"  • {improvement}")

    # -------------------------------------------------------------------------
    # AI Report preview
    # -------------------------------------------------------------------------
    if insights and insights.get("report"):
        console.print("\n[bold]AI Insights Preview:[bold]")

        report = insights["report"]
        # Show first 300 characters
        preview = report[:300].replace("\n", " ")
        if len(report) > 300:
            preview += "..."

        console.print(f"  [dim]{preview}[/dim]")
        console.print(f"\n  [dim]Model: {insights.get('model', 'unknown')} | "
                    f"Tokens: {insights.get('tokens_used', 0)} | "
                    f"Time: {insights.get('generation_time_ms', 0)}ms[/dim]")

    # -------------------------------------------------------------------------
    # Fun facts
    # -------------------------------------------------------------------------
    fun_facts = result.get("fun_facts", {})
    if fun_facts and fun_facts.get("facts"):
        console.print("\n[bold]🎉 Fun Facts:[/bold]")
        for i, fact in enumerate(fun_facts["facts"][:3], 1):
            console.print(f"  {i}. [green]{fact}[/green]")

    # -------------------------------------------------------------------------
    # Execution info
    # -------------------------------------------------------------------------
    console.print(f"\n[dim]Execution time: {result.get('execution_time', 0):.2f}s[/dim]")

    errors = result.get("errors", [])
    if errors:
        console.print(f"[red]Errors: {len(errors)}[/red]")

    console.print("=" * 70)


def _day_name(day_num: Optional[int]) -> str:
    """Convert day number (0-6) to day name."""
    if day_num is None:
        return "Unknown"

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    try:
        return days[int(day_num)]
    except (IndexError, ValueError):
        return "Unknown"


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> int:
    """
    CLI entry point.

    WHY: Provides a user-friendly command-line interface with:
        - Help text and examples
        - Argument parsing and validation
        - Error handling with clear messages
        - Output options (console, JSON file)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="GitHub Profile Analyzer - Comprehensive analysis of GitHub profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py octocat
  python main.py octocat --force-refresh
  python main.py octocat --no-ai
  python main.py octocat --token ghp_xxxx
  python main.py octocat --output results.json
        """,
    )

    parser.add_argument(
        "username",
        help="GitHub username to analyze"
    )

    parser.add_argument(
        "--token", "-t",
        help="GitHub token (increases rate limit from 60 to 5000 req/hr)"
    )

    parser.add_argument(
        "--force-refresh", "-f",
        action="store_true",
        help="Skip cache and fetch fresh data"
    )

    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip AI insights generation"
    )

    parser.add_argument(
        "--output", "-o",
        help="Save results to JSON file"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = run_analysis(
            username=args.username,
            token=args.token,
            force_refresh=args.force_refresh,
            generate_ai_insights=not args.no_ai,
            verbose=args.verbose,
        )

        # Save to file if requested
        if args.output:
            # Convert non-serializable objects
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                if hasattr(obj, "to_dict"):
                    return obj.to_dict()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=json_serializer)

            console.print(f"\n[green]Results saved to: {args.output}[/green]")

        # Return exit code based on errors
        return 0 if not result.get("errors") else 1

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        return 130  # Standard exit code for Ctrl+C

    except Exception as e:
        console.print(f"\n[red]Analysis failed: {e}[/red]")
        logger.exception("Unhandled exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
