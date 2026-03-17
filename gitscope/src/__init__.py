"""
GitScope - GitHub Profile Analyzer.

See the full scope of any GitHub developer.

This package provides comprehensive GitHub profile analysis:
    - Data collection from GitHub API
    - SQLite database storage
    - Data transformation and metrics
    - Language and repository analytics
    - Plotly visualizations
    - LLM-powered insights

Modules:
    - config: Configuration management with Pydantic Settings
    - collector: GitHub API data collection with caching
    - storage: SQLAlchemy ORM models and database operations
    - transformer: Data cleaning and aggregation pipeline
    - analytics: Language, repository, and commit analysis
    - visualizations: Interactive Plotly charts
    - insights: LLM-powered insights generation

Example:
    from gitscope.collector import GitHubCollector
    from gitscope.analytics import generate_developer_profile

    collector = GitHubCollector("octocat")
    data = collector.collect_all()
"""

__version__ = "1.0.0"
__author__ = "GitScope"
__project__ = "GitScope"

from src.config import settings, configure_logging

# Configure logging when package is imported
configure_logging()

__all__ = [
    "settings",
    "configure_logging",
]
