"""
Configuration Management Module for GitHub Profile Analyzer.

This module uses Pydantic Settings to manage configuration from multiple sources:
    1. Environment variables (highest priority)
    2. .env files
    3. Default values (lowest priority)

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates understanding of the 12-Factor App methodology
    - Shows proper separation of configuration from code
    - Implements environment-specific settings for dev/staging/prod
    - Uses type-safe configuration with runtime validation

Usage:
    from src.config import settings

    # Access configuration values
    print(settings.github_token)
    print(settings.db_path)
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from loguru import logger


# =============================================================================
# Configuration Settings Class
# =============================================================================

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    This class uses Pydantic Settings to:
        - Load values from .env files automatically
        - Validate types at runtime
        - Provide default values for optional settings
        - Convert string values to appropriate Python types

    WHY: Centralizing configuration makes the application:
        - Easier to test (can override settings)
        - Easier to deploy (environment-specific config)
        - More secure (sensitive values from env vars, not hardcoded)
    """

    # =========================================================================
    # Pydantic Settings Configuration
    # =========================================================================
    model_config = SettingsConfigDict(
        # Load settings from .env file in the project root
        # WHY: Convenient for local development, follows 12-factor app
        env_file=".env",
        # Also check parent directory for .env (for running from subdirectories)
        env_file_encoding="utf-8",
        # Allow extra fields in .env without validation errors
        # WHY: Prevents errors when .env has comments or unknown keys
        extra="ignore",
        # Make all fields case-insensitive
        # WHY: Environment variables are often uppercase, but we use lowercase
        case_sensitive=False,
    )

    # =========================================================================
    # GitHub API Configuration
    # =========================================================================

    github_token: Optional[str] = Field(
        default=None,
        description="GitHub Personal Access Token for API authentication",
    )
    # WHY: Optional because unauthenticated access is allowed (60 req/hr)
    # With token: 5000 req/hr, which is essential for large profiles

    # =========================================================================
    # ZhipuAI API Configuration
    # =========================================================================

    zhipuai_api_key: Optional[str] = Field(
        default=None,
        description="API key for ZhipuAI GLM models",
    )
    # WHY: Required for AI insights but made optional to allow
    # running the tool without LLM features (CLI-only mode)

    # =========================================================================
    # Database Configuration
    # =========================================================================

    db_path: str = Field(
        default="./data/github_analyzer.db",
        description="Path to SQLite database file",
    )
    # WHY: SQLite is perfect for this use case:
    # - Single-user application (one analyst at a time)
    # - Embedded database (no server setup required)
    # - Good performance for read-heavy workloads

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """
        Ensure database directory exists.

        WHY: Prevents "No such file or directory" errors when
        trying to create the database file for the first time.
        """
        db_file = Path(v)
        db_dir = db_file.parent

        # Create directory if it doesn't exist
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")

        return v

    # =========================================================================
    # Cache Configuration
    # =========================================================================

    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Cache time-to-live in hours",
    )
    # WHY: 24 hours is a good balance between:
    # - Data freshness (GitHub profiles don't change drastically daily)
    # - API efficiency (reduce repeated calls for same profile)

    # =========================================================================
    # Rate Limiting Configuration
    # =========================================================================

    rate_limit_warning_threshold: int = Field(
        default=100,
        ge=10,
        description="Warn when remaining API calls drop below this threshold",
    )
    # WHY: Gives buffer before hitting rate limit, allowing
    # graceful degradation rather than hard failures

    rate_limit_sleep_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,  # Max 1 hour sleep
        description="Seconds to sleep when approaching rate limit",
    )
    # WHY: GitHub rate limit resets after 1 hour, so sleeping
    # for shorter periods is more responsive

    # =========================================================================
    # Data Collection Limits
    # =========================================================================

    max_repos: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum repositories to analyze per profile",
    )
    # WHY: Prevents runaway collection on profiles with thousands of repos
    # (e.g., organizations or prolific open-source contributors)

    max_commits_per_repo: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum commits to fetch per repository",
    )
    # WHY: Popular repos can have 10,000+ commits; we focus on recent activity

    # =========================================================================
    # Analysis Configuration
    # =========================================================================

    anomaly_threshold: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Z-score threshold for detecting anomalies",
    )
    # WHY: Z-score of 2.0 means ~95% confidence interval
    # Values outside this range are statistically unusual

    min_commits_for_analysis: int = Field(
        default=10,
        ge=5,
        description="Minimum commits needed for reliable statistics",
    )
    # WHY: Statistical measures (mean, std, etc.) require minimum sample size
    # Smaller samples produce unreliable metrics

    burnout_threshold_days: int = Field(
        default=7,
        ge=3,
        le=30,
        description="Consecutive inactive days to flag as burnout",
    )
    # WHY: 7 days is a common vacation length; longer periods may indicate
    # burnout, job change, or project completion

    # =========================================================================
    # Logging Configuration
    # =========================================================================

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum log level to display",
    )
    # WHY: DEBUG for development (verbose), INFO for production (cleaner logs)

    log_file: Optional[str] = Field(
        default="./logs/github_analyzer.log",
        description="Path to log file (None for console-only)",
    )
    # WHY: File logging enables debugging production issues after the fact

    # =========================================================================
    # LLM Configuration
    # =========================================================================

    llm_model: str = Field(
        default="glm-4-flash",
        description="ZhipuAI model for insights generation",
    )
    # WHY: glm-4-flash offers best speed/quality balance for this use case
    # glm-4 for higher quality, glm-3-turbo for faster/cheaper

    llm_temperature_insights: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for analytical insights (lower = more factual)",
    )
    # WHY: Low temperature (0.3) for insights ensures factual, consistent outputs
    # High temperature would produce creative but potentially inaccurate analysis

    llm_temperature_fun_facts: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for fun facts (higher = more creative)",
    )
    # WHY: Higher temperature (0.7) for fun facts allows more creative, engaging outputs
    # While still grounded in the actual data

    # =========================================================================
    # Streamlit Dashboard Configuration
    # =========================================================================

    streamlit_port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Port for Streamlit dashboard",
    )
    # WHY: Non-privileged ports (>1024) don't require root access

    streamlit_address: str = Field(
        default="0.0.0.0",
        description="Network interface to bind (0.0.0.0 for all interfaces)",
    )
    # WHY: 0.0.0.0 allows access from any network interface
    # Required for Docker/container deployment

    # =========================================================================
    # Computed Properties
    # =========================================================================

    @property
    def github_api_base_url(self) -> str:
        """GitHub API base URL (constant, but configurable for testing)."""
        return "https://api.github.com"

    @property
    def is_configured_for_llm(self) -> bool:
        """Check if LLM features are available."""
        return self.zhipuai_api_key is not None and len(self.zhipuai_api_key) > 0

    @property
    def is_authenticated(self) -> bool:
        """Check if GitHub API is authenticated."""
        return self.github_token is not None and len(self.github_token) > 0


# =============================================================================
# Singleton Instance with Caching
# =============================================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.

    WHY: Using lru_cache ensures:
        - Settings are loaded only once
        - Same instance used throughout application
        - Environment changes during runtime don't cause confusion

    Returns:
        Settings: Cached settings instance
    """
    logger.info("Loading application settings")
    return Settings()


# Create a module-level singleton for convenient import
settings = get_settings()


# =============================================================================
# Logging Configuration
# =============================================================================

def configure_logging() -> None:
    """
    Configure loguru logging based on settings.

    WHY: Centralized logging configuration ensures consistent format
    across all modules and enables easy debugging.

    This function:
        1. Removes default handler (to reconfigure)
        2. Adds console handler with colorized output
        3. Adds file handler with rotation if log_file is set
    """
    import sys

    # Remove default handler added by loguru
    # WHY: Allows us to reconfigure with custom settings
    logger.remove()

    # Add console handler with custom format
    # WHY: Colorized output improves readability in terminal
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )

    # Add file handler if log file is configured
    # WHY: Persistent logs enable post-mortem debugging
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            settings.log_file,
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip",  # Compress rotated logs
        )
        logger.info(f"Logging to file: {settings.log_file}")


# Configure logging when module is imported
# WHY: Ensures logging is ready before any other code runs
configure_logging()


# =============================================================================
# Export Public Interface
# =============================================================================

__all__ = ["Settings", "settings", "get_settings", "configure_logging"]
