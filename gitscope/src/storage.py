"""
Database Storage Module for GitHub Profile Analyzer.

This module implements data persistence using SQLAlchemy ORM:
    - User profile storage
    - Repository metadata storage
    - Commit history storage
    - Language breakdown storage

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates SQL/ORM proficiency with SQLAlchemy
    - Shows proper database design (normalization, indexes, constraints)
    - Implements computed columns for efficiency
    - Uses context managers for connection handling
    - Supports both insert and query operations

Usage:
    from src.storage import save_all_data, load_commits_df

    # Save collected data
    save_all_data(data, "./data/github_analyzer.db")

    # Load as DataFrames
    commits_df = load_commits_df("./data/github_analyzer.db")
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    func,
    text,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.types import JSON

from loguru import logger


# =============================================================================
# SQLAlchemy Base Class
# =============================================================================

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.

    WHY: Using DeclarativeBase (SQLAlchemy 2.0 style) provides:
        - Modern type annotations
        - Better IDE support
        - Cleaner model definitions
    """
    pass


# =============================================================================
# Database Models
# =============================================================================

class UserModel(Base):
    """
    SQLAlchemy model for GitHub user profiles.

    WHY: Storing user profiles allows:
        - Historical tracking of profile changes
        - Quick lookup without API calls
        - Correlation with commit/activity data
    """
    __tablename__ = "users"

    # Primary key
    # WHY: Using GitHub's user ID ensures uniqueness across username changes
    id = Column(Integer, primary_key=True, autoincrement=True)
    github_id = Column(Integer, unique=True, nullable=True, index=True)

    # Core profile fields
    username = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    company = Column(String(255), nullable=True)
    blog = Column(String(500), nullable=True)
    email = Column(String(255), nullable=True)
    twitter_username = Column(String(100), nullable=True)

    # URLs
    avatar_url = Column(String(500), nullable=True)
    html_url = Column(String(500), nullable=True)

    # Stats
    public_repos = Column(Integer, default=0)
    public_gists = Column(Integer, default=0)
    followers = Column(Integer, default=0)
    following = Column(Integer, default=0)

    # Timestamps from GitHub
    github_created_at = Column(DateTime(timezone=True), nullable=True)
    github_updated_at = Column(DateTime(timezone=True), nullable=True)

    # Our metadata
    collected_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    # WHY: Enables ORM-level joins without manual foreign key handling
    repositories = relationship("RepositoryModel", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(username='{self.username}', name='{self.name}')>"


class RepositoryModel(Base):
    """
    SQLAlchemy model for GitHub repositories.

    WHY: Repository metadata provides context for commit analysis:
        - Stars/forks indicate project impact
        - Language shows technology focus
        - Activity dates show project lifecycle
    """
    __tablename__ = "repositories"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    github_id = Column(Integer, unique=True, nullable=True, index=True)

    # Foreign key to user
    # WHY: Enables queries like "all repos for user X"
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # Core fields
    name = Column(String(255), nullable=False, index=True)
    full_name = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    html_url = Column(String(500), nullable=True)

    # Stats
    stargazers_count = Column(Integer, default=0)
    watchers_count = Column(Integer, default=0)
    forks_count = Column(Integer, default=0)
    open_issues_count = Column(Integer, default=0)
    size = Column(Integer, default=0)  # KB

    # Language
    primary_language = Column(String(100), nullable=True, index=True)
    # WHY: Indexed for fast queries like "repos using Python"

    # Flags
    is_fork = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    is_template = Column(Boolean, default=False)
    has_issues = Column(Boolean, default=False)
    has_projects = Column(Boolean, default=False)
    has_wiki = Column(Boolean, default=False)
    has_pages = Column(Boolean, default=False)

    # Additional metadata
    license = Column(String(100), nullable=True)
    topics = Column(JSON, default=list)
    default_branch = Column(String(100), default="main")

    # Timestamps from GitHub
    github_created_at = Column(DateTime(timezone=True), nullable=True)
    github_updated_at = Column(DateTime(timezone=True), nullable=True)
    github_pushed_at = Column(DateTime(timezone=True), nullable=True)

    # Our metadata
    collected_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user = relationship("UserModel", back_populates="repositories")
    commits = relationship("CommitModel", back_populates="repository", cascade="all, delete-orphan")
    languages = relationship("RepoLanguageModel", back_populates="repository", cascade="all, delete-orphan")

    # Composite index for common queries
    # WHY: Enables efficient queries like "repos by user X ordered by stars"
    __table_args__ = (
        Index("ix_repositories_user_stars", "user_id", "stargazers_count"),
        Index("ix_repositories_user_language", "user_id", "primary_language"),
    )

    def __repr__(self) -> str:
        return f"<Repository(name='{self.name}', stars={self.stargazers_count})>"


class CommitModel(Base):
    """
    SQLAlchemy model for Git commits.

    WHY: Commit history is the core data for behavioral analysis:
        - Timing patterns reveal work habits
        - Message patterns reveal communication style
        - Frequency patterns reveal productivity

    Note: We store computed fields to avoid recalculation during analysis.
    """
    __tablename__ = "commits"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to repository
    # WHY: Enables queries like "all commits for repo X"
    repo_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True)

    # Core fields
    sha = Column(String(40), unique=True, nullable=False, index=True)
    # WHY: SHA is globally unique across all repos

    # Commit details
    message = Column(Text, nullable=True)
    message_length = Column(Integer, default=0)
    # WHY: Message length indicates commit quality (too short or too long)

    # Author info
    author_name = Column(String(255), nullable=True)
    author_email = Column(String(255), nullable=True, index=True)
    # WHY: Email index enables queries like "commits by author"

    # Committer info (may differ from author for squash/rebase)
    committer_name = Column(String(255), nullable=True)
    committer_email = Column(String(255), nullable=True)

    # Timestamps
    author_date = Column(DateTime(timezone=True), nullable=True, index=True)
    committer_date = Column(DateTime(timezone=True), nullable=True)

    # Computed fields for efficient analysis
    # WHY: Pre-computed fields avoid repeated calculations during analysis
    hour_of_day = Column(Integer, nullable=True)
    # WHY: Enables fast hourly distribution queries
    day_of_week = Column(Integer, nullable=True)
    # WHY: 0=Monday, 6=Sunday for day patterns
    week_number = Column(Integer, nullable=True)
    # WHY: ISO week number for weekly aggregations
    year = Column(Integer, nullable=True)
    # WHY: For yearly comparisons
    is_weekend = Column(Boolean, default=False)
    # WHY: Work/life balance analysis

    # URL
    html_url = Column(String(500), nullable=True)

    # Our metadata
    collected_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    repository = relationship("RepositoryModel", back_populates="commits")

    # Composite indexes for common queries
    # WHY: Enables efficient time-based queries
    __table_args__ = (
        Index("ix_commits_repo_date", "repo_id", "author_date"),
        Index("ix_commits_repo_hour", "repo_id", "hour_of_day"),
        Index("ix_commits_date_hour", "author_date", "hour_of_day"),
    )

    def __repr__(self) -> str:
        return f"<Commit(sha='{self.sha[:8]}...', repo_id={self.repo_id})>"


class RepoLanguageModel(Base):
    """
    SQLAlchemy model for repository language breakdown.

    WHY: Language data reveals technology expertise:
        - Primary languages show specialization
        - Language diversity shows breadth
        - Changes over time show learning trajectory

    Note: We store bytes per language to calculate accurate percentages.
    """
    __tablename__ = "repo_languages"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to repository
    repo_id = Column(Integer, ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False, index=True)

    # Language info
    language = Column(String(100), nullable=False, index=True)
    bytes_count = Column(Integer, default=0)

    # Computed percentage (for historical tracking)
    percentage = Column(Float, default=0.0)

    # Our metadata
    collected_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    repository = relationship("RepositoryModel", back_populates="languages")

    # Composite unique constraint
    # WHY: Ensures no duplicate language entries per repo
    __table_args__ = (
        Index("ix_repo_languages_repo_lang", "repo_id", "language", unique=True),
    )

    def __repr__(self) -> str:
        return f"<RepoLanguage(repo_id={self.repo_id}, language='{self.language}', bytes={self.bytes_count})>"


# =============================================================================
# Database Engine and Session Management
# =============================================================================

class DatabaseManager:
    """
    Manages database connections and sessions.

    WHY: Centralized database management provides:
        - Connection pooling for efficiency
        - Consistent session handling
        - Clean resource cleanup
        - Transaction management
    """

    def __init__(self, db_path: str):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)

        # Ensure directory exists
        # WHY: SQLite fails silently if directory doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine with optimized settings
        # WHY: These settings improve performance and reliability
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,  # Set True for SQL debugging
            pool_pre_ping=True,  # Check connection health
            connect_args={"check_same_thread": False},  # Allow multi-threading
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        # Create tables if they don't exist
        # WHY: Ensures schema is ready before any operations
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized: {self.db_path}")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        WHY: Ensures proper cleanup even if exceptions occur.
        Using context managers is a Python best practice for resource management.

        Yields:
            SQLAlchemy Session object
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# =============================================================================
# Data Persistence Functions
# =============================================================================

def _parse_github_datetime(dt_string: Optional[str]) -> Optional[datetime]:
    """
    Parse GitHub's ISO 8601 datetime string.

    WHY: GitHub uses various ISO 8601 formats, and we need to handle
    timezone info correctly for accurate time-based analysis.

    Args:
        dt_string: ISO 8601 datetime string

    Returns:
        Parsed datetime or None
    """
    if not dt_string:
        return None

    try:
        # Handle GitHub's format: 2024-01-15T10:30:00Z
        if dt_string.endswith("Z"):
            dt_string = dt_string[:-1] + "+00:00"

        return datetime.fromisoformat(dt_string)
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse datetime '{dt_string}': {e}")
        return None


def _compute_commit_fields(author_date: Optional[datetime]) -> Dict[str, Any]:
    """
    Compute derived fields for a commit.

    WHY: Pre-computing these fields during insert avoids expensive
    calculations during analysis. Trade-off: slightly slower insert,
    much faster queries.

    Args:
        author_date: Commit author timestamp

    Returns:
        Dictionary with computed fields
    """
    if not author_date:
        return {
            "hour_of_day": None,
            "day_of_week": None,
            "week_number": None,
            "year": None,
            "is_weekend": False,
        }

    return {
        "hour_of_day": author_date.hour,
        "day_of_week": author_date.weekday(),  # 0=Monday, 6=Sunday
        "week_number": author_date.isocalendar()[1],
        "year": author_date.year,
        "is_weekend": author_date.weekday() >= 5,  # Saturday or Sunday
    }


def save_all_data(data: Dict[str, Any], db_path: str) -> Dict[str, int]:
    """
    Save all collected data to the database.

    WHY: Provides a single entry point for persisting all collected data.
    Handles the complexity of relationships and computed fields.

    Args:
        data: Dictionary containing profile, repositories, commits, and languages
        db_path: Path to SQLite database file

    Returns:
        Dictionary with counts of inserted records

    Raises:
        SQLAlchemyError: On database errors
    """
    logger.info(f"Saving data to {db_path}")

    db = DatabaseManager(db_path)
    counts = {
        "users": 0,
        "repositories": 0,
        "commits": 0,
        "languages": 0,
    }

    with db.get_session() as session:
        # =====================================================================
        # Save User Profile
        # =====================================================================
        profile = data.get("profile", {})

        # Check if user already exists
        # WHY: Upsert pattern prevents duplicates on re-analysis
        user = session.query(UserModel).filter_by(username=profile.get("username")).first()

        if user:
            # Update existing user
            for key, value in profile.items():
                if key not in ["username", "from_cache"]:
                    # Map field names
                    db_key = f"github_{key}" if key in ["created_at", "updated_at"] else key
                    if hasattr(user, db_key):
                        if key in ["created_at", "updated_at"]:
                            setattr(user, db_key, _parse_github_datetime(value))
                        else:
                            setattr(user, db_key, value)

            user.updated_at = datetime.now(timezone.utc)
            logger.debug(f"Updated existing user: {user.username}")
        else:
            # Create new user
            user = UserModel(
                username=profile.get("username"),
                name=profile.get("name"),
                bio=profile.get("bio"),
                location=profile.get("location"),
                company=profile.get("company"),
                blog=profile.get("blog"),
                email=profile.get("email"),
                twitter_username=profile.get("twitter_username"),
                avatar_url=profile.get("avatar_url"),
                html_url=profile.get("html_url"),
                public_repos=profile.get("public_repos", 0),
                public_gists=profile.get("public_gists", 0),
                followers=profile.get("followers", 0),
                following=profile.get("following", 0),
                github_created_at=_parse_github_datetime(profile.get("created_at")),
                github_updated_at=_parse_github_datetime(profile.get("updated_at")),
            )
            session.add(user)

        session.flush()  # Get user.id
        counts["users"] = 1

        # =====================================================================
        # Save Repositories
        # =====================================================================
        repositories = data.get("repositories", [])

        for repo_data in repositories:
            # Check if repository already exists
            repo = session.query(RepositoryModel).filter_by(
                user_id=user.id,
                name=repo_data.get("name"),
            ).first()

            if repo:
                # Update existing repository
                for key, value in repo_data.items():
                    if key not in ["id", "name"] and hasattr(repo, key):
                        setattr(repo, key, value)
            else:
                # Create new repository
                repo = RepositoryModel(
                    user_id=user.id,
                    github_id=repo_data.get("id"),
                    name=repo_data.get("name"),
                    full_name=repo_data.get("full_name"),
                    description=repo_data.get("description"),
                    html_url=repo_data.get("html_url"),
                    primary_language=repo_data.get("language"),
                    stargazers_count=repo_data.get("stargazers_count", 0),
                    watchers_count=repo_data.get("watchers_count", 0),
                    forks_count=repo_data.get("forks_count", 0),
                    open_issues_count=repo_data.get("open_issues_count", 0),
                    size=repo_data.get("size", 0),
                    is_fork=repo_data.get("is_fork", False),
                    is_archived=repo_data.get("is_archived", False),
                    is_template=repo_data.get("is_template", False),
                    has_issues=repo_data.get("has_issues", False),
                    has_projects=repo_data.get("has_projects", False),
                    has_wiki=repo_data.get("has_wiki", False),
                    has_pages=repo_data.get("has_pages", False),
                    license=repo_data.get("license"),
                    topics=repo_data.get("topics", []),
                    default_branch=repo_data.get("default_branch", "main"),
                    github_created_at=_parse_github_datetime(repo_data.get("created_at")),
                    github_updated_at=_parse_github_datetime(repo_data.get("updated_at")),
                    github_pushed_at=_parse_github_datetime(repo_data.get("pushed_at")),
                )
                session.add(repo)

            session.flush()  # Get repo.id
            counts["repositories"] += 1

            # =====================================================================
            # Save Commits for this Repository
            # =====================================================================
            repo_name = repo_data.get("name")
            commits_data = data.get("commits", {}).get(repo_name, [])

            for commit_data in commits_data:
                # Parse author date for computed fields
                author_date = _parse_github_datetime(commit_data.get("author_date"))
                computed = _compute_commit_fields(author_date)

                # Check if commit already exists
                existing = session.query(CommitModel).filter_by(
                    sha=commit_data.get("sha")
                ).first()

                if existing:
                    continue  # Skip duplicates

                commit = CommitModel(
                    repo_id=repo.id,
                    sha=commit_data.get("sha"),
                    message=commit_data.get("message", ""),
                    message_length=len(commit_data.get("message", "")),
                    author_name=commit_data.get("author_name"),
                    author_email=commit_data.get("author_email"),
                    author_date=author_date,
                    committer_name=commit_data.get("committer_name"),
                    committer_email=commit_data.get("committer_email"),
                    committer_date=_parse_github_datetime(commit_data.get("committer_date")),
                    html_url=commit_data.get("html_url"),
                    **computed,  # Include computed fields
                )
                session.add(commit)
                counts["commits"] += 1

            # =====================================================================
            # Save Languages for this Repository
            # =====================================================================
            languages_data = data.get("languages", {}).get(repo_name, {})

            if languages_data:
                total_bytes = sum(languages_data.values()) or 1  # Avoid division by zero

                for language, bytes_count in languages_data.items():
                    # Check if language entry already exists
                    existing = session.query(RepoLanguageModel).filter_by(
                        repo_id=repo.id,
                        language=language,
                    ).first()

                    if existing:
                        existing.bytes_count = bytes_count
                        existing.percentage = (bytes_count / total_bytes) * 100
                    else:
                        lang_entry = RepoLanguageModel(
                            repo_id=repo.id,
                            language=language,
                            bytes_count=bytes_count,
                            percentage=(bytes_count / total_bytes) * 100,
                        )
                        session.add(lang_entry)
                        counts["languages"] += 1

    logger.info(
        f"Saved: {counts['users']} users, {counts['repositories']} repos, "
        f"{counts['commits']} commits, {counts['languages']} language entries"
    )

    return counts


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_commits_df(db_path: str, username: Optional[str] = None) -> pd.DataFrame:
    """
    Load commits from database into a pandas DataFrame.

    WHY: DataFrame format enables efficient vectorized operations
    for analysis. Much faster than iterating over ORM objects.

    Args:
        db_path: Path to SQLite database file
        username: Optional username filter

    Returns:
        DataFrame with commit data
    """
    logger.info(f"Loading commits from {db_path}")

    db = DatabaseManager(db_path)

    # Build query with optional username filter
    # WHY: SQLAlchemy Core queries are faster than ORM for bulk reads
    query = """
        SELECT
            c.id,
            c.sha,
            c.repo_id,
            r.name as repo_name,
            r.primary_language as repo_language,
            c.message,
            c.message_length,
            c.author_name,
            c.author_email,
            c.author_date,
            c.committer_date,
            c.hour_of_day,
            c.day_of_week,
            c.week_number,
            c.year,
            c.is_weekend,
            c.html_url
        FROM commits c
        JOIN repositories r ON c.repo_id = r.id
        JOIN users u ON r.user_id = u.id
    """

    if username:
        query += " WHERE u.username = :username"
        params = {"username": username}
    else:
        params = {}

    with db.engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, params=params)

    # Convert timestamp columns to datetime
    # WHY: Enables pandas time-series functionality
    datetime_cols = ["author_date", "committer_date"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    logger.info(f"Loaded {len(df)} commits")
    return df


def load_repos_df(db_path: str, username: Optional[str] = None) -> pd.DataFrame:
    """
    Load repositories from database into a pandas DataFrame.

    WHY: DataFrame format enables efficient filtering and aggregation
    for repository-level analysis.

    Args:
        db_path: Path to SQLite database file
        username: Optional username filter

    Returns:
        DataFrame with repository data
    """
    logger.info(f"Loading repositories from {db_path}")

    db = DatabaseManager(db_path)

    query = """
        SELECT
            r.id,
            r.name,
            r.full_name,
            r.description,
            r.html_url,
            r.primary_language,
            r.stargazers_count,
            r.watchers_count,
            r.forks_count,
            r.open_issues_count,
            r.size,
            r.is_fork,
            r.is_archived,
            r.is_template,
            r.license,
            r.topics,
            r.github_created_at,
            r.github_updated_at,
            r.github_pushed_at,
            u.username
        FROM repositories r
        JOIN users u ON r.user_id = u.id
    """

    if username:
        query += " WHERE u.username = :username"
        params = {"username": username}
    else:
        params = {}

    with db.engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, params=params)

    # Convert timestamp columns to datetime
    datetime_cols = ["github_created_at", "github_updated_at", "github_pushed_at"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    logger.info(f"Loaded {len(df)} repositories")
    return df


def load_languages_df(db_path: str, username: Optional[str] = None) -> pd.DataFrame:
    """
    Load language breakdown from database into a pandas DataFrame.

    WHY: DataFrame format enables efficient aggregation and percentage
    calculations across repositories.

    Args:
        db_path: Path to SQLite database file
        username: Optional username filter

    Returns:
        DataFrame with language data
    """
    logger.info(f"Loading languages from {db_path}")

    db = DatabaseManager(db_path)

    query = """
        SELECT
            rl.id,
            rl.repo_id,
            r.name as repo_name,
            rl.language,
            rl.bytes_count,
            rl.percentage,
            u.username
        FROM repo_languages rl
        JOIN repositories r ON rl.repo_id = r.id
        JOIN users u ON r.user_id = u.id
    """

    if username:
        query += " WHERE u.username = :username"
        params = {"username": username}
    else:
        params = {}

    with db.engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, params=params)

    logger.info(f"Loaded {len(df)} language entries")
    return df


def get_db_stats(db_path: str) -> Dict[str, Any]:
    """
    Get database statistics.

    WHY: Provides quick overview of collected data for UI display
    and validation.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Dictionary with statistics
    """
    db = DatabaseManager(db_path)

    stats = {}

    with db.get_session() as session:
        stats["users"] = session.query(func.count(UserModel.id)).scalar() or 0
        stats["repositories"] = session.query(func.count(RepositoryModel.id)).scalar() or 0
        stats["commits"] = session.query(func.count(CommitModel.id)).scalar() or 0
        stats["languages"] = session.query(func.count(RepoLanguageModel.id)).scalar() or 0

        # Get date range of commits
        result = session.query(
            func.min(CommitModel.author_date),
            func.max(CommitModel.author_date),
        ).first()

        if result and result[0]:
            stats["earliest_commit"] = result[0].isoformat()
            stats["latest_commit"] = result[1].isoformat()

            # Calculate span in days
            delta = result[1] - result[0]
            stats["commit_span_days"] = delta.days
        else:
            stats["earliest_commit"] = None
            stats["latest_commit"] = None
            stats["commit_span_days"] = 0

    return stats


# =============================================================================
# Export Public Interface
# =============================================================================

__all__ = [
    # Models
    "Base",
    "UserModel",
    "RepositoryModel",
    "CommitModel",
    "RepoLanguageModel",
    # Database
    "DatabaseManager",
    # Persistence
    "save_all_data",
    # Loading
    "load_commits_df",
    "load_repos_df",
    "load_languages_df",
    "get_db_stats",
]
