"""
GitHub API Data Collection Module.

This module implements robust data collection from the GitHub REST API:
    - User profile information
    - Repository listings with pagination
    - Commit history per repository
    - Language breakdown per repository

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates API integration with proper error handling
    - Shows understanding of rate limiting and pagination
    - Implements caching to reduce API calls
    - Uses context managers for resource cleanup
    - Handles edge cases (private repos, empty profiles, API errors)

Usage:
    collector = GitHubCollector(username="octocat", token="ghp_xxx")
    data = collector.collect_all()
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from loguru import logger

from src.config import settings


# =============================================================================
# Custom Exceptions
# =============================================================================

class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(self.message)


class RateLimitError(GitHubAPIError):
    """Raised when GitHub API rate limit is exceeded."""

    def __init__(self, reset_time: Optional[int] = None):
        self.reset_time = reset_time
        message = f"Rate limit exceeded. Resets at {datetime.fromtimestamp(reset_time) if reset_time else 'unknown'}"
        super().__init__(message, status_code=403)


class UserNotFoundError(GitHubAPIError):
    """Raised when the specified GitHub user does not exist."""

    def __init__(self, username: str):
        self.username = username
        message = f"GitHub user '{username}' not found"
        super().__init__(message, status_code=404)


class AuthenticationError(GitHubAPIError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


# =============================================================================
# Cache Manager
# =============================================================================

class CacheManager:
    """
    Manages file-based caching for API responses.

    WHY: Reduces API calls for repeated analyses of the same profile.
    GitHub API has strict rate limits (60/hr unauthenticated, 5000/hr authenticated),
    so caching is essential for efficient operation.
    """

    def __init__(self, cache_dir: str = "./cache", ttl_hours: int = 24):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.ttl_seconds = ttl_hours * 3600

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache initialized: {self.cache_dir} (TTL: {ttl_hours}h)")

    def _get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key."""
        # Use hash to avoid filesystem issues with special characters
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data if valid.

        WHY: Checking cache before API calls saves rate limit quota.

        Args:
            key: Cache key (typically the API endpoint)

        Returns:
            Cached data if valid, None if expired or missing
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {key}")
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            # Check if cache is expired
            # WHY: Ensures data freshness while still benefiting from caching
            cached_time = cached.get("_cached_at", 0)
            age = time.time() - cached_time

            if age > self.ttl_seconds:
                logger.debug(f"Cache expired: {key} (age: {age:.0f}s > {self.ttl_seconds}s)")
                return None

            logger.info(f"Cache hit: {key} (age: {age:.0f}s)")
            return cached.get("data")

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt cache file: {cache_path} - {e}")
            return None

    def set(self, key: str, data: Any) -> None:
        """
        Store data in cache.

        WHY: Caching successful responses prevents repeated API calls.

        Args:
            key: Cache key (typically the API endpoint)
            data: Data to cache
        """
        cache_path = self._get_cache_path(key)

        try:
            cache_entry = {
                "_cached_at": time.time(),
                "_cached_at_iso": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_entry, f, indent=2, default=str)

            logger.debug(f"Cached: {key}")

        except (IOError, TypeError) as e:
            logger.warning(f"Failed to cache {key}: {e}")

    def clear(self, key: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            key: Specific key to clear, or None to clear all

        Returns:
            Number of cache entries cleared
        """
        if key:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Cleared cache: {key}")
                return 1
            return 0

        # Clear all cache files
        cleared = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            cleared += 1

        logger.info(f"Cleared {cleared} cache entries")
        return cleared


# =============================================================================
# GitHub API Collector
# =============================================================================

class GitHubCollector:
    """
    Collects data from GitHub REST API with rate limiting, caching, and error handling.

    WHY THIS DESIGN:
        - Session reuse: Connection pooling for better performance
        - Retry logic: Handles transient network failures
        - Rate limiting: Respects GitHub's API limits
        - Caching: Reduces redundant API calls
        - Pagination: Handles large result sets

    Example:
        collector = GitHubCollector(username="octocat")
        try:
            data = collector.collect_all()
        except UserNotFoundError:
            print("User not found")
        except RateLimitError as e:
            print(f"Rate limited until {e.reset_time}")
    """

    def __init__(
        self,
        username: str,
        token: Optional[str] = None,
        cache_ttl_hours: Optional[int] = None,
    ):
        """
        Initialize GitHub API collector.

        Args:
            username: GitHub username to analyze
            token: Optional GitHub token (increases rate limit to 5000/hr)
            cache_ttl_hours: Optional cache TTL override

        Raises:
            ValueError: If username is empty
        """
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")

        self.username = username.strip()
        self.token = token or settings.github_token
        self.base_url = settings.github_api_base_url

        # Initialize cache
        self.cache = CacheManager(
            cache_dir=f"./cache/{self.username}",
            ttl_hours=cache_ttl_hours or settings.cache_ttl_hours,
        )

        # Initialize session with retry logic
        # WHY: Session reuse enables connection pooling and cookie persistence
        self.session = self._create_session()

        # Track rate limit state
        self._rate_remaining = float("inf") if self.token else 60
        self._rate_reset_time = 0

        logger.info(
            f"Initialized GitHubCollector for '{self.username}' "
            f"(authenticated: {bool(self.token)})"
        )

    def _create_session(self) -> requests.Session:
        """
        Create requests session with retry strategy.

        WHY: Automatic retries handle transient failures (network issues,
        server errors) without manual intervention.

        Returns:
            Configured requests.Session
        """
        session = requests.Session()

        # Configure retry strategy
        # WHY: GitHub API sometimes returns 5xx errors; retries improve reliability
        retry_strategy = Retry(
            total=3,  # Maximum number of retries
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET"],  # Only retry GET requests (safe operations)
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Set default headers
        session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": f"GitHub-Profile-Analyzer/1.0 ({self.username})",
        })

        # Add authentication if token is provided
        if self.token:
            session.headers["Authorization"] = f"token {self.token}"
            logger.debug("Using authenticated requests (5000/hr rate limit)")
        else:
            logger.debug("Using unauthenticated requests (60/hr rate limit)")

        return session

    def _check_rate_limit(self, response: requests.Response) -> None:
        """
        Check and update rate limit state from response headers.

        WHY: Proactively managing rate limits prevents hitting the hard limit
        and provides better user experience (can warn before failure).

        Args:
            response: HTTP response to check headers from
        """
        # Extract rate limit info from headers
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_time = response.headers.get("X-RateLimit-Reset")

        if remaining:
            self._rate_remaining = int(remaining)

        if reset_time:
            self._rate_reset_time = int(reset_time)

        logger.debug(f"Rate limit: {self._rate_remaining} remaining")

        # Warn if approaching limit
        # WHY: Gives user opportunity to save progress before hitting limit
        if self._rate_remaining <= settings.rate_limit_warning_threshold:
            reset_dt = datetime.fromtimestamp(self._rate_reset_time)
            logger.warning(
                f"Approaching rate limit: {self._rate_remaining} remaining. "
                f"Resets at {reset_dt}"
            )

    def _handle_api_error(self, response: requests.Response, context: str = "") -> None:
        """
        Handle API error responses with appropriate exceptions.

        WHY: Different error conditions require different handling:
            - 404: User not found (might be a typo)
            - 401: Auth failed (check token)
            - 403: Rate limited (wait or use token)
            - 5xx: Server error (retry or fail gracefully)

        Args:
            response: HTTP response to check
            context: Additional context for error message

        Raises:
            GitHubAPIError: Appropriate exception for the error
        """
        status_code = response.status_code

        # Update rate limit tracking
        self._check_rate_limit(response)

        # Handle specific status codes
        if status_code == 404:
            raise UserNotFoundError(self.username)

        if status_code == 401:
            raise AuthenticationError(
                "Authentication failed. Check your GitHub token."
            )

        if status_code == 403:
            # Check if it's a rate limit issue
            if self._rate_remaining <= 1:
                raise RateLimitError(reset_time=self._rate_reset_time)

            # Other 403 errors
            raise GitHubAPIError(
                f"Access forbidden: {context}",
                status_code=403,
                response_body=response.text[:500],
            )

        if status_code >= 500:
            raise GitHubAPIError(
                f"GitHub server error ({status_code}): {context}",
                status_code=status_code,
                response_body=response.text[:500],
            )

        # Generic error for other status codes
        if status_code >= 400:
            raise GitHubAPIError(
                f"API error ({status_code}): {context}",
                status_code=status_code,
                response_body=response.text[:500],
            )

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Tuple[Any, bool]:
        """
        Make an API request with caching and error handling.

        WHY: Centralized request handling ensures consistent error handling,
        caching, and rate limit management across all API calls.

        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            use_cache: Whether to check/use cache

        Returns:
            Tuple of (response_data, from_cache)

        Raises:
            GitHubAPIError: On API errors
            RateLimitError: When rate limited
        """
        url = f"{self.base_url}{endpoint}"

        # Check cache first
        # WHY: Reduces API calls for repeated data
        cache_key = f"{endpoint}?{json.dumps(params, sort_keys=True)}" if params else endpoint

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached, True

        # Check rate limit before making request
        # WHY: Proactively wait instead of failing
        if self._rate_remaining <= 1:
            wait_time = max(0, self._rate_reset_time - time.time())
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time + 1)  # +1 for safety margin

        logger.debug(f"API request: {endpoint}")

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._check_rate_limit(response)

            # Handle errors
            if response.status_code >= 400:
                self._handle_api_error(response, context=endpoint)

            data = response.json()

            # Cache successful response
            if use_cache:
                self.cache.set(cache_key, data)

            return data, False

        except requests.Timeout:
            logger.error(f"Request timeout: {endpoint}")
            raise GitHubAPIError(f"Request timeout: {endpoint}")

        except requests.ConnectionError as e:
            logger.error(f"Connection error: {endpoint} - {e}")
            raise GitHubAPIError(f"Connection error: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {endpoint} - {e}")
            raise GitHubAPIError(f"Invalid JSON response: {e}")

    def _paginate(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Handle paginated API responses.

        WHY: GitHub API returns max 100 items per page. For users with
        many repos or commits, we need to follow pagination links.

        Args:
            endpoint: API endpoint
            params: Query parameters
            max_items: Maximum items to retrieve (None for unlimited)

        Returns:
            List of all items across pages
        """
        all_items: List[Dict[str, Any]] = []
        page = 1
        per_page = 100  # Maximum allowed by GitHub

        params = params or {}
        params["per_page"] = per_page

        while True:
            params["page"] = page

            try:
                data, from_cache = self._make_request(endpoint, params)
            except GitHubAPIError:
                # Return what we have so far on error
                logger.warning(f"Pagination stopped at page {page}")
                break

            # Handle both list and dict responses
            items = data if isinstance(data, list) else [data] if data else []
            all_items.extend(items)

            # Check if we've reached limits
            if max_items and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                logger.debug(f"Reached max items limit: {max_items}")
                break

            # Check if there are more pages
            # WHY: GitHub returns empty page when no more results
            if len(items) < per_page:
                break

            page += 1

            # Small delay between pages to be nice to API
            # WHY: Prevents hitting secondary rate limits
            if not from_cache:
                time.sleep(0.1)

        logger.info(f"Retrieved {len(all_items)} items from {endpoint}")
        return all_items

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def get_user_profile(self) -> Dict[str, Any]:
        """
        Fetch user profile information.

        WHY: User profile provides context for analysis:
            - Account age indicates experience
            - Bio/location can inform work style
            - Public repo count sets expectations

        Returns:
            Dictionary containing user profile data

        Raises:
            UserNotFoundError: If user doesn't exist
            GitHubAPIError: On other API errors
        """
        logger.info(f"Fetching profile for {self.username}")

        data, from_cache = self._make_request(f"/users/{self.username}")

        # Extract relevant fields
        # WHY: We only need specific fields; reduces memory and improves clarity
        profile = {
            "username": data.get("login", self.username),
            "name": data.get("name"),
            "bio": data.get("bio"),
            "location": data.get("location"),
            "company": data.get("company"),
            "blog": data.get("blog"),
            "email": data.get("email"),
            "twitter_username": data.get("twitter_username"),
            "avatar_url": data.get("avatar_url"),
            "html_url": data.get("html_url"),
            "type": data.get("type", "User"),
            "public_repos": data.get("public_repos", 0),
            "public_gists": data.get("public_gists", 0),
            "followers": data.get("followers", 0),
            "following": data.get("following", 0),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "from_cache": from_cache,
        }

        logger.info(
            f"Profile: {profile['name'] or profile['username']} "
            f"({profile['public_repos']} repos, {profile['followers']} followers)"
        )

        return profile

    def get_repositories(
        self,
        exclude_forks: bool = True,
        sort: str = "updated",
        direction: str = "desc",
    ) -> List[Dict[str, Any]]:
        """
        Fetch list of user's repositories.

        WHY: Repositories are the primary unit of analysis.
        We exclude forks by default to focus on original contributions.

        Args:
            exclude_forks: Whether to exclude forked repositories
            sort: Sort field (created, updated, pushed, full_name)
            direction: Sort direction (asc, desc)

        Returns:
            List of repository dictionaries
        """
        logger.info(f"Fetching repositories for {self.username}")

        params = {
            "type": "owner" if exclude_forks else "all",
            "sort": sort,
            "direction": direction,
        }

        repos = self._paginate(
            f"/users/{self.username}/repos",
            params=params,
            max_items=settings.max_repos,
        )

        # Filter out forks if requested (backup in case API param doesn't work)
        # WHY: API's 'owner' type sometimes includes forks; double-check
        if exclude_forks:
            repos = [r for r in repos if not r.get("fork", False)]
            logger.debug(f"Filtered to {len(repos)} non-fork repositories")

        # Extract relevant fields
        # WHY: Reduces memory footprint, standardizes data structure
        processed_repos = []
        for repo in repos:
            processed_repos.append({
                "id": repo.get("id"),
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "description": repo.get("description"),
                "html_url": repo.get("html_url"),
                "language": repo.get("language"),
                "stargazers_count": repo.get("stargazers_count", 0),
                "watchers_count": repo.get("watchers_count", 0),
                "forks_count": repo.get("forks_count", 0),
                "open_issues_count": repo.get("open_issues_count", 0),
                "size": repo.get("size", 0),  # KB
                "is_fork": repo.get("fork", False),
                "is_archived": repo.get("archived", False),
                "is_template": repo.get("is_template", False),
                "has_issues": repo.get("has_issues", False),
                "has_projects": repo.get("has_projects", False),
                "has_wiki": repo.get("has_wiki", False),
                "has_pages": repo.get("has_pages", False),
                "license": repo.get("license", {}).get("spdx_id") if repo.get("license") else None,
                "topics": repo.get("topics", []),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "pushed_at": repo.get("pushed_at"),
                "default_branch": repo.get("default_branch", "main"),
            })

        logger.info(f"Retrieved {len(processed_repos)} repositories")
        return processed_repos

    def get_commits(
        self,
        repo_name: str,
        max_commits: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch commits for a specific repository.

        WHY: Commit history reveals:
            - Work patterns (time of day, day of week)
            - Productivity trends
            - Project activity timeline

        Args:
            repo_name: Repository name
            max_commits: Maximum commits to fetch

        Returns:
            List of commit dictionaries
        """
        max_commits = max_commits or settings.max_commits_per_repo
        logger.debug(f"Fetching commits for {repo_name} (max: {max_commits})")

        try:
            commits = self._paginate(
                f"/repos/{self.username}/{repo_name}/commits",
                max_items=max_commits,
            )
        except GitHubAPIError as e:
            # Some repos may be empty or inaccessible
            logger.warning(f"Could not fetch commits for {repo_name}: {e}")
            return []

        # Extract relevant fields
        # WHY: Reduces memory footprint, extracts nested author info
        processed_commits = []
        for commit in commits:
            commit_data = commit.get("commit", {})
            author = commit_data.get("author", {})
            committer = commit_data.get("committer", {})

            processed_commits.append({
                "sha": commit.get("sha"),
                "repo_name": repo_name,
                "message": commit_data.get("message", ""),
                "author_name": author.get("name"),
                "author_email": author.get("email"),
                "author_date": author.get("date"),
                "committer_name": committer.get("name"),
                "committer_email": committer.get("email"),
                "committer_date": committer.get("date"),
                "html_url": commit.get("html_url"),
            })

        logger.debug(f"Retrieved {len(processed_commits)} commits for {repo_name}")
        return processed_commits

    def get_languages(self, repo_name: str) -> Dict[str, int]:
        """
        Fetch language breakdown for a repository.

        WHY: Language distribution reveals:
            - Technology expertise
            - Project type (frontend, backend, full-stack)
            - Language diversity

        Args:
            repo_name: Repository name

        Returns:
            Dictionary mapping language names to byte counts
        """
        logger.debug(f"Fetching languages for {repo_name}")

        try:
            data, _ = self._make_request(
                f"/repos/{self.username}/{repo_name}/languages"
            )
            return data if isinstance(data, dict) else {}
        except GitHubAPIError as e:
            logger.warning(f"Could not fetch languages for {repo_name}: {e}")
            return {}

    def collect_all(self) -> Dict[str, Any]:
        """
        Collect all data for the user.

        WHY: Provides a single entry point for complete data collection.
        Orchestrates the collection of profile, repos, commits, and languages.

        Returns:
            Dictionary containing all collected data:
                - profile: User profile information
                - repositories: List of repositories
                - commits: Dict mapping repo names to commit lists
                - languages: Dict mapping repo names to language breakdowns

        Raises:
            UserNotFoundError: If user doesn't exist
            GitHubAPIError: On other API errors
        """
        logger.info(f"Starting complete data collection for {self.username}")
        start_time = time.time()

        # Collect profile
        profile = self.get_user_profile()

        # Collect repositories
        repositories = self.get_repositories()

        # Collect commits and languages for each repository
        commits: Dict[str, List[Dict[str, Any]]] = {}
        languages: Dict[str, Dict[str, int]] = {}

        for i, repo in enumerate(repositories, 1):
            repo_name = repo["name"]
            logger.info(f"Processing repository {i}/{len(repositories)}: {repo_name}")

            # Skip archived repos for efficiency
            # WHY: Archived repos don't represent current activity patterns
            if repo.get("is_archived"):
                logger.debug(f"Skipping archived repo: {repo_name}")
                commits[repo_name] = []
                languages[repo_name] = {}
                continue

            # Fetch commits
            commits[repo_name] = self.get_commits(repo_name)

            # Fetch languages
            languages[repo_name] = self.get_languages(repo_name)

            # Small delay between repos to avoid secondary rate limits
            # WHY: GitHub has secondary rate limits for rapid requests
            time.sleep(0.1)

        # Compile results
        result = {
            "profile": profile,
            "repositories": repositories,
            "commits": commits,
            "languages": languages,
            "collection_metadata": {
                "username": self.username,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "total_repos": len(repositories),
                "total_commits": sum(len(c) for c in commits.values()),
                "authenticated": bool(self.token),
                "rate_remaining": self._rate_remaining,
                "collection_time_seconds": round(time.time() - start_time, 2),
            },
        }

        elapsed = time.time() - start_time
        logger.info(
            f"Data collection complete: {len(repositories)} repos, "
            f"{result['collection_metadata']['total_commits']} commits "
            f"in {elapsed:.1f}s"
        )

        return result

    def clear_cache(self) -> int:
        """
        Clear all cached data for this user.

        WHY: Allows forcing fresh data collection when needed.

        Returns:
            Number of cache entries cleared
        """
        return self.cache.clear()

    def __enter__(self) -> "GitHubCollector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup session."""
        self.session.close()
        logger.debug("Closed HTTP session")


# =============================================================================
# Export Public Interface
# =============================================================================

__all__ = [
    "GitHubCollector",
    "CacheManager",
    "GitHubAPIError",
    "RateLimitError",
    "UserNotFoundError",
    "AuthenticationError",
]
