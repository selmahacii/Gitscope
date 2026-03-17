"""
Unit Tests for Analytics Module.

This module tests the core analytics functions:
    - Language analysis
    - Repository analysis
    - Commit message analysis
    - Developer profile generation

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates testing best practices
    - Shows pytest proficiency
    - Tests edge cases and error handling
    - Uses fixtures for test data

Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.analytics import (
    analyze_languages,
    analyze_repositories,
    analyze_commit_messages,
    generate_developer_profile,
    _calculate_consistency_score,
    _classify_developer_type,
    _classify_activity_level,
    _calculate_impact_score,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_languages_df() -> pd.DataFrame:
    """Create sample languages DataFrame for testing."""
    return pd.DataFrame({
        "repo_name": ["repo1", "repo1", "repo2", "repo2", "repo3"],
        "language": ["Python", "JavaScript", "Python", "HTML", "TypeScript"],
        "bytes_count": [10000, 2000, 8000, 1000, 5000],
        "percentage": [83.3, 16.7, 88.9, 11.1, 100.0],
    })


@pytest.fixture
def sample_repos_df() -> pd.DataFrame:
    """Create sample repositories DataFrame for testing."""
    return pd.DataFrame({
        "name": ["project-alpha", "project-beta", "project-gamma"],
        "full_name": ["user/project-alpha", "user/project-beta", "user/project-gamma"],
        "primary_language": ["Python", "JavaScript", "Python"],
        "stargazers_count": [100, 50, 25],
        "forks_count": [20, 10, 5],
        "open_issues_count": [5, 2, 1],
        "is_fork": [False, False, True],
        "is_archived": [False, True, False],
        "size": [1000, 500, 200],
        "license": ["MIT", "Apache-2.0", None],
        "topics": [["api", "python"], ["frontend"], ["test"]],
    })


@pytest.fixture
def sample_commits_df() -> pd.DataFrame:
    """Create sample commits DataFrame for testing."""
    dates = [
        datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 16, 9, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 16, 22, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 20, 11, 0, tzinfo=timezone.utc),
    ]

    return pd.DataFrame({
        "sha": ["abc1", "abc2", "abc3", "abc4", "abc5"],
        "repo_name": ["repo1", "repo1", "repo2", "repo1", "repo3"],
        "message": [
            "feat: add new feature",
            "fix: resolve bug",
            "update docs",
            "x",
            "refactor: clean up code"
        ],
        "message_length": [22, 16, 11, 1, 23],
        "author_date": dates,
        "hour_of_day": [10, 14, 9, 22, 11],
        "day_of_week": [0, 0, 1, 1, 5],  # Mon, Mon, Tue, Tue, Sat
        "is_weekend": [False, False, False, False, True],
        "is_conventional": [True, True, False, False, True],
        "time_of_day": ["morning", "afternoon", "morning", "night", "morning"],
        "commit_size": ["medium", "short", "short", "tiny", "medium"],
    })


# =============================================================================
# Language Analysis Tests
# =============================================================================

class TestAnalyzeLanguages:
    """Tests for analyze_languages function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        result = analyze_languages(pd.DataFrame())
        assert result["language_count"] == 0
        assert result["primary_language"] is None
        assert result["diversity_score"] == 0.0

    def test_single_language(self, sample_languages_df):
        """Test with single language data."""
        # Filter to only Python
        df = sample_languages_df[sample_languages_df["language"] == "Python"]
        result = analyze_languages(df)

        assert result["language_count"] == 1
        assert result["primary_language"] == "Python"
        assert result["diversity_score"] == 0.0  # No diversity with single language

    def test_multiple_languages(self, sample_languages_df):
        """Test with multiple languages."""
        result = analyze_languages(sample_languages_df)

        assert result["language_count"] == 4  # Python, JavaScript, HTML, TypeScript
        assert result["primary_language"] == "Python"  # Most bytes
        assert result["diversity_score"] > 0  # Should have some diversity
        assert len(result["top_languages"]) <= 10

    def test_diversity_score_calculation(self, sample_languages_df):
        """Test diversity score is calculated correctly."""
        result = analyze_languages(sample_languages_df)

        # Diversity score should be between 0 and max theoretical
        assert 0 <= result["diversity_score"] <= 5

        # Should have diversity level classification
        assert result["diversity_level"] in ["specialized", "moderate", "diverse"]

    def test_category_breakdown(self, sample_languages_df):
        """Test language category breakdown."""
        result = analyze_languages(sample_languages_df)

        assert "category_breakdown" in result
        assert isinstance(result["category_breakdown"], dict)


# =============================================================================
# Repository Analysis Tests
# =============================================================================

class TestAnalyzeRepositories:
    """Tests for analyze_repositories function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        result = analyze_repositories(pd.DataFrame())
        assert result["total_repos"] == 0
        assert result["total_stars"] == 0

    def test_basic_metrics(self, sample_repos_df):
        """Test basic repository metrics."""
        result = analyze_repositories(sample_repos_df)

        assert result["total_repos"] == 3
        assert result["total_stars"] == 175  # 100 + 50 + 25
        assert result["total_forks"] == 35   # 20 + 10 + 5
        assert result["avg_stars"] == 175 / 3

    def test_activity_analysis(self, sample_repos_df):
        """Test activity analysis."""
        result = analyze_repositories(sample_repos_df)

        assert result["active_repos"] == 2  # 2 non-archived
        assert result["archived_repos"] == 1
        assert result["fork_count"] == 1
        assert result["fork_ratio"] == pytest.approx(33.3, rel=0.1)

    def test_top_repositories(self, sample_repos_df):
        """Test top repositories identification."""
        result = analyze_repositories(sample_repos_df)

        assert "top_repositories" in result
        assert len(result["top_repositories"]) <= 5

        # First should be most starred
        if result["top_repositories"]:
            assert result["top_repositories"][0]["name"] == "project-alpha"

    def test_license_distribution(self, sample_repos_df):
        """Test license distribution calculation."""
        result = analyze_repositories(sample_repos_df)

        assert "license_distribution" in result
        assert result["license_distribution"]["MIT"] == 1
        assert result["license_distribution"]["Apache-2.0"] == 1


# =============================================================================
# Commit Message Analysis Tests
# =============================================================================

class TestAnalyzeCommitMessages:
    """Tests for analyze_commit_messages function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        result = analyze_commit_messages(pd.DataFrame())
        assert result["total_commits"] == 0
        assert result["quality_score"] == 0

    def test_basic_metrics(self, sample_commits_df):
        """Test basic commit metrics."""
        result = analyze_commit_messages(sample_commits_df)

        assert result["total_commits"] == 5
        assert result["avg_message_length"] > 0
        assert "quality_score" in result

    def test_conventional_commits(self, sample_commits_df):
        """Test conventional commit detection."""
        result = analyze_commit_messages(sample_commits_df)

        # 3 out of 5 are conventional
        assert result["conventional_commits"] == 3
        assert result["conventional_commit_ratio"] == 60.0

    def test_single_word_commits(self, sample_commits_df):
        """Test single-word commit detection."""
        result = analyze_commit_messages(sample_commits_df)

        # 1 commit has single word ("x")
        assert result["single_word_commits"] == 1
        assert result["single_word_ratio"] == 20.0

    def test_quality_score_bounds(self, sample_commits_df):
        """Test quality score is within bounds."""
        result = analyze_commit_messages(sample_commits_df)

        assert 0 <= result["quality_score"] <= 100

    def test_time_distribution(self, sample_commits_df):
        """Test time of day distribution."""
        result = analyze_commit_messages(sample_commits_df)

        assert "time_distribution" in result
        # Morning has 3 commits (most)
        assert result["time_distribution"]["morning"] == 3


# =============================================================================
# Developer Profile Tests
# =============================================================================

class TestGenerateDeveloperProfile:
    """Tests for generate_developer_profile function."""

    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics for profile generation."""
        return {
            "total_commits": 100,
            "total_repos": 5,
            "active_days": 30,
            "consistency_score": 75.0,
            "longest_streak": 14,
            "current_streak": 5,
            "peak_hour": 14,
            "peak_day": 2,  # Wednesday
            "preferred_time": "afternoon",
            "work_style": "balanced",
            "productivity_trend": "increasing",
            "burnout_periods": [],
            "weekend_commit_ratio": 10.0,
        }

    @pytest.fixture
    def sample_languages(self):
        """Sample language analysis."""
        return {
            "primary_language": "Python",
            "language_count": 3,
            "diversity_score": 0.8,
            "top_languages": [
                {"language": "Python", "percentage": 60.0},
                {"language": "JavaScript", "percentage": 30.0},
                {"language": "HTML", "percentage": 10.0},
            ],
            "primary_category": "backend",
        }

    @pytest.fixture
    def sample_repos(self):
        """Sample repository analysis."""
        return {
            "total_repos": 5,
            "total_stars": 150,
            "active_repos": 4,
            "top_repositories": [],
        }

    @pytest.fixture
    def sample_commits(self):
        """Sample commit analysis."""
        return {
            "total_commits": 100,
            "quality_score": 65.0,
            "conventional_commit_ratio": 40.0,
        }

    def test_profile_generation(
        self,
        sample_metrics,
        sample_languages,
        sample_repos,
        sample_commits,
    ):
        """Test basic profile generation."""
        profile = generate_developer_profile(
            metrics=sample_metrics,
            languages=sample_languages,
            repos=sample_repos,
            commits=sample_commits,
        )

        assert "labels" in profile
        assert "scores" in profile
        assert "strengths" in profile
        assert "improvements" in profile
        assert "summary" in profile

    def test_labels_generated(
        self,
        sample_metrics,
        sample_languages,
        sample_repos,
        sample_commits,
    ):
        """Test that all labels are generated."""
        profile = generate_developer_profile(
            metrics=sample_metrics,
            languages=sample_languages,
            repos=sample_repos,
            commits=sample_commits,
        )

        labels = profile["labels"]
        assert "developer_type" in labels
        assert "activity_level" in labels
        assert "experience_level" in labels
        assert "work_style" in labels

    def test_scores_generated(
        self,
        sample_metrics,
        sample_languages,
        sample_repos,
        sample_commits,
    ):
        """Test that all scores are generated."""
        profile = generate_developer_profile(
            metrics=sample_metrics,
            languages=sample_languages,
            repos=sample_repos,
            commits=sample_commits,
        )

        scores = profile["scores"]
        assert "consistency" in scores
        assert "quality" in scores
        assert "impact" in scores
        assert "diversity" in scores
        assert "overall" in scores

        # Overall should be average of other scores
        expected_overall = (
            scores["consistency"] +
            scores["quality"] +
            scores["impact"] +
            scores["diversity"]
        ) / 4
        assert scores["overall"] == pytest.approx(expected_overall, rel=0.1)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_classify_developer_type_backend(self):
        """Test backend developer classification."""
        dev_type = _classify_developer_type(
            metrics={"total_commits": 100},
            languages={"diversity_score": 0.5, "category_breakdown": {"backend": 70, "frontend": 10}},
            repos={"total_repos": 10, "total_stars": 50},
        )
        assert dev_type == "Backend Developer"

    def test_classify_developer_type_polyglot(self):
        """Test polyglot classification."""
        dev_type = _classify_developer_type(
            metrics={"total_commits": 100},
            languages={"diversity_score": 2.5, "category_breakdown": {}},
            repos={"total_repos": 10, "total_stars": 50},
        )
        assert dev_type == "Polyglot"

    def test_classify_activity_level_high(self):
        """Test high activity classification."""
        level = _classify_activity_level({
            "consistency_score": 75,
            "active_days": 30,
            "total_commits": 100,
        })
        assert level in ["High", "Very High", "Extremely High"]

    def test_calculate_impact_score(self):
        """Test impact score calculation."""
        score = _calculate_impact_score({
            "total_stars": 1000,
            "total_forks": 200,
        })

        # Logarithmic scale: 1000 stars + 200 forks should give decent score
        assert 0 <= score <= 100
        assert score > 30  # Should be significant


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframes(self):
        """Test all functions with empty DataFrames."""
        empty_df = pd.DataFrame()

        assert analyze_languages(empty_df)["language_count"] == 0
        assert analyze_repositories(empty_df)["total_retries:
            analyze_languages(empty_df)
            analyze_repositories(empty_df)
            analyze_commit_messages(empty_df)
        except Exception as e:
            pytest.fail(f"Should handle empty DataFrames gracefully: {e}")

    def test_missing_columns(self):
        """Test handling of missing columns."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})

        # Should not crash with missing columns
        result = analyze_languages(df)
        assert result is not None

    def test_nan_values(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            "language": ["Python", "JavaScript", None],
            "bytes_count": [100, 200, np.nan],
        })

        result = analyze_languages(df)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
