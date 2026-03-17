"""
Data Pipeline Orchestration Module.

This module provides the main orchestration layer for the GitHub
Profile Analyzer, coordinating data collection, transformation,
analysis, and insight generation.

The pipeline follows the architecture:
    GitHub API → Data Collection → SQLite Storage → pandas Transformations 
    → Analytics & Metrics → Plotly Visualizations → GLM Insights

Features:
    - End-to-end pipeline execution
    - Incremental updates with caching
    - Error handling and recovery
    - Progress tracking and logging

Example:
    >>> from src.pipeline import GitHubPipeline
    >>> pipeline = GitHubPipeline()
    >>> 
    >>> # Run full analysis
    >>> results = await pipeline.analyze_user("octocat")
    >>> 
    >>> # Generate report
    >>> report = pipeline.generate_report(results)
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .analytics import GitHubAnalytics
from .collector import GitHubCollector
from .config import settings
from .insights import InsightGenerator
from .storage import Database, init_db
from .transformer import DataTransformer
from .visualizations import GitHubVisualizer


console = Console()


@dataclass
class PipelineResult:
    """Results from a pipeline execution."""
    username: str
    success: bool
    user_data: Optional[dict] = None
    cleaned_data: Optional[dict] = None
    analysis_results: Optional[dict] = None
    insights: Optional[dict] = None
    visualizations: Optional[dict] = None
    errors: list = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GitHubPipeline:
    """
    Main orchestration pipeline for GitHub profile analysis.
    
    Coordinates all components of the analysis system:
    - Data collection from GitHub API
    - Storage in SQLite database
    - Data transformation with pandas
    - Analytics and metrics calculation
    - Visualization generation
    - LLM insight generation
    
    Attributes:
        db: Database instance
        collector: GitHub API collector
        transformer: Data transformer
        analytics: Analytics engine
        visualizer: Visualization generator
        insight_generator: LLM insight generator
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        github_token: Optional[str] = None,
        zhipuai_key: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            db_path: Database path (falls back to settings)
            github_token: GitHub token (falls back to settings)
            zhipuai_key: ZhipuAI key (falls back to settings)
        """
        # Initialize components
        self.db = init_db(db_path or settings.db_path)
        self.collector = GitHubCollector(token=github_token)
        self.transformer = DataTransformer()
        self.analytics = GitHubAnalytics()
        self.visualizer = GitHubVisualizer()
        self.insight_generator = InsightGenerator(api_key=zhipuai_key)
        
        self._async_collector = None
    
    async def _ensure_collector(self) -> GitHubCollector:
        """Ensure async collector is initialized."""
        if self._async_collector is None:
            self._async_collector = GitHubCollector(
                token=self.collector.token,
                use_cache=self.collector.use_cache,
            )
            self._async_collector = await self._async_collector.__aenter__()
        return self._async_collector
    
    async def collect_user_data(
        self, 
        username: str,
        force_refresh: bool = False,
    ) -> dict:
        """
        Collect data from GitHub API.
        
        Args:
            username: GitHub username
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            Raw data from GitHub API
        """
        # Check cache first
        if not force_refresh:
            cached = self.db.get_analysis_result(
                username, "raw_data", max_age_hours=settings.cache_ttl_hours
            )
            if cached:
                console.print("[dim]Using cached data...[/dim]")
                return cached
        
        # Fetch fresh data
        collector = await self._ensure_collector()
        raw_data = await collector.collect_user_data(
            username,
            include_commits=True,
            include_languages=True,
            include_events=True,
        )
        
        # Cache the raw data
        self.db.store_full_user_data(raw_data)
        self.db.store_analysis_result(username, "raw_data", raw_data)
        
        return raw_data
    
    def transform_data(self, raw_data: dict) -> dict:
        """
        Transform raw data through the cleaning pipeline.
        
        Args:
            raw_data: Raw data from GitHub API
            
        Returns:
            Dictionary of cleaned DataFrames
        """
        console.print("[bold]Transforming data...[/bold]")
        
        cleaned = {}
        
        # Transform user data
        if raw_data.get("user"):
            cleaned["user_df"] = self.transformer.transform_user_data(raw_data["user"])
            console.print(f"  ✓ User data: {len(cleaned['user_df'])} row")
        
        # Transform repositories
        if raw_data.get("repositories"):
            cleaned["repos_df"] = self.transformer.transform_repositories(raw_data["repositories"])
            console.print(f"  ✓ Repositories: {len(cleaned['repos_df'])} repos")
        
        # Transform commits
        if raw_data.get("commits"):
            cleaned["commits_df"] = self.transformer.transform_commits(raw_data["commits"])
            console.print(f"  ✓ Commits: {len(cleaned['commits_df'])} commits")
        
        # Transform languages
        if raw_data.get("languages"):
            cleaned["languages_df"] = self.transformer.transform_languages(raw_data["languages"])
            console.print(f"  ✓ Languages: {len(cleaned['languages_df'])} records")
        
        # Transform events
        if raw_data.get("events"):
            cleaned["events_df"] = self.transformer.transform_events(raw_data["events"])
            console.print(f"  ✓ Events: {len(cleaned['events_df'])} events")
        
        # Create aggregations
        if cleaned.get("commits_df") is not None and not cleaned["commits_df"].empty:
            cleaned["commits_by_time"] = self.transformer.aggregate_commits_by_time(cleaned["commits_df"])
            cleaned["commits_by_repo"] = self.transformer.aggregate_commits_by_repo(cleaned["commits_df"])
        
        if cleaned.get("languages_df") is not None and not cleaned["languages_df"].empty:
            cleaned["languages_agg"] = self.transformer.aggregate_languages(cleaned["languages_df"])
        
        return cleaned
    
    def analyze_data(
        self, 
        username: str,
        cleaned_data: dict,
    ) -> dict:
        """
        Run analytics on transformed data.
        
        Args:
            username: GitHub username
            cleaned_data: Cleaned data from transform_data()
            
        Returns:
            Analysis results dictionary
        """
        console.print("[bold]Running analytics...[/bold]")
        
        commits_df = cleaned_data.get("commits_df")
        repos_df = cleaned_data.get("repos_df")
        languages_df = cleaned_data.get("languages_df")
        
        # Run comprehensive analysis
        results = self.analytics.analyze_user(
            username,
            commits_df if commits_df is not None else commits_df.__class__(),
            repos_df if repos_df is not None else repos_df.__class__(),
            languages_df,
        )
        
        # Add productivity metrics
        if commits_df is not None and not commits_df.empty:
            results["productivity"] = self.transformer.calculate_productivity_metrics(commits_df)
            results["time_series"] = self.transformer.calculate_time_series_features(commits_df)
        
        # Cache results
        self.db.store_analysis_result(username, "full_analysis", results)
        
        return results
    
    def generate_visualizations(
        self, 
        cleaned_data: dict,
        analysis_results: dict,
        output_dir: Optional[Path] = None,
    ) -> dict:
        """
        Generate visualizations from analysis results.
        
        Args:
            cleaned_data: Cleaned data
            analysis_results: Analysis results
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of visualization figures
        """
        console.print("[bold]Generating visualizations...[/bold]")
        
        visualizations = {}
        
        commits_df = cleaned_data.get("commits_df")
        repos_df = cleaned_data.get("repos_df")
        languages_df = cleaned_data.get("languages_df")
        
        # Commit timeline
        if commits_df is not None and not commits_df.empty:
            anomalies = analysis_results.get("anomalies", [])
            visualizations["commit_timeline"] = self.visualizer.commit_timeline(
                commits_df, show_anomalies=True, anomalies=anomalies
            )
            visualizations["commit_velocity"] = self.visualizer.commit_velocity_chart(commits_df)
            visualizations["commit_sizes"] = self.visualizer.commit_size_distribution(commits_df)
            visualizations["hourly_heatmap"] = self.visualizer.hourly_activity_heatmap(commits_df)
            visualizations["work_radar"] = self.visualizer.work_pattern_radar(commits_df)
        
        # Repository visualizations
        if repos_df is not None and not repos_df.empty:
            visualizations["repo_stats"] = self.visualizer.repository_stats_chart(repos_df)
            visualizations["repo_status"] = self.visualizer.repo_status_pie(repos_df)
        
        # Language visualizations
        if languages_df is not None and not languages_df.empty:
            visualizations["languages"] = self.visualizer.language_distribution_chart(languages_df)
        
        # Score radar
        scores = analysis_results.get("scores", {})
        if scores:
            visualizations["scores"] = self.visualizer.scores_radar(scores)
        
        # Save visualizations if output directory specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in visualizations.items():
                path = self.visualizer.save_figure(fig, name, output_dir, format="html")
                console.print(f"  ✓ Saved: {path}")
        
        return visualizations
    
    def generate_insights(
        self, 
        analysis_results: dict,
        user_data: Optional[dict] = None,
    ) -> dict:
        """
        Generate LLM-powered insights.
        
        Args:
            analysis_results: Analysis results
            user_data: Optional user profile data
            
        Returns:
            Dictionary of generated insights
        """
        console.print("[bold]Generating AI insights...[/bold]")
        
        insights = self.insight_generator.generate_full_report(analysis_results, user_data)
        
        console.print(f"  ✓ Generated {len(insights.get('insights', {}))} insight sections")
        
        return insights
    
    async def analyze_user(
        self, 
        username: str,
        force_refresh: bool = False,
        include_visualizations: bool = True,
        include_insights: bool = True,
        output_dir: Optional[Path] = None,
    ) -> PipelineResult:
        """
        Run the complete analysis pipeline for a GitHub user.
        
        This is the main entry point for running a full analysis.
        
        Args:
            username: GitHub username
            force_refresh: Skip cache and fetch fresh data
            include_visualizations: Generate visualizations
            include_insights: Generate LLM insights
            output_dir: Directory to save outputs
            
        Returns:
            PipelineResult with all analysis data
        """
        start_time = datetime.now()
        errors = []
        
        console.print(Panel(
            f"[bold]GitHub Profile Analyzer[/bold]\n"
            f"Analyzing: [cyan]{username}[/cyan]",
            expand=False,
        ))
        
        result = PipelineResult(username=username, success=False)
        
        try:
            # Step 1: Collect data
            console.print("\n[bold blue]Step 1: Data Collection[/bold blue]")
            raw_data = await self.collect_user_data(username, force_refresh)
            result.user_data = raw_data
            
            # Step 2: Transform data
            console.print("\n[bold blue]Step 2: Data Transformation[/bold blue]")
            cleaned_data = self.transform_data(raw_data)
            result.cleaned_data = {
                k: v.to_dict() if hasattr(v, 'to_dict') else v 
                for k, v in cleaned_data.items()
            }
            
            # Step 3: Run analytics
            console.print("\n[bold blue]Step 3: Analytics[/bold blue]")
            analysis_results = self.analyze_data(username, cleaned_data)
            result.analysis_results = analysis_results
            
            # Step 4: Generate visualizations
            if include_visualizations:
                console.print("\n[bold blue]Step 4: Visualizations[/bold blue]")
                result.visualizations = self.generate_visualizations(
                    cleaned_data, analysis_results, output_dir
                )
            
            # Step 5: Generate insights
            if include_insights:
                console.print("\n[bold blue]Step 5: AI Insights[/bold blue]")
                result.insights = self.generate_insights(
                    analysis_results, raw_data.get("user")
                )
            
            result.success = True
            
        except Exception as e:
            errors.append(str(e))
            console.print(f"[bold red]Error: {e}[/bold red]")
            import traceback
            traceback.print_exc()
        
        result.errors = errors
        result.execution_time = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: PipelineResult) -> None:
        """Print a summary of the pipeline results."""
        console.print("\n")
        
        # Create summary table
        table = Table(title="Analysis Summary", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Username", result.username)
        table.add_row("Status", "✅ Success" if result.success else "❌ Failed")
        table.add_row("Execution Time", f"{result.execution_time:.2f}s")
        
        if result.analysis_results:
            metrics = result.analysis_results.get("metrics", {})
            velocity = metrics.get("velocity", {})
            repos = metrics.get("repositories", {})
            
            table.add_row("Total Commits", str(velocity.get("total_commits", "N/A")))
            table.add_row("Total Repositories", str(repos.get("total_repos", "N/A")))
            table.add_row("Total Stars", str(repos.get("total_stars", "N/A")))
            
            scores = result.analysis_results.get("scores", {})
            if scores:
                table.add_row("Overall Score", f"{scores.get('overall', 0):.1f}/100")
        
        console.print(table)
    
    def generate_report(
        self, 
        result: PipelineResult,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate a comprehensive markdown report.
        
        Args:
            result: Pipeline result
            output_path: Path to save report
            
        Returns:
            Report content as string
        """
        report_lines = [
            f"# GitHub Profile Analysis: {result.username}",
            f"\n*Generated on {result.timestamp}*",
            f"\n---\n",
        ]
        
        # Add insights
        if result.insights:
            insights = result.insights.get("insights", {})
            
            if insights.get("profile_summary"):
                report_lines.extend([
                    "## Profile Summary\n",
                    insights["profile_summary"],
                    "\n",
                ])
            
            if insights.get("technical_summary"):
                report_lines.extend([
                    "## Technical Profile\n",
                    insights["technical_summary"],
                    "\n",
                ])
            
            if insights.get("recommendations"):
                report_lines.extend([
                    "## Recommendations\n",
                    insights["recommendations"],
                    "\n",
                ])
            
            if insights.get("career_insights"):
                report_lines.extend([
                    "## Career Insights\n",
                    insights["career_insights"],
                    "\n",
                ])
        
        # Add metrics summary
        if result.analysis_results:
            scores = result.analysis_results.get("scores", {})
            metrics = result.analysis_results.get("metrics", {})
            
            report_lines.extend([
                "## Key Metrics\n",
                "| Category | Score |",
                "|----------|-------|",
            ])
            
            for name, score in scores.items():
                report_lines.append(f"| {name.title()} | {score:.1f}/100 |")
            
            report_lines.append("\n")
            
            # Add productivity metrics
            if metrics.get("velocity"):
                velocity = metrics["velocity"]
                report_lines.extend([
                    "### Activity Metrics\n",
                    f"- Average commits per day: {velocity.get('avg_commits_per_day', 0):.2f}",
                    f"- Total commits analyzed: {velocity.get('total_commits', 'N/A')}",
                    f"- Velocity trend: {velocity.get('velocity_trend', 'N/A')}",
                    "\n",
                ])
        
        # Add detected anomalies
        if result.analysis_results and result.analysis_results.get("anomalies"):
            report_lines.extend([
                "## Detected Patterns\n",
                "### Anomalies\n",
            ])
            for anomaly in result.analysis_results["anomalies"][:5]:
                report_lines.append(f"- **{anomaly.anomaly_type}**: {anomaly.description}")
            report_lines.append("\n")
        
        # Add insights from analysis
        if result.analysis_results and result.analysis_results.get("insights"):
            report_lines.extend([
                "## Quick Insights\n",
            ])
            for insight in result.analysis_results["insights"]:
                report_lines.append(f"- {insight}")
            report_lines.append("\n")
        
        report_content = "\n".join(report_lines)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_content)
            console.print(f"Report saved to: {output_path}")
        
        return report_content
    
    def save_results(
        self, 
        result: PipelineResult,
        output_dir: Path,
    ) -> dict:
        """
        Save all results to files.
        
        Args:
            result: Pipeline result
            output_dir: Output directory
            
        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save analysis results
        if result.analysis_results:
            path = output_dir / f"{result.username}_analysis.json"
            with open(path, "w") as f:
                # Convert dataclass objects to dicts
                def to_dict(obj):
                    if hasattr(obj, '__dict__'):
                        return {k: to_dict(v) for k, v in obj.__dict__.items()}
                    elif isinstance(obj, list):
                        return [to_dict(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: to_dict(v) for k, v in obj.items()}
                    else:
                        return obj
                
                json.dump(to_dict(result.analysis_results), f, indent=2, default=str)
            saved_files["analysis"] = path
        
        # Save insights
        if result.insights:
            path = output_dir / f"{result.username}_insights.json"
            with open(path, "w") as f:
                json.dump(result.insights, f, indent=2, default=str)
            saved_files["insights"] = path
        
        # Save report
        report_path = output_dir / f"{result.username}_report.md"
        self.generate_report(result, report_path)
        saved_files["report"] = report_path
        
        return saved_files


async def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Profile Analyzer")
    parser.add_argument("username", help="GitHub username to analyze")
    parser.add_argument("--force-refresh", action="store_true", help="Skip cache")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--no-visualizations", action="store_true", help="Skip visualizations")
    parser.add_argument("--no-insights", action="store_true", help="Skip LLM insights")
    
    args = parser.parse_args()
    
    pipeline = GitHubPipeline()
    
    result = await pipeline.analyze_user(
        args.username,
        force_refresh=args.force_refresh,
        include_visualizations=not args.no_visualizations,
        include_insights=not args.no_insights,
        output_dir=Path(args.output_dir) / "visualizations",
    )
    
    # Save results
    saved = pipeline.save_results(result, Path(args.output_dir))
    
    console.print("\n[bold green]Analysis complete![/bold green]")
    for name, path in saved.items():
        console.print(f"  📄 {name}: {path}")


if __name__ == "__main__":
    asyncio.run(main())
