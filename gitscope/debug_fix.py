import sys
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/ZBOOK/Downloads/Gitscope/gitscope")
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import get_db_stats, load_repos_df, load_commits_df, load_languages_df
from src.config import settings
from src.transformer import clean_commits, aggregate_by_time, compute_advanced_metrics

def main():
    db_path = settings.db_path
    print(f"Checking database at {db_path}")
    stats = get_db_stats(db_path)
    print(f"Stats: {stats}")
    
    username = "selmahacii"
    repos = load_repos_df(db_path, username)
    commits = load_commits_df(db_path, username)
    langs = load_languages_df(db_path, username)
    
    print(f"User {username}:")
    print(f"  Repos: {len(repos)}")
    print(f"  Commits: {len(commits)}")
    print(f"  Langs: {len(langs)}")
    
    # Run full pipeline
    from src.analytics import (
        analyze_languages,
        analyze_repositories,
        analyze_commit_messages,
        generate_developer_profile,
    )
    
    print("Testing pipeline...")
    try:
        clean_df = clean_commits(commits)
        aggregations = aggregate_by_time(clean_df)
        metrics = compute_advanced_metrics(clean_df)
        
        lang_analysis = analyze_languages(langs)
        repo_analysis = analyze_repositories(repos)
        commit_analysis = analyze_commit_messages(clean_df)
        
        profile = generate_developer_profile(
            metrics=metrics,
            languages=lang_analysis,
            repos=repo_analysis,
            commits=commit_analysis,
        )
        print("Pipeline Success!")
        print(f"Developer Type: {profile['labels']['developer_type']}")
        print(f"Overall Score: {profile['scores']['overall']}")
        
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
