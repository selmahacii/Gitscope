import sys
from pathlib import Path

PROJECT_ROOT = Path("c:/Users/ZBOOK/Downloads/Gitscope/gitscope")
sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import get_db_stats, load_repos_df, load_commits_df, load_languages_df
from src.config import settings

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
    
    if not repos.empty:
        from src.analytics import analyze_repositories
        print("Testing analyze_repositories...")
        try:
            res = analyze_repositories(repos)
            print("Success!")
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
