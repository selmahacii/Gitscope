# GitScope

**See the full scope of any GitHub developer.**

A production-grade data pipeline for analyzing GitHub profiles, designed as a **Data Analyst portfolio project for 2026**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-Powered%20Insights-green.svg" alt="AI">
</p>

## 🎯 Project Overview

This tool provides comprehensive analysis of GitHub profiles, transforming raw API data into actionable insights through:

- **Data Collection**: GitHub REST API with rate limiting and caching
- **Storage**: SQLite with SQLAlchemy ORM
- **Transformation**: pandas DataFrame pipeline
- **Analytics**: Language, repository, and commit pattern analysis
- **Visualization**: Interactive Plotly charts
- **AI Insights**: LLM-powered analysis with ZhipuAI GLM

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   GitHub API    │────▶│   Collector  │────▶│    SQLite DB    │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                     │
        ┌────────────────────────────────────────────┘
        ▼
┌───────────────┐     ┌──────────────┐     ┌─────────────────┐
│  DataLoader   │────▶│ Transformer  │────▶│    Analytics    │
└───────────────┘     └──────────────┘     └─────────────────┘
                                                   │
        ┌──────────────────────────────────────────┘
        ▼
┌───────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Visualizations│────▶│    Insights  │────▶│   Streamlit UI  │
└───────────────┘     └──────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- GitHub Personal Access Token (optional, increases rate limit)
- ZhipuAI API Key (optional, for AI insights)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/github-analyzer.git
cd github-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Command Line Usage

```bash
# Basic analysis
python main.py octocat

# Force fresh data (skip cache)
python main.py octocat --force-refresh

# Skip AI insights
python main.py octocat --no-ai

# Save results to JSON
python main.py octocat --output results.json

# With GitHub token
python main.py octocat --token ghp_xxxx
```

### Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## 📊 Features

### Activity Analysis
- Commit frequency patterns
- Active days and streaks
- Peak hours and preferred times
- Productivity trends
- Burnout period detection

### Language Analysis
- Primary language identification
- Shannon entropy diversity score
- Technology category breakdown
- Language evolution over time

### Repository Analysis
- Star and fork metrics
- Active vs archived ratio
- License distribution
- Topic frequency analysis

### Commit Quality Analysis
- Conventional commit detection
- Message length statistics
- Single-word commit ratio
- Quality score calculation

### AI-Powered Insights
- Developer profile summary
- Strengths and improvements
- Work style analysis
- Fun facts generation

## 📁 Project Structure

```
gitscope/
├── app/
│   └── streamlit_app.py      # Streamlit dashboard
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Pydantic Settings configuration
│   ├── collector.py          # GitHub API collector
│   ├── storage.py            # SQLAlchemy ORM models
│   ├── transformer.py        # Data transformation pipeline
│   ├── analytics.py          # Analysis functions
│   ├── visualizations.py     # Plotly charts
│   └── insights.py           # LLM integration
├── tests/
│   └── test_analytics.py     # Unit tests
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
├── .env.example              # Environment template
├── ANALYSIS.md               # Methodology documentation
└── README.md                 # This file
```

## ⚙️ Configuration

Configuration is managed via environment variables and `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub Personal Access Token | None |
| `ZHIPUAI_API_KEY` | ZhipuAI API key for AI insights | None |
| `DB_PATH` | SQLite database path | `./data/github_analyzer.db` |
| `CACHE_TTL_HOURS` | Cache TTL in hours | 24 |
| `MAX_REPOS` | Max repos to analyze | 50 |
| `MAX_COMMITS_PER_REPO` | Max commits per repo | 100 |
| `ANOMALY_THRESHOLD` | Z-score threshold | 2.0 |
| `BURNOUT_THRESHOLD_DAYS` | Days for burnout detection | 7 |

## 🐳 Docker Deployment

```bash
# Build image
docker build -t github-analyzer .

# Run container
docker run -p 8501:8501 \
  -e GITHUB_TOKEN=ghp_xxx \
  -e ZHIPUAI_API_KEY=xxx \
  github-analyzer

# Using Docker Compose
docker-compose up
```

## 📈 Metrics Explained

### Consistency Score (0-100)
Measures regularity of contributions:
- Lower coefficient of variation = higher score
- Considers both regularity and coverage

### Quality Score (0-100)
Measures commit message quality:
- Conventional commits bonus
- Appropriate message length
- Low single-word/empty commit ratio

### Impact Score (0-100)
Measures real-world influence:
- Logarithmic scale for stars/forks
- Prevents outlier domination

### Diversity Score (0-100)
Measures technology breadth:
- Based on Shannon entropy
- Normalized for interpretability

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analytics.py -v
```

## 📝 Methodology

See [ANALYSIS.md](ANALYSIS.md) for detailed methodology documentation including:
- Statistical formulas for all metrics
- Anomaly detection algorithm
- Scoring methodology
- Sample analyses of real profiles

## 🔒 Security Considerations

- API keys are loaded from environment variables (never hardcoded)
- Non-root Docker user for container security
- Rate limiting respects GitHub API limits
- Caching reduces API exposure

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

*Built with 🔭 GitScope - Python, pandas, Streamlit, and ZhipuAI GLM*
