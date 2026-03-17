# GitScope

GitScope is a comprehensive tool for analyzing GitHub profiles and repository data. It was built as a data analyst portfolio project to demonstrate end-to-end data pipeline construction, from API collection and storage to advanced metrics and AI-driven insights.

## Project Structure

This repository is organized into two main parts:

- **Frontend (`/`)**: A Next.js application (using Bun and Prisma) that provides the web interface.
- **Backend Analytics (`/gitscope`)**: A Python-based engine that handles the heavy lifting—data collection, transformation with pandas, and LLM analysis.

## Core Features

- **Deep Data Collection**: Fetches commits, repository metadata, and language statistics via the GitHub REST API.
- **Structured Storage**: Uses SQLite with SQLAlchemy for local data persistence and caching.
- **Metric Computation**: Calculates advanced developer metrics like consistency scores, impact ratings, and "burnout" period detection.
- **AI Insights**: Integrates with LLMs (like ZhipuAI) to summarize developer profiles and provide qualitative feedback.
- **Interactive Visualizations**: Terminal output via Rich and a dedicated Streamlit dashboard for visual exploration.

## Tech Stack

- **Frontend**: Next.js, Bun, Tailwind CSS, Prisma, Radix UI.
- **Backend**: Python 3.11+, pandas, SQLAlchemy, Loguru.
- **Visualization**: Streamlit, Plotly, Rich (CLI).
- **Proxy/Deployment**: Caddy.

## Getting Started

### 1. Prerequisites
- [Bun](https://bun.sh/) for the frontend.
- [Python 3.11+](https://www.python.org/) for the analytics engine.
- A GitHub Personal Access Token (to avoid rate limits).

### 2. Frontend Setup
```bash
bun install
bun run db:generate
bun run dev
```

### 3. Backend Setup
Navigate to the `gitscope` directory:
```bash
cd gitscope
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```
*Make sure to add your `GITHUB_TOKEN` and optional `ZHIPUAI_API_KEY` to the `.env` file.*

## Usage

### CLI Analysis
Run the analyzer directly from the terminal:
```bash
python gitscope/main.py <github_username>
```

### Dashboard
Launch the interactive Streamlit UI:
```bash
cd gitscope
streamlit run app/streamlit_app.py
```

## Methodology
The analytical approach behind the metrics (Consistency, Quality, Impact, and Diversity) is documented in [gitscope/ANALYSIS.md](gitscope/ANALYSIS.md).

## License
MIT
