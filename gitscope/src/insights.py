"""
LLM Insights Module for GitHub Profile Analyzer.

This module integrates with ZhipuAI GLM models to generate:
    - Developer profile insights and recommendations
    - Fun facts from activity metrics
    - Professional summaries for recruiters

WHY THIS MATTERS FOR RECRUITERS:
    - Demonstrates LLM integration skills
    - Shows prompt engineering capabilities
    - Implements proper error handling for AI services
    - Creates actionable insights from raw data
    - Shows understanding of AI limitations and safeguards

Usage:
    from src.insights import generate_insights, generate_fun_facts

    insights = generate_insights(developer_profile)
    fun_facts = generate_fun_facts(metrics)
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from src.config import settings


# =============================================================================
# System Prompts
# =============================================================================

# WHY: System prompts define the AI's behavior and constraints
# A well-crafted system prompt ensures consistent, high-quality outputs

SYSTEM_PROMPT_INSIGHTS = """You are an expert software engineering career analyst. Your role is to analyze GitHub profiles and provide actionable insights for developers and recruiters.

Your analysis should be:
1. **Data-driven**: Base all claims on the provided metrics, not assumptions
2. **Balanced**: Acknowledge both strengths and areas for improvement
3. **Actionable**: Provide specific, concrete recommendations
4. **Professional**: Use clear, professional language suitable for reports

Structure your response as:
## Developer Profile Summary
[Brief 2-3 sentence overview]

## Key Strengths
- [List 2-3 specific strengths with evidence from data]

## Areas for Growth
- [List 1-2 areas with specific improvement suggestions]

## Work Style Analysis
[Paragraph on work patterns based on time/consistency data]

## Recommendations
- [2-3 actionable recommendations for career development]

## Fun Observation
[One interesting or surprising insight from the data]

IMPORTANT: Never invent or assume data not provided. If data is insufficient, note limitations honestly."""

SYSTEM_PROMPT_FUN_FACTS = """You are a witty tech culture enthusiast who finds interesting patterns in developer data.

Generate 3 fun, engaging facts about the developer's GitHub activity. Each fact should:
1. Be based on actual data provided
2. Be written in a conversational, engaging tone
3. Include a relevant comparison or context when possible
4. Avoid being judgmental or negative

Format: Return a JSON array of 3 strings, each being one fun fact.
Example: ["They've written enough Python to fill 50 novels!", "Their peak coding hour is midnight - a true night owl!", "They've maintained a 30-day streak - longer than most New Year's resolutions!"]"""


# =============================================================================
# Prompt Building Functions
# =============================================================================

def build_profile_prompt(profile: Dict[str, Any]) -> tuple[str, str]:
    """
    Build the system and user prompts for insights generation.

    WHY: Structured prompts ensure consistent, high-quality outputs.
    Including specific metrics helps the LLM make data-driven insights.

    Args:
        profile: Developer profile dictionary from generate_developer_profile()

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Extract key information
    labels = profile.get("labels", {})
    scores = profile.get("scores", {})
    languages = profile.get("languages", {})
    repos = profile.get("repositories", {})
    commits = profile.get("commits", {})
    wellbeing = profile.get("wellbeing", {})
    strengths = profile.get("strengths", [])
    improvements = profile.get("improvements", [])

    # Build user prompt with structured data
    # WHY: JSON-like structure is easier for LLM to parse
    user_prompt = f"""Analyze the following GitHub developer profile:

## Classification Labels
- Developer Type: {labels.get('developer_type', 'Unknown')}
- Activity Level: {labels.get('activity_level', 'Unknown')}
- Experience Level: {labels.get('experience_level', 'Unknown')}
- Work Style: {labels.get('work_style', 'Unknown')}

## Scores (0-100 scale)
- Overall Score: {scores.get('overall', 0)}
- Consistency Score: {scores.get('consistency', 0)}
- Quality Score: {scores.get('quality', 0)}
- Impact Score: {scores.get('impact', 0)}
- Diversity Score: {scores.get('diversity', 0)}

## Technology Profile
- Primary Language: {languages.get('primary', 'Unknown')}
- Total Languages Used: {languages.get('count', 0)}
- Language Diversity Score: {languages.get('diversity_score', 0)}
- Top Languages: {json.dumps([l.get('language') for l in languages.get('top_languages', [])[:3]])}
- Primary Category: {languages.get('primary_category', 'Unknown')}

## Repository Metrics
- Total Repositories: {repos.get('total', 0)}
- Total Stars: {repos.get('total_stars', 0)}
- Active Repositories: {repos.get('active', 0)}

## Commit Activity
- Total Commits: {commits.get('total', 0)}
- Active Days: {commits.get('active_days', 0)}
- Longest Streak: {commits.get('longest_streak', 0)} days
- Current Streak: {commits.get('current_streak', 0)} days
- Peak Hour: {commits.get('peak_hour', 'Unknown')}:00
- Peak Day: {commits.get('peak_day', 'Unknown')}
- Preferred Time: {commits.get('preferred_time', 'Unknown')}
- Conventional Commit Ratio: {commits.get('conventional_ratio', 0)}%

## Wellbeing Indicators
- Burnout Periods: {wellbeing.get('burnout_periods', 0)}
- Weekend Commit Ratio: {wellbeing.get('weekend_ratio', 0)}%
- Productivity Trend: {wellbeing.get('productivity_trend', 'Unknown')}

## Pre-identified Strengths
{json.dumps(strengths) if strengths else 'None identified'}

## Areas for Improvement
{json.dumps(improvements) if improvements else 'None identified'}

Please provide a comprehensive analysis of this developer profile."""

    return SYSTEM_PROMPT_INSIGHTS, user_prompt


def build_fun_facts_prompt(metrics: Dict[str, Any]) -> tuple[str, str]:
    """
    Build prompts for fun facts generation.

    WHY: Separate prompts for different output types allow for
    different tones and formats.

    Args:
        metrics: Advanced metrics dictionary

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Extract interesting metrics
    total_commits = metrics.get("total_commits", 0)
    active_days = metrics.get("active_days", 0)
    longest_streak = metrics.get("longest_streak", 0)
    peak_hour = metrics.get("peak_hour", "unknown")
    weekend_ratio = metrics.get("weekend_commit_ratio", 0)
    preferred_time = metrics.get("preferred_time", "unknown")

    user_prompt = f"""Generate 3 fun facts about this developer based on their GitHub activity:

- Total Commits: {total_commits}
- Active Days: {active_days}
- Longest Streak: {longest_streak} days
- Peak Coding Hour: {peak_hour}:00
- Weekend Commit Ratio: {weekend_ratio}%
- Preferred Time: {preferred_time}

Remember: Return a JSON array of 3 strings."""

    return SYSTEM_PROMPT_FUN_FACTS, user_prompt


# =============================================================================
# LLM Integration Functions
# =============================================================================

def generate_insights(
    profile: Dict[str, Any],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate AI-powered insights from a developer profile.

    WHY: LLM can synthesize complex data into human-readable insights
    that would be difficult to generate with rule-based systems.

    Args:
        profile: Developer profile dictionary
        model: Override model name (uses settings default if None)
        temperature: Override temperature (uses settings default if None)

    Returns:
        Dictionary containing:
            - report: Generated insight text
            - model: Model used for generation
            - tokens_used: Approximate token count
            - generation_time_ms: Time taken to generate
            - error: Error message if generation failed
    """
    import time

    logger.info("Generating AI insights for developer profile")

    # Initialize result
    result = {
        "report": None,
        "model": None,
        "tokens_used": 0,
        "generation_time_ms": 0,
        "error": None,
    }

    # Check if LLM is configured
    # WHY: Graceful degradation when LLM is not available
    if not settings.is_configured_for_llm:
        result["error"] = "LLM not configured. Set ZHIPUAI_API_KEY environment variable."
        logger.warning(result["error"])
        return result

    # Use provided values or defaults from settings
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature_insights

    result["model"] = model

    try:
        # Import ZhipuAI SDK
        # WHY: Lazy import to avoid import errors when SDK not installed
        from zhipuai import ZhipuAI

        # Initialize client
        client = ZhipuAI(api_key=settings.zhipuai_api_key)

        # Build prompts
        system_prompt, user_prompt = build_profile_prompt(profile)

        # Call API
        # WHY: Timing the API call helps monitor performance
        start_time = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1000,  # Limit response length
        )

        end_time = time.time()
        result["generation_time_ms"] = round((end_time - start_time) * 1000)

        # Extract response
        if response.choices and len(response.choices) > 0:
            result["report"] = response.choices[0].message.content

            # Estimate tokens (approximate)
            # WHY: Usage info might not always be available
            if hasattr(response, "usage") and response.usage:
                result["tokens_used"] = response.usage.total_tokens
            else:
                # Rough estimate: ~4 chars per token
                result["tokens_used"] = (
                    len(system_prompt) + len(user_prompt) + len(result["report"] or "")
                ) // 4

        logger.info(
            f"Generated insights in {result['generation_time_ms']}ms "
            f"using {result['tokens_used']} tokens"
        )

    except ImportError as e:
        result["error"] = f"ZhipuAI SDK not installed: {e}"
        logger.error(result["error"])

    except Exception as e:
        result["error"] = f"Failed to generate insights: {e}"
        logger.error(result["error"])

    return result


def generate_fun_facts(
    metrics: Dict[str, Any],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate fun facts from developer metrics.

    WHY: Fun facts add personality to reports and make them more engaging.
    They're generated with higher temperature for more creative output.

    Args:
        metrics: Advanced metrics dictionary
        model: Override model name
        temperature: Override temperature

    Returns:
        Dictionary containing:
            - facts: List of fun fact strings
            - model: Model used
            - error: Error message if generation failed
    """
    logger.info("Generating fun facts from metrics")

    result = {
        "facts": [],
        "model": None,
        "error": None,
    }

    # Check if LLM is configured
    if not settings.is_configured_for_llm:
        result["error"] = "LLM not configured"
        logger.warning(result["error"])
        return result

    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature_fun_facts
    result["model"] = model

    try:
        from zhipuai import ZhipuAI

        client = ZhipuAI(api_key=settings.zhipuai_api_key)

        system_prompt, user_prompt = build_fun_facts_prompt(metrics)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=500,
        )

        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content

            # Parse JSON array from response
            # WHY: LLM should return JSON array for structured facts
            try:
                # Try to extract JSON from response
                # WHY: LLM might wrap JSON in markdown code blocks
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    facts = json.loads(json_match.group())
                    result["facts"] = facts[:3]  # Limit to 3 facts
                else:
                    # Fallback: treat as single fact
                    result["facts"] = [content]
            except json.JSONDecodeError:
                # Fallback: split by newlines and clean up
                facts = [
                    line.strip().lstrip("- ").lstrip('"').rstrip('"')
                    for line in content.split("\n")
                    if line.strip() and len(line.strip()) > 10
                ]
                result["facts"] = facts[:3]

        logger.info(f"Generated {len(result['facts'])} fun facts")

    except ImportError as e:
        result["error"] = f"ZhipuAI SDK not installed: {e}"
        logger.error(result["error"])

    except Exception as e:
        result["error"] = f"Failed to generate fun facts: {e}"
        logger.error(result["error"])

    return result


# =============================================================================
# Alternative: Generate Insights without LLM (Fallback)
# =============================================================================

def generate_basic_insights(profile: Dict[str, Any]) -> str:
    """
    Generate basic insights without using LLM.

    WHY: Provides a fallback when LLM is not available.
    Uses template-based generation for reliability.

    Args:
        profile: Developer profile dictionary

    Returns:
        Generated insight text
    """
    logger.info("Generating basic insights (no LLM)")

    labels = profile.get("labels", {})
    scores = profile.get("scores", {})
    languages = profile.get("languages", {})
    repos = profile.get("repositories", {})
    commits = profile.get("commits", {})
    strengths = profile.get("strengths", [])
    improvements = profile.get("improvements", [])

    # Build report sections
    sections = []

    # Summary
    sections.append("## Developer Profile Summary")
    summary = (
        f"This developer is a {labels.get('activity_level', 'Unknown').lower()} "
        f"{labels.get('developer_type', 'developer').lower()} "
        f"with {labels.get('experience_level', 'unknown experience').lower()}. "
        f"Their primary technology is {languages.get('primary', 'Unknown')} "
        f"with an overall profile score of {scores.get('overall', 0)}/100."
    )
    sections.append(summary)

    # Strengths
    if strengths:
        sections.append("\n## Key Strengths")
        for strength in strengths:
            sections.append(f"- {strength}")

    # Areas for improvement
    if improvements:
        sections.append("\n## Areas for Growth")
        for improvement in improvements:
            sections.append(f"- {improvement}")

    # Work style
    sections.append("\n## Work Style Analysis")
    work_style = labels.get("work_style", "balanced")
    preferred_time = commits.get("preferred_time", "unknown")
    peak_hour = commits.get("peak_hour", "unknown")

    work_analysis = f"Based on commit patterns, this developer shows a {work_style} work style. "
    if peak_hour and peak_hour != "unknown":
        work_analysis += f"Their peak coding time is around {peak_hour}:00. "
    if preferred_time and preferred_time != "unknown":
        work_analysis += f"They tend to be most productive during the {preferred_time}."

    sections.append(work_analysis)

    # Scores
    sections.append("\n## Profile Scores")
    sections.append(f"- Consistency: {scores.get('consistency', 0)}/100")
    sections.append(f"- Quality: {scores.get('quality', 0)}/100")
    sections.append(f"- Impact: {scores.get('impact', 0)}/100")
    sections.append(f"- Diversity: {scores.get('diversity', 0)}/100")

    return "\n".join(sections)


# =============================================================================
# Export Public Interface
# =============================================================================

__all__ = [
    "generate_insights",
    "generate_fun_facts",
    "generate_basic_insights",
    "build_profile_prompt",
    "build_fun_facts_prompt",
    "SYSTEM_PROMPT_INSIGHTS",
    "SYSTEM_PROMPT_FUN_FACTS",
]
