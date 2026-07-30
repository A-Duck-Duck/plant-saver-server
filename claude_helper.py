"""
Uses the Claude API to "dive deeper" on a diagnosis: explains likely causes,
how to confirm it, and how to fix it.

Requires an ANTHROPIC_API_KEY environment variable set.
"""

import os
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"


def dive_deeper(label: str, confidence: float, all_scores: dict) -> str:
    """
    Sends the model's classification to Claude and asks for a plain-language
    explanation + care steps. Returns Claude's text response.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "Deep dive is unavailable: no ANTHROPIC_API_KEY is set on this server yet. "
            "The diagnosis above is still accurate — this just skips the extra explanation."
        )

    client = Anthropic(api_key=api_key)

    # Show Claude the runner-up classes too, in case confidence is low/ambiguous
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    scores_summary = ", ".join(f"{name}: {score:.0%}" for name, score in sorted_scores)

    prompt = f"""A plant image classifier just diagnosed a houseplant photo as:

Top diagnosis: {label} (confidence: {confidence:.0%})
Full class scores: {scores_summary}

Please give a short, practical breakdown for a student's plant-monitoring app:
1. What "{label}" typically looks like and why the model likely flagged it
2. How to double check this diagnosis yourself (what to look/feel for)
3. 2-3 concrete steps to fix or improve the plant's condition
4. If the confidence is low (under 60%) or another class scored close to the top one, mention that the result is uncertain and suggest what to check to tell them apart

Keep it concise and beginner-friendly, formatted with short headers."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
